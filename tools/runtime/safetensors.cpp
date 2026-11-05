#include "hotarubi/safetensors.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/json.hpp"
#include "hotarubi/require.hpp"

namespace hotarubi {
namespace {

constexpr uint64_t kMaxHeader = uint64_t{1} << 28;

std::expected<DType, std::string> ParseDType(std::string_view name) {
  static const std::map<std::string, DType, std::less<>> kNames = {
      {"F32", DType::kF32}, {"BF16", DType::kBf16}, {"F16", DType::kF16},
      {"U8", DType::kU8},   {"I8", DType::kI8},     {"I32", DType::kI32},
      {"U32", DType::kU32}, {"I64", DType::kI64},   {"F64", DType::kF64}};
  const auto found = kNames.find(name);
  if (found == kNames.end()) {
    return std::unexpected(std::format("unsupported dtype {}", name));
  }
  return found->second;
}

size_t ElementBytes(DType dtype) {
  switch (dtype) {
    case DType::kU8:
    case DType::kI8:
      return 1;
    case DType::kBf16:
    case DType::kF16:
      return 2;
    case DType::kF32:
    case DType::kI32:
    case DType::kU32:
      return 4;
    case DType::kI64:
    case DType::kF64:
      return 8;
  }
  return 0;
}

uint64_t LoadLittle(std::span<const std::byte> bytes) {
  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
  }
  return value;
}

uint32_t HalfToFloatBits(uint32_t half) {
  const uint32_t sign = (half & 0x8000U) << 16;
  const uint32_t exponent = (half >> 10) & 0x1FU;
  const uint32_t fraction = half & 0x3FFU;
  Require(exponent != 0x1FU, "F16 infinity or NaN");
  if (exponent != 0) {
    return sign | ((exponent + 112) << 23) | (fraction << 13);
  }
  if (fraction == 0) {
    return sign;
  }
  const auto top = static_cast<uint32_t>(std::bit_width(fraction)) - 1;
  return sign | ((top + 103) << 23) | ((fraction << (23 - top)) & 0x7FFFFFU);
}

std::expected<TensorInfo, std::string> ParseTensor(const Json& entry) {
  const Json* dtype = entry.Find("dtype");
  const Json* shape = entry.Find("shape");
  const Json* offsets = entry.Find("data_offsets");
  if (dtype == nullptr || !dtype->IsString() || shape == nullptr ||
      !shape->IsArray() || offsets == nullptr || !offsets->IsArray() ||
      offsets->Items().size() != 2) {
    return std::unexpected("malformed tensor entry");
  }
  auto parsed = ParseDType(dtype->String());
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  TensorInfo info;
  info.dtype = *parsed;
  uint64_t elements = 1;
  for (const Json& dim : shape->Items()) {
    auto value = dim.Integer();
    if (!value || *value < 0) {
      return std::unexpected("bad shape");
    }
    info.shape.push_back(static_cast<uint64_t>(*value));
    elements *= static_cast<uint64_t>(*value);
  }
  auto begin = offsets->Items()[0].Integer();
  auto end = offsets->Items()[1].Integer();
  if (!begin || !end || *begin < 0 || *end < *begin) {
    return std::unexpected("bad data offsets");
  }
  info.begin = static_cast<uint64_t>(*begin);
  info.end = static_cast<uint64_t>(*end);
  if (info.end - info.begin != elements * ElementBytes(info.dtype)) {
    return std::unexpected("data size does not match the shape");
  }
  return info;
}

std::expected<void, std::string> ParseMetadata(
    const Json& entry, std::map<std::string, std::string, std::less<>>& out) {
  if (!entry.IsObject()) {
    return std::unexpected("metadata is not an object");
  }
  for (const auto& [key, value] : entry.Members()) {
    if (value.IsString()) {
      out[key] = value.String();
    }
  }
  return {};
}

}  // namespace

std::expected<SafeTensors, std::string> SafeTensors::Open(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return std::unexpected(std::format("cannot open {}", path.string()));
  }
  const auto size = static_cast<size_t>(file.tellg());
  if (size < 8) {
    return std::unexpected(std::format("{} is too short", path.string()));
  }
  SafeTensors tensors;
  tensors.path_ = path;
  tensors.data_.resize(size);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(tensors.data_.data()),
            static_cast<std::streamsize>(size));
  if (!file) {
    return std::unexpected(std::format("cannot read {}", path.string()));
  }
  const uint64_t header_bytes =
      LoadLittle(std::span<const std::byte>(tensors.data_).first(8));
  if (header_bytes > kMaxHeader || 8 + header_bytes > tensors.data_.size()) {
    return std::unexpected(
        std::format("{} has a bad header length", path.string()));
  }
  auto header = Json::Parse(std::string_view(
      reinterpret_cast<const char*>(tensors.data_.data() + 8), header_bytes));
  if (!header || !header->IsObject()) {
    return std::unexpected(
        std::format("{}: {}", path.string(),
                    header ? "header is not an object" : header.error()));
  }
  const uint64_t data_begin = 8 + header_bytes;
  for (const auto& [name, entry] : header->Members()) {
    if (name == "__metadata__") {
      if (auto parsed = ParseMetadata(entry, tensors.metadata_); !parsed) {
        return std::unexpected(
            std::format("{}: {}", path.string(), parsed.error()));
      }
      continue;
    }
    auto info = ParseTensor(entry);
    if (!info) {
      return std::unexpected(
          std::format("{}: tensor {}: {}", path.string(), name, info.error()));
    }
    info->begin += data_begin;
    info->end += data_begin;
    if (info->end > tensors.data_.size()) {
      return std::unexpected(
          std::format("{}: tensor {} is beyond the file", path.string(), name));
    }
    tensors.tensors_[name] = *info;
  }
  return tensors;
}

bool SafeTensors::Contains(std::string_view name) const {
  return tensors_.contains(name);
}

const TensorInfo& SafeTensors::Info(std::string_view name) const {
  const auto found = tensors_.find(name);
  Require(found != tensors_.end(),
          std::format("{} has no tensor {}", path_.string(), name));
  return found->second;
}

std::span<const std::byte> SafeTensors::Raw(const TensorInfo& info) const {
  return std::span<const std::byte>(data_).subspan(info.begin,
                                                   info.end - info.begin);
}

std::vector<uint32_t> SafeTensors::Float32Bits(std::string_view name) const {
  const TensorInfo& info = Info(name);
  const std::span<const std::byte> raw = Raw(info);
  const size_t width = ElementBytes(info.dtype);
  std::vector<uint32_t> out(raw.size() / width);
  for (size_t i = 0; i < out.size(); ++i) {
    const auto bits =
        static_cast<uint32_t>(LoadLittle(raw.subspan(i * width, width)));
    switch (info.dtype) {
      case DType::kF32:
        out[i] = bits;
        break;
      case DType::kBf16:
        out[i] = bits << 16;
        break;
      case DType::kF16:
        out[i] = HalfToFloatBits(bits);
        break;
      default:
        Fail(std::format("tensor {} is not floating point", name));
    }
  }
  return out;
}

std::vector<uint8_t> SafeTensors::Bytes(std::string_view name) const {
  const TensorInfo& info = Info(name);
  Require(info.dtype == DType::kU8, std::format("tensor {} is not U8", name));
  const std::span<const std::byte> raw = Raw(info);
  std::vector<uint8_t> out(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    out[i] = static_cast<uint8_t>(raw[i]);
  }
  return out;
}

std::vector<int8_t> SafeTensors::Int8s(std::string_view name) const {
  const TensorInfo& info = Info(name);
  Require(info.dtype == DType::kI8, std::format("tensor {} is not I8", name));
  const std::span<const std::byte> raw = Raw(info);
  std::vector<int8_t> out(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    out[i] = static_cast<int8_t>(static_cast<uint8_t>(raw[i]));
  }
  return out;
}

std::vector<int64_t> SafeTensors::Int64s(std::string_view name) const {
  const TensorInfo& info = Info(name);
  Require(info.dtype == DType::kI64, std::format("tensor {} is not I64", name));
  const std::span<const std::byte> raw = Raw(info);
  std::vector<int64_t> out(raw.size() / 8);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<int64_t>(LoadLittle(raw.subspan(i * 8, 8)));
  }
  return out;
}

std::vector<std::string> SafeTensors::Names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& [name, info] : tensors_) {
    names.push_back(name);
  }
  return names;
}

namespace {

std::string_view DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kF32:
      return "F32";
    case DType::kBf16:
      return "BF16";
    case DType::kF16:
      return "F16";
    case DType::kU8:
      return "U8";
    case DType::kI8:
      return "I8";
    case DType::kI32:
      return "I32";
    case DType::kU32:
      return "U32";
    case DType::kI64:
      return "I64";
    case DType::kF64:
      return "F64";
  }
  return {};
}

void AppendLittle(std::vector<std::byte>& out, uint64_t value, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) {
    out.push_back(static_cast<std::byte>(value >> (8 * i)));
  }
}

uint64_t Elements(const std::vector<uint64_t>& shape) {
  uint64_t elements = 1;
  for (const uint64_t dim : shape) {
    elements *= dim;
  }
  return elements;
}

Json Number(uint64_t value) { return Json(static_cast<double>(value)); }

}  // namespace

Tensor FloatTensor(std::vector<uint64_t> shape, std::span<const float> values) {
  Tensor tensor{.dtype = DType::kF32, .shape = std::move(shape), .data = {}};
  tensor.data.reserve(values.size() * 4);
  for (const float value : values) {
    AppendLittle(tensor.data, std::bit_cast<uint32_t>(value), 4);
  }
  return tensor;
}

Tensor Uint32Tensor(std::vector<uint64_t> shape,
                    std::span<const uint32_t> values) {
  Tensor tensor{.dtype = DType::kU32, .shape = std::move(shape), .data = {}};
  tensor.data.reserve(values.size() * 4);
  for (const uint32_t value : values) {
    AppendLittle(tensor.data, value, 4);
  }
  return tensor;
}

Tensor Int64Tensor(std::vector<uint64_t> shape,
                   std::span<const int64_t> values) {
  Tensor tensor{.dtype = DType::kI64, .shape = std::move(shape), .data = {}};
  tensor.data.reserve(values.size() * 8);
  for (const int64_t value : values) {
    AppendLittle(tensor.data, static_cast<uint64_t>(value), 8);
  }
  return tensor;
}

std::expected<void, std::string> WriteSafeTensors(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, Tensor>>& tensors,
    const std::vector<std::pair<std::string, std::string>>& metadata) {
  Json::Object header;
  if (!metadata.empty()) {
    Json::Object entries;
    for (const auto& [key, value] : metadata) {
      entries.emplace_back(key, Json(value));
    }
    header.emplace_back("__metadata__", Json(std::move(entries)));
  }
  uint64_t offset = 0;
  for (const auto& [name, tensor] : tensors) {
    if (tensor.data.size() !=
        Elements(tensor.shape) * ElementBytes(tensor.dtype)) {
      return std::unexpected(
          std::format("tensor {} has {} bytes, which does not match its shape",
                      name, tensor.data.size()));
    }
    Json::Array shape;
    shape.reserve(tensor.shape.size());
    for (const uint64_t dim : tensor.shape) {
      shape.push_back(Number(dim));
    }
    const uint64_t end = offset + tensor.data.size();
    header.emplace_back(
        name,
        Json(Json::Object{
            {"dtype", Json(std::string(DTypeName(tensor.dtype)))},
            {"shape", Json(std::move(shape))},
            {"data_offsets", Json(Json::Array{Number(offset), Number(end)})}}));
    offset = end;
  }
  std::string text = Json(std::move(header)).Dump();
  text.append((8 - (text.size() % 8)) % 8, ' ');
  std::vector<std::byte> length;
  AppendLittle(length, text.size(), 8);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(length.data()),
             static_cast<std::streamsize>(length.size()));
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  for (const auto& [name, tensor] : tensors) {
    file.write(reinterpret_cast<const char*>(tensor.data.data()),
               static_cast<std::streamsize>(tensor.data.size()));
  }
  if (!file) {
    return std::unexpected(std::format("cannot write {}", path.string()));
  }
  return {};
}

}  // namespace hotarubi
