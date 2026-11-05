#include "frame_files.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "hotarubi/safetensors.hpp"

namespace hotarubi {
namespace {

std::string ShapeText(const std::vector<uint64_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    text += std::format("{}{}", i == 0 ? "" : ", ", shape[i]);
  }
  return text + "]";
}

}  // namespace

std::expected<std::vector<float>, std::string> ReadFloats(
    const SafeTensors& file, std::string_view name,
    const std::vector<uint64_t>& shape) {
  if (!file.Contains(name)) {
    return std::unexpected(std::format("no tensor {}", name));
  }
  const TensorInfo& info = file.Info(name);
  if (info.dtype != DType::kF32 && info.dtype != DType::kBf16 &&
      info.dtype != DType::kF16) {
    return std::unexpected(
        std::format("tensor {} is not floating point", name));
  }
  if (info.shape != shape) {
    return std::unexpected(std::format("tensor {} has shape {}, not {}", name,
                                       ShapeText(info.shape),
                                       ShapeText(shape)));
  }
  const std::vector<uint32_t> bits = file.Float32Bits(name);
  std::vector<float> values(bits.size());
  std::ranges::transform(bits, values.begin(), [](uint32_t value) {
    return std::bit_cast<float>(value);
  });
  return values;
}

std::expected<std::string, std::string> ReadText(const SafeTensors& file,
                                                 std::string_view key) {
  const auto found = file.Metadata().find(key);
  if (found == file.Metadata().end()) {
    return std::unexpected(std::format("no metadata {}", key));
  }
  return found->second;
}

std::expected<std::vector<std::filesystem::path>, std::string> FramePaths(
    const std::filesystem::path& directory) {
  std::error_code error;
  std::vector<std::filesystem::path> paths;
  for (std::filesystem::directory_iterator it(directory, error);
       !error && it != std::filesystem::directory_iterator();
       it.increment(error)) {
    if (it->path().extension() == ".safetensors") {
      paths.push_back(it->path());
    }
  }
  if (error) {
    return std::unexpected(
        std::format("cannot list {}: {}", directory.string(), error.message()));
  }
  std::ranges::sort(paths);
  return paths;
}

}  // namespace hotarubi
