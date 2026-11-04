#include "checkpoint.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "isa.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint32_t kEndOfDirectory = 0x06054b50;
constexpr uint32_t kZip64Locator = 0x07064b50;
constexpr uint32_t kZip64End = 0x06064b50;
constexpr uint32_t kDirectoryEntry = 0x02014b50;
constexpr uint32_t kLocalHeader = 0x04034b50;
constexpr uint16_t kZip64Extra = 0x0001;
constexpr uint32_t kSaturated32 = 0xFFFFFFFFU;
constexpr uint16_t kSaturated16 = 0xFFFFU;
constexpr size_t kEndRecordBytes = 22;
constexpr size_t kMaxComment = 0xFFFF;

template <typename T>
using Result = std::expected<T, std::string>;

class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  [[nodiscard]] bool Has(size_t at, size_t n) const {
    return at <= bytes_.size() && n <= bytes_.size() - at;
  }

  [[nodiscard]] uint64_t Little(size_t at, size_t n) const {
    uint64_t value = 0;
    for (size_t i = 0; i < n; ++i) {
      value |= static_cast<uint64_t>(bytes_[at + i]) << (8 * i);
    }
    return value;
  }

  [[nodiscard]] std::span<const std::byte> Span(size_t at, size_t n) const {
    return bytes_.subspan(at, n);
  }

  [[nodiscard]] size_t Size() const { return bytes_.size(); }

 private:
  std::span<const std::byte> bytes_;
};

std::string Text(std::span<const std::byte> bytes) {
  std::string text;
  text.reserve(bytes.size());
  for (const std::byte b : bytes) {
    text.push_back(static_cast<char>(b));
  }
  return text;
}

struct Directory {
  uint64_t entries = 0;
  uint64_t offset = 0;
};

Result<Directory> FindDirectory(const Reader& zip) {
  if (zip.Size() < kEndRecordBytes) {
    return std::unexpected("too short for a zip archive");
  }
  const size_t lowest = zip.Size() > kEndRecordBytes + kMaxComment
                            ? zip.Size() - kEndRecordBytes - kMaxComment
                            : 0;
  for (size_t at = zip.Size() - kEndRecordBytes + 1; at-- > lowest;) {
    if (zip.Little(at, 4) != kEndOfDirectory) {
      continue;
    }
    Directory directory{.entries = zip.Little(at + 10, 2),
                        .offset = zip.Little(at + 16, 4)};
    if (directory.entries != kSaturated16 && directory.offset != kSaturated32) {
      return directory;
    }
    if (at < 20 || zip.Little(at - 20, 4) != kZip64Locator) {
      return std::unexpected("missing ZIP64 locator");
    }
    const uint64_t end64 = zip.Little(at - 12, 8);
    if (!zip.Has(end64, 56) || zip.Little(end64, 4) != kZip64End) {
      return std::unexpected("bad ZIP64 end record");
    }
    return Directory{.entries = zip.Little(end64 + 32, 8),
                     .offset = zip.Little(end64 + 48, 8)};
  }
  return std::unexpected("no end of central directory");
}

struct Sizes {
  uint64_t compressed = 0;
  uint64_t size = 0;
  uint64_t local = 0;
};

void ApplyZip64(const Reader& zip, size_t extra, size_t extra_bytes,
                Sizes& sizes) {
  size_t at = extra;
  while (at + 4 <= extra + extra_bytes) {
    const auto id = static_cast<uint16_t>(zip.Little(at, 2));
    const size_t n = zip.Little(at + 2, 2);
    size_t field = at + 4;
    if (id == kZip64Extra) {
      for (uint64_t* value : {&sizes.size, &sizes.compressed, &sizes.local}) {
        if (*value == kSaturated32 && field + 8 <= at + 4 + n) {
          *value = zip.Little(field, 8);
          field += 8;
        }
      }
    }
    at += 4 + n;
  }
}

using Entries =
    std::map<std::string, std::pair<uint64_t, uint64_t>, std::less<>>;

Result<Entries> ReadZip(std::span<const std::byte> bytes) {
  const Reader zip(bytes);
  const auto directory = FindDirectory(zip);
  if (!directory) {
    return std::unexpected(directory.error());
  }
  Entries entries;
  uint64_t at = directory->offset;
  for (uint64_t i = 0; i < directory->entries; ++i) {
    if (!zip.Has(at, 46) || zip.Little(at, 4) != kDirectoryEntry) {
      return std::unexpected("bad central directory entry");
    }
    const uint64_t method = zip.Little(at + 10, 2);
    Sizes sizes{.compressed = zip.Little(at + 20, 4),
                .size = zip.Little(at + 24, 4),
                .local = zip.Little(at + 42, 4)};
    const size_t name_bytes = zip.Little(at + 28, 2);
    const size_t extra_bytes = zip.Little(at + 30, 2);
    const size_t comment_bytes = zip.Little(at + 32, 2);
    if (!zip.Has(at + 46, name_bytes + extra_bytes + comment_bytes)) {
      return std::unexpected("truncated central directory");
    }
    const std::string name = Text(zip.Span(at + 46, name_bytes));
    ApplyZip64(zip, at + 46 + name_bytes, extra_bytes, sizes);
    if (method != 0 || sizes.compressed != sizes.size) {
      return std::unexpected(std::format("{} is compressed", name));
    }
    if (!zip.Has(sizes.local, 30) ||
        zip.Little(sizes.local, 4) != kLocalHeader) {
      return std::unexpected(std::format("bad local header for {}", name));
    }
    const uint64_t data = sizes.local + 30 + zip.Little(sizes.local + 26, 2) +
                          zip.Little(sizes.local + 28, 2);
    if (!zip.Has(data, sizes.size)) {
      return std::unexpected(std::format("{} extends beyond the file", name));
    }
    entries.emplace(name, std::pair{data, sizes.size});
    at += 46 + name_bytes + extra_bytes + comment_bytes;
  }
  return entries;
}

struct Value;
using Ref = std::shared_ptr<Value>;
using Items = std::vector<Ref>;
using Members = std::vector<std::pair<Ref, Ref>>;

struct Global {
  std::string module;
  std::string name;
};

struct Storage {
  StorageType type = StorageType::kFloat;
  std::string key;
};

struct Opaque {};

struct Value {
  std::variant<std::monostate, bool, int64_t, double, std::string, Items,
               Members, Global, Storage, CheckpointTensor, Opaque>
      data;
};

Ref Make(auto value) {
  auto ref = std::make_shared<Value>();
  ref->data = std::move(value);
  return ref;
}

class Unpickler {
 public:
  explicit Unpickler(std::span<const std::byte> bytes) : bytes_(bytes) {}

  Result<Ref> Run() {
    while (true) {
      if (at_ >= bytes_.size()) {
        return std::unexpected("pickle ends without STOP");
      }
      const auto op = static_cast<uint8_t>(bytes_[at_++]);
      if (op == '.') {
        if (stack_.size() != 1) {
          return std::unexpected("pickle stack is unbalanced at STOP");
        }
        return stack_.back();
      }
      if (auto step = Step(op); !step) {
        return std::unexpected(step.error());
      }
    }
  }

 private:
  Result<uint64_t> Little(size_t n) {
    if (n > bytes_.size() - at_) {
      return std::unexpected("pickle is truncated");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < n; ++i) {
      value |= static_cast<uint64_t>(bytes_[at_ + i]) << (8 * i);
    }
    at_ += n;
    return value;
  }

  Result<std::string> Bytes(size_t n) {
    if (n > bytes_.size() - at_) {
      return std::unexpected("pickle is truncated");
    }
    std::string text = Text(bytes_.subspan(at_, n));
    at_ += n;
    return text;
  }

  Result<std::string> Line() {
    std::string text;
    while (at_ < bytes_.size() && bytes_[at_] != std::byte{'\n'}) {
      text.push_back(static_cast<char>(bytes_[at_++]));
    }
    if (at_ >= bytes_.size()) {
      return std::unexpected("pickle is truncated");
    }
    ++at_;
    return text;
  }

  Result<Ref> Pop() {
    if (stack_.empty() || (!marks_.empty() && stack_.size() <= marks_.back())) {
      return std::unexpected("pickle stack underflow");
    }
    Ref top = stack_.back();
    stack_.pop_back();
    return top;
  }

  Result<Items> PopMark() {
    if (marks_.empty() || marks_.back() > stack_.size()) {
      return std::unexpected("pickle MARK is missing");
    }
    const size_t mark = marks_.back();
    marks_.pop_back();
    Items items(stack_.begin() + static_cast<std::ptrdiff_t>(mark),
                stack_.end());
    stack_.resize(mark);
    return items;
  }

  Result<void> Push(Result<uint64_t> value, bool sign_extend_32) {
    if (!value) {
      return std::unexpected(value.error());
    }
    const int64_t number =
        sign_extend_32 ? static_cast<int64_t>(static_cast<int32_t>(*value))
                       : static_cast<int64_t>(*value);
    stack_.push_back(Make(number));
    return {};
  }

  Result<void> Memo(Result<uint64_t> index) {
    if (!index) {
      return std::unexpected(index.error());
    }
    if (stack_.empty()) {
      return std::unexpected("pickle memo of an empty stack");
    }
    memo_[*index] = stack_.back();
    return {};
  }

  Result<void> Recall(Result<uint64_t> index) {
    if (!index) {
      return std::unexpected(index.error());
    }
    const auto found = memo_.find(*index);
    if (found == memo_.end()) {
      return std::unexpected("pickle refers to a missing memo entry");
    }
    stack_.push_back(found->second);
    return {};
  }

  Result<void> Tuple(size_t n) {
    if (stack_.size() < n) {
      return std::unexpected("pickle stack underflow");
    }
    Items items(stack_.end() - static_cast<std::ptrdiff_t>(n), stack_.end());
    stack_.resize(stack_.size() - n);
    stack_.push_back(Make(std::move(items)));
    return {};
  }

  Result<void> SetItems(Items items) {
    if (items.size() % 2 != 0 || stack_.empty()) {
      return std::unexpected("pickle SETITEMS has an odd count");
    }
    auto* members = std::get_if<Members>(&stack_.back()->data);
    if (members == nullptr) {
      return std::unexpected("pickle SETITEMS on a non-dict");
    }
    for (size_t i = 0; i < items.size(); i += 2) {
      members->emplace_back(items[i], items[i + 1]);
    }
    return {};
  }

  Result<void> Append(Items items) {
    if (stack_.empty()) {
      return std::unexpected("pickle APPEND on an empty stack");
    }
    auto* list = std::get_if<Items>(&stack_.back()->data);
    if (list == nullptr) {
      return std::unexpected("pickle APPEND on a non-list");
    }
    list->insert(list->end(), items.begin(), items.end());
    return {};
  }

  static Result<Ref> Reduce(const Ref& callable, const Ref& arguments) {
    const auto* global = std::get_if<Global>(&callable->data);
    const auto* args = std::get_if<Items>(&arguments->data);
    if (global == nullptr || args == nullptr) {
      return Make(Opaque{});
    }
    if (global->module == "collections" && global->name == "OrderedDict") {
      return Make(Members{});
    }
    if (global->module != "torch._utils" ||
        global->name != "_rebuild_tensor_v2") {
      return Make(Opaque{});
    }
    if (args->size() < 4) {
      return std::unexpected("_rebuild_tensor_v2 has too few arguments");
    }
    const auto* storage = std::get_if<Storage>(&(*args)[0]->data);
    const auto* offset = std::get_if<int64_t>(&(*args)[1]->data);
    const auto* shape = std::get_if<Items>(&(*args)[2]->data);
    const auto* stride = std::get_if<Items>(&(*args)[3]->data);
    if (storage == nullptr || offset == nullptr || shape == nullptr ||
        stride == nullptr || shape->size() != stride->size() || *offset < 0) {
      return std::unexpected("malformed _rebuild_tensor_v2");
    }
    CheckpointTensor tensor{.type = storage->type,
                            .key = storage->key,
                            .offset = static_cast<uint64_t>(*offset),
                            .shape = {},
                            .stride = {}};
    for (size_t i = 0; i < shape->size(); ++i) {
      const auto* dim = std::get_if<int64_t>(&(*shape)[i]->data);
      const auto* step = std::get_if<int64_t>(&(*stride)[i]->data);
      if (dim == nullptr || step == nullptr || *dim < 0 || *step < 0) {
        return std::unexpected("malformed tensor shape");
      }
      tensor.shape.push_back(static_cast<uint64_t>(*dim));
      tensor.stride.push_back(static_cast<uint64_t>(*step));
    }
    return Make(std::move(tensor));
  }

  static Result<Ref> PersistentLoad(const Ref& id) {
    const auto* items = std::get_if<Items>(&id->data);
    if (items == nullptr || items->size() < 3) {
      return std::unexpected("unsupported persistent id");
    }
    const auto* kind = std::get_if<std::string>(&(*items)[0]->data);
    const auto* type = std::get_if<Global>(&(*items)[1]->data);
    const auto* key = std::get_if<std::string>(&(*items)[2]->data);
    if (kind == nullptr || *kind != "storage" || type == nullptr ||
        key == nullptr || type->module != "torch") {
      return std::unexpected("unsupported persistent id");
    }
    if (type->name == "FloatStorage") {
      return Make(Storage{.type = StorageType::kFloat, .key = *key});
    }
    if (type->name == "LongStorage") {
      return Make(Storage{.type = StorageType::kLong, .key = *key});
    }
    return std::unexpected(std::format("unsupported storage {}", type->name));
  }

  Result<void> Binary(uint8_t op) {
    switch (op) {
      case 'R': {
        const auto arguments = Pop();
        const auto callable = arguments ? Pop() : arguments;
        if (!callable) {
          return std::unexpected(callable.error());
        }
        const auto reduced = Reduce(*callable, *arguments);
        if (!reduced) {
          return std::unexpected(reduced.error());
        }
        stack_.push_back(*reduced);
        return {};
      }
      case 'Q': {
        const auto id = Pop();
        const auto loaded = id ? PersistentLoad(*id) : id;
        if (!loaded) {
          return std::unexpected(loaded.error());
        }
        stack_.push_back(*loaded);
        return {};
      }
      case 'b': {
        const auto state = Pop();
        if (!state) {
          return std::unexpected(state.error());
        }
        return {};
      }
      case 's': {
        const auto value = Pop();
        const auto key = value ? Pop() : value;
        if (!key) {
          return std::unexpected(key.error());
        }
        return SetItems({*key, *value});
      }
      case 'a': {
        const auto value = Pop();
        if (!value) {
          return std::unexpected(value.error());
        }
        return Append({*value});
      }
      default:
        return std::unexpected(
            std::format("unsupported pickle opcode {:#x}", op));
    }
  }

  Result<void> Step(uint8_t op) {
    switch (op) {
      case 0x80:
      case 0x95: {
        const auto skipped = Little(op == 0x80 ? 1 : 8);
        if (!skipped) {
          return std::unexpected(skipped.error());
        }
        return {};
      }
      case '}':
        stack_.push_back(Make(Members{}));
        return {};
      case ']':
      case ')':
        stack_.push_back(Make(Items{}));
        return {};
      case '(':
        marks_.push_back(stack_.size());
        return {};
      case 'N':
        stack_.push_back(Make(std::monostate{}));
        return {};
      case 0x88:
      case 0x89:
        stack_.push_back(Make(op == 0x88));
        return {};
      case 'K':
        return Push(Little(1), false);
      case 'M':
        return Push(Little(2), false);
      case 'J':
        return Push(Little(4), true);
      case 'G': {
        const auto bits = Little(8);
        if (!bits) {
          return std::unexpected(bits.error());
        }
        stack_.push_back(Make(std::bit_cast<double>(std::byteswap(*bits))));
        return {};
      }
      case 'X':
      case 0x8c: {
        const auto n = Little(op == 'X' ? 4 : 1);
        if (!n) {
          return std::unexpected(n.error());
        }
        const auto text = Bytes(*n);
        if (!text) {
          return std::unexpected(text.error());
        }
        stack_.push_back(Make(*text));
        return {};
      }
      case 'c': {
        const auto module = Line();
        const auto name = module ? Line() : module;
        if (!name) {
          return std::unexpected(name.error());
        }
        stack_.push_back(Make(Global{.module = *module, .name = *name}));
        return {};
      }
      case 'q':
        return Memo(Little(1));
      case 'r':
        return Memo(Little(4));
      case 0x94:
        return Memo(memo_.size());
      case 'h':
        return Recall(Little(1));
      case 'j':
        return Recall(Little(4));
      case 't': {
        auto items = PopMark();
        if (!items) {
          return std::unexpected(items.error());
        }
        stack_.push_back(Make(std::move(*items)));
        return {};
      }
      case 0x85:
      case 0x86:
      case 0x87:
        return Tuple(op - 0x84U);
      case 'u':
      case 'e': {
        auto items = PopMark();
        if (!items) {
          return std::unexpected(items.error());
        }
        return op == 'u' ? SetItems(std::move(*items))
                         : Append(std::move(*items));
      }
      default:
        return Binary(op);
    }
  }

  std::span<const std::byte> bytes_;
  size_t at_ = 0;
  std::vector<Ref> stack_;
  std::vector<size_t> marks_;
  std::map<uint64_t, Ref> memo_;
};

const Members* StateDict(const Ref& root) {
  const auto* members = std::get_if<Members>(&root->data);
  if (members == nullptr) {
    return nullptr;
  }
  for (const auto& [key, value] : *members) {
    const auto* name = std::get_if<std::string>(&key->data);
    if (name != nullptr && *name == "state_dict") {
      return std::get_if<Members>(&value->data);
    }
  }
  return members;
}

uint64_t Elements(const CheckpointTensor& tensor) {
  uint64_t count = 1;
  for (const uint64_t dim : tensor.shape) {
    count *= dim;
  }
  return count;
}

}  // namespace

std::expected<Checkpoint, std::string> Checkpoint::Open(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return std::unexpected(std::format("cannot open {}", path.string()));
  }
  std::vector<std::byte> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    return std::unexpected(std::format("cannot read {}", path.string()));
  }
  return FromBytes(std::move(bytes), path.string());
}

std::expected<Checkpoint, std::string> Checkpoint::FromBytes(
    std::vector<std::byte> bytes, std::string origin) {
  Checkpoint checkpoint;
  checkpoint.origin_ = std::move(origin);
  checkpoint.bytes_ = std::move(bytes);
  auto entries = ReadZip(checkpoint.bytes_);
  if (!entries) {
    return std::unexpected(
        std::format("{}: {}", checkpoint.origin_, entries.error()));
  }
  checkpoint.entries_ = std::move(*entries);
  const auto pickle = std::ranges::find_if(
      checkpoint.entries_,
      [](const auto& entry) { return entry.first.ends_with("/data.pkl"); });
  if (pickle == checkpoint.entries_.end()) {
    return std::unexpected(std::format("{}: no data.pkl", checkpoint.origin_));
  }
  const std::string prefix = pickle->first.substr(
      0, pickle->first.size() - std::string_view("data.pkl").size());
  const auto root =
      Unpickler(std::span<const std::byte>(checkpoint.bytes_)
                    .subspan(pickle->second.first, pickle->second.second))
          .Run();
  if (!root) {
    return std::unexpected(
        std::format("{}: {}", checkpoint.origin_, root.error()));
  }
  const Members* state = StateDict(*root);
  if (state == nullptr) {
    return std::unexpected(
        std::format("{}: no state dict", checkpoint.origin_));
  }
  for (const auto& [key, value] : *state) {
    const auto* name = std::get_if<std::string>(&key->data);
    const auto* tensor = std::get_if<CheckpointTensor>(&value->data);
    if (name == nullptr || tensor == nullptr) {
      continue;
    }
    CheckpointTensor info = *tensor;
    info.key = prefix + "data/" + info.key;
    const auto data = checkpoint.entries_.find(info.key);
    const uint64_t width = info.type == StorageType::kFloat ? 4 : 8;
    uint64_t last = info.offset;
    for (size_t i = 0; i < info.shape.size(); ++i) {
      if (info.shape[i] != 0) {
        last += (info.shape[i] - 1) * info.stride[i];
      }
    }
    if (data == checkpoint.entries_.end() ||
        (Elements(info) != 0 && (last + 1) * width > data->second.second)) {
      return std::unexpected(std::format("{}: tensor {} is outside its storage",
                                         checkpoint.origin_, *name));
    }
    checkpoint.tensors_.emplace(*name, std::move(info));
  }
  return checkpoint;
}

bool Checkpoint::Contains(std::string_view name) const {
  return tensors_.contains(name);
}

const CheckpointTensor& Checkpoint::Info(std::string_view name) const {
  const auto found = tensors_.find(name);
  Require(found != tensors_.end(),
          std::format("{} has no tensor {}", origin_, name));
  return found->second;
}

std::span<const std::byte> Checkpoint::Entry(std::string_view key) const {
  const auto& [offset, size] = entries_.find(key)->second;
  return std::span<const std::byte>(bytes_).subspan(offset, size);
}

std::vector<uint32_t> Checkpoint::Float32Bits(std::string_view name) const {
  const CheckpointTensor& info = Info(name);
  Require(info.type == StorageType::kFloat,
          std::format("tensor {} is not float", name));
  const std::span<const std::byte> data = Entry(info.key);
  const uint64_t count = Elements(info);
  std::vector<uint32_t> out;
  out.reserve(count);
  std::vector<uint64_t> index(info.shape.size());
  for (uint64_t n = 0; n < count; ++n) {
    uint64_t element = info.offset;
    for (size_t d = 0; d < index.size(); ++d) {
      element += index[d] * info.stride[d];
    }
    uint32_t bits = 0;
    for (uint32_t b = 0; b < 4; ++b) {
      bits |= static_cast<uint32_t>(data[(element * 4) + b]) << (8 * b);
    }
    out.push_back(bits);
    for (size_t d = index.size(); d-- > 0;) {
      if (++index[d] < info.shape[d]) {
        break;
      }
      index[d] = 0;
    }
  }
  return out;
}

std::vector<std::string> Checkpoint::Names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& [name, info] : tensors_) {
    names.push_back(name);
  }
  return names;
}

}  // namespace hotarubi::compiler
