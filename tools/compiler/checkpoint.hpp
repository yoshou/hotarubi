#ifndef HOTARUBI_TOOLS_COMPILER_CHECKPOINT_HPP_
#define HOTARUBI_TOOLS_COMPILER_CHECKPOINT_HPP_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hotarubi::compiler {

enum class StorageType : uint8_t {
  kFloat,
  kLong,
};

struct CheckpointTensor {
  StorageType type = StorageType::kFloat;
  std::string key;
  uint64_t offset = 0;
  std::vector<uint64_t> shape;
  std::vector<uint64_t> stride;
};

class Checkpoint {
 public:
  static std::expected<Checkpoint, std::string> Open(
      const std::filesystem::path& path);
  static std::expected<Checkpoint, std::string> FromBytes(
      std::vector<std::byte> bytes, std::string origin);

  [[nodiscard]] bool Contains(std::string_view name) const;
  [[nodiscard]] const CheckpointTensor& Info(std::string_view name) const;
  [[nodiscard]] std::vector<uint32_t> Float32Bits(std::string_view name) const;
  [[nodiscard]] std::vector<std::string> Names() const;

 private:
  [[nodiscard]] std::span<const std::byte> Entry(std::string_view key) const;

  std::string origin_;
  std::vector<std::byte> bytes_;
  std::map<std::string, std::pair<uint64_t, uint64_t>, std::less<>> entries_;
  std::map<std::string, CheckpointTensor, std::less<>> tensors_;
};

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_CHECKPOINT_HPP_
