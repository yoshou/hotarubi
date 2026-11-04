#ifndef HOTARUBI_TOOLS_COMPILER_SAFETENSORS_HPP_
#define HOTARUBI_TOOLS_COMPILER_SAFETENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hotarubi::compiler {

enum class DType : uint8_t {
  kF32,
  kBf16,
  kF16,
  kU8,
  kI8,
  kI32,
  kI64,
  kF64,
};

struct TensorInfo {
  DType dtype = DType::kF32;
  std::vector<uint64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

class SafeTensors {
 public:
  static std::expected<SafeTensors, std::string> Open(
      const std::filesystem::path& path);

  [[nodiscard]] bool Contains(std::string_view name) const;
  [[nodiscard]] const TensorInfo& Info(std::string_view name) const;
  [[nodiscard]] std::vector<uint32_t> Float32Bits(std::string_view name) const;
  [[nodiscard]] std::vector<uint8_t> Bytes(std::string_view name) const;
  [[nodiscard]] std::vector<int8_t> Int8s(std::string_view name) const;
  [[nodiscard]] std::vector<std::string> Names() const;
  [[nodiscard]] const std::map<std::string, std::string, std::less<>>&
  Metadata() const {
    return metadata_;
  }

 private:
  [[nodiscard]] std::span<const std::byte> Raw(const TensorInfo& info) const;

  std::filesystem::path path_;
  std::vector<std::byte> data_;
  std::map<std::string, TensorInfo, std::less<>> tensors_;
  std::map<std::string, std::string, std::less<>> metadata_;
};

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_SAFETENSORS_HPP_
