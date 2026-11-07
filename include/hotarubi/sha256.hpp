#ifndef HOTARUBI_SHA256_HPP_
#define HOTARUBI_SHA256_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace hotarubi {

class Sha256 {
 public:
  static constexpr size_t kDigestBytes = 32;
  static constexpr size_t kBlockBytes = 64;

  Sha256();

  void Update(std::span<const std::byte> bytes);
  [[nodiscard]] std::array<std::byte, kDigestBytes> Digest() const;
  [[nodiscard]] std::string Hex() const;

 private:
  void Block(std::span<const std::byte, kBlockBytes> block);

  std::array<uint32_t, 8> state_;
  std::array<std::byte, kBlockBytes> buffer_{};
  size_t buffered_ = 0;
  uint64_t length_ = 0;
};

std::string Sha256Hex(std::span<const std::byte> bytes);
std::expected<std::string, std::string> Sha256File(
    const std::filesystem::path& path);

}  // namespace hotarubi

#endif  // HOTARUBI_SHA256_HPP_
