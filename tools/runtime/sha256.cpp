#include "hotarubi/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace hotarubi {
namespace {

constexpr std::array<uint32_t, 64> kRound = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::array<uint32_t, 8> kInitial = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

uint32_t Load32(std::span<const std::byte, 4> bytes) {
  uint32_t value = 0;
  for (const std::byte byte : bytes) {
    value = (value << 8) | std::to_integer<uint32_t>(byte);
  }
  return value;
}

}  // namespace

Sha256::Sha256() : state_(kInitial) {}

void Sha256::Block(std::span<const std::byte, kBlockBytes> block) {
  std::array<uint32_t, 64> w{};
  for (size_t i = 0; i < 16; ++i) {
    w.at(i) = Load32(block.subspan(i * 4).first<4>());
  }
  for (size_t i = 16; i < 64; ++i) {
    const uint32_t s0 = std::rotr(w.at(i - 15), 7) ^
                        std::rotr(w.at(i - 15), 18) ^ (w.at(i - 15) >> 3);
    const uint32_t s1 = std::rotr(w.at(i - 2), 17) ^
                        std::rotr(w.at(i - 2), 19) ^ (w.at(i - 2) >> 10);
    w.at(i) = w.at(i - 16) + s0 + w.at(i - 7) + s1;
  }
  std::array<uint32_t, 8> v = state_;
  for (size_t i = 0; i < 64; ++i) {
    const uint32_t s1 =
        std::rotr(v[4], 6) ^ std::rotr(v[4], 11) ^ std::rotr(v[4], 25);
    const uint32_t choice = (v[4] & v[5]) ^ (~v[4] & v[6]);
    const uint32_t t1 = v[7] + s1 + choice + kRound.at(i) + w.at(i);
    const uint32_t s0 =
        std::rotr(v[0], 2) ^ std::rotr(v[0], 13) ^ std::rotr(v[0], 22);
    const uint32_t majority = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
    const uint32_t t2 = s0 + majority;
    v[7] = v[6];
    v[6] = v[5];
    v[5] = v[4];
    v[4] = v[3] + t1;
    v[3] = v[2];
    v[2] = v[1];
    v[1] = v[0];
    v[0] = t1 + t2;
  }
  for (size_t i = 0; i < 8; ++i) {
    state_.at(i) += v.at(i);
  }
}

void Sha256::Update(std::span<const std::byte> bytes) {
  length_ += bytes.size();
  while (!bytes.empty()) {
    const size_t take = std::min(bytes.size(), kBlockBytes - buffered_);
    std::ranges::copy(bytes.first(take),
                      buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_));
    buffered_ += take;
    bytes = bytes.subspan(take);
    if (buffered_ == kBlockBytes) {
      Block(buffer_);
      buffered_ = 0;
    }
  }
}

std::array<std::byte, Sha256::kDigestBytes> Sha256::Digest() const {
  Sha256 copy = *this;
  const uint64_t bits = length_ * 8;
  std::array<std::byte, 1> one = {std::byte{0x80}};
  copy.Update(one);
  const std::array<std::byte, 1> zero = {std::byte{0}};
  while (copy.buffered_ != kBlockBytes - 8) {
    copy.Update(zero);
  }
  std::array<std::byte, 8> length{};
  for (size_t i = 0; i < 8; ++i) {
    length.at(i) = static_cast<std::byte>(bits >> (56 - (8 * i)));
  }
  copy.Update(length);
  std::array<std::byte, kDigestBytes> digest{};
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      digest.at((4 * i) + j) =
          static_cast<std::byte>(copy.state_.at(i) >> (24 - (8 * j)));
    }
  }
  return digest;
}

std::string Sha256::Hex() const {
  std::string text;
  for (const std::byte byte : Digest()) {
    text += std::format("{:02x}", std::to_integer<unsigned>(byte));
  }
  return text;
}

std::string Sha256Hex(std::span<const std::byte> bytes) {
  Sha256 hash;
  hash.Update(bytes);
  return hash.Hex();
}

std::expected<std::string, std::string> Sha256File(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("cannot open {}", path.string()));
  }
  Sha256 hash;
  std::vector<char> chunk(size_t{1} << 20);
  while (file) {
    file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    const auto count = static_cast<size_t>(file.gcount());
    hash.Update(std::as_bytes(std::span<const char>(chunk).first(count)));
  }
  return hash.Hex();
}

}  // namespace hotarubi
