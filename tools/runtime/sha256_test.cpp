#include "hotarubi/sha256.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hotarubi {
namespace {

std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span<const char>(text));
}

TEST(Sha256Test, MatchesTheStandardVectors) {
  EXPECT_EQ(Sha256Hex(Bytes("")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Sha256Hex(Bytes("abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(
      Sha256Hex(Bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomno"
                      "pnopq")),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256Test, SplitUpdatesGiveTheSameDigest) {
  const std::string text(1000, 'a');
  Sha256 whole;
  whole.Update(Bytes(text));
  Sha256 pieces;
  for (size_t i = 0; i < text.size(); i += 7) {
    pieces.Update(Bytes(std::string_view(text).substr(i, 7)));
  }
  EXPECT_EQ(whole.Hex(), pieces.Hex());
  EXPECT_EQ(whole.Hex(),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST(Sha256Test, HashesFilesAndReportsMissingOnes) {
  const std::filesystem::path path = std::filesystem::path(testing::TempDir()) /
                                     ("sha256_" + std::to_string(getpid()));
  std::ofstream(path, std::ios::binary) << "abc";
  const auto digest = Sha256File(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(digest.has_value()) << digest.error();
  EXPECT_EQ(*digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_FALSE(Sha256File(path).has_value());
}

}  // namespace
}  // namespace hotarubi
