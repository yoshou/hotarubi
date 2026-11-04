#include "weights.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "exact.hpp"
#include "safetensors.hpp"
#include "tensors.hpp"

namespace hotarubi::compiler {
namespace {

constexpr std::array<int64_t, 8> kHalves = {0, 1, 2, 3, 4, 6, 8, 12};

exact::Real Value(uint8_t code, int32_t e) {
  const exact::Real magnitude =
      exact::Real::Ratio(kHalves[code & 7U], 2).Scaled(e);
  return (code & 8U) != 0 ? magnitude.Negated() : magnitude;
}

bool SameValue(const exact::Real& a, const exact::Real& b) {
  return (a - b).IsZero();
}

Mxfp4Matrix Blocks(std::vector<std::vector<uint8_t>> blocks,
                   std::vector<int32_t> e) {
  Mxfp4Matrix matrix{.rows = static_cast<uint32_t>(blocks.size()),
                     .cols = kBlock,
                     .code = {},
                     .e = std::move(e)};
  for (auto& block : blocks) {
    block.resize(kBlock);
    matrix.code.insert(matrix.code.end(), block.begin(), block.end());
  }
  return matrix;
}

TEST(WeightsTest, CanonicalMxfp4TakesTheScaleFromTheLargestValue) {
  const Mxfp4Matrix in = Blocks(
      {{7, 1, 9}, {5, 1, 0xD}, {3, 0xA}, {1, 9}, {0, 8}}, {4, 2, -1, 5, 6});
  const Mxfp4Matrix out = CanonicalMxfp4(in, 0, in.rows);
  EXPECT_EQ(out.e, (std::vector<int32_t>{4, 1, -3, 2, 0}));
  for (uint32_t r = 0; r < in.rows; ++r) {
    for (uint32_t k = 0; k < kBlock; ++k) {
      const size_t at = (static_cast<size_t>(r) * kBlock) + k;
      EXPECT_TRUE(
          SameValue(Value(in.code[at], in.e[r]), Value(out.code[at], out.e[r])))
          << "row " << r << " element " << k;
    }
  }
  EXPECT_EQ(out.code[0], 7);
  EXPECT_EQ(out.code[kBlock], 7);
  EXPECT_EQ(out.code[kBlock + 2], 0xF);
  EXPECT_EQ(out.code[size_t{3} * kBlock], 6);
  EXPECT_EQ(out.code[(size_t{4} * kBlock) + 1], 0);
}

TEST(WeightsTest, CanonicalMxfp4ScalesNonZeroBlocksAndPadsRows) {
  const Mxfp4Matrix in = Blocks({{6, 2}, {0, 0}}, {3, 9});
  const Mxfp4Matrix out = CanonicalMxfp4(in, -3, 8);
  ASSERT_EQ(out.rows, 8U);
  ASSERT_EQ(out.code.size(), 8U * kBlock);
  EXPECT_EQ(out.e[0], 3 - 3);
  EXPECT_EQ(out.code[0], 6);
  EXPECT_EQ(out.code[1], 2);
  for (uint32_t r = 1; r < out.rows; ++r) {
    EXPECT_EQ(out.e[r], 0);
    for (uint32_t k = 0; k < kBlock; ++k) {
      EXPECT_EQ(out.code[(static_cast<size_t>(r) * kBlock) + k], 0);
    }
  }
}

TEST(WeightsTest, SplitW8RebuildsEveryInt8) {
  Mxint8Matrix matrix{
      .rows = 8, .cols = 32, .q = {}, .e = std::vector<int32_t>(8, -2)};
  for (int32_t q = -128; q < 128; ++q) {
    matrix.q.push_back(static_cast<int8_t>(q));
  }
  const W8Planes planes = SplitW8(matrix);
  EXPECT_EQ(planes.hi.e, matrix.e);
  EXPECT_EQ(planes.lo.e, matrix.e);
  for (size_t i = 0; i < matrix.q.size(); ++i) {
    const int32_t hi = static_cast<int32_t>(planes.hi.code[i] ^ 8U) - 8;
    const int32_t lo = planes.lo.code[i];
    EXPECT_LE(planes.hi.code[i], 15);
    EXPECT_LE(lo, 15);
    EXPECT_EQ((16 * hi) + lo, matrix.q[i]);
  }
}

TEST(WeightsTest, PadsAndSlicesMatrices) {
  const Matrix matrix{.rows = 2, .cols = 3, .bits = {1, 2, 3, 4, 5, 6}};
  const Matrix padded = PadRows(matrix, 4);
  EXPECT_EQ(padded.bits,
            (std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}));
  const Matrix columns = Columns(matrix, 1, 3);
  EXPECT_EQ(columns.cols, 2U);
  EXPECT_EQ(columns.bits, (std::vector<uint32_t>{2, 3, 5, 6}));
  EXPECT_EQ(PadValues(std::vector<uint32_t>{7, 8}, 4),
            (std::vector<uint32_t>{7, 8, 0, 0}));
}

class WeightsFileTest : public testing::Test {
 public:
  WeightsFileTest(const WeightsFileTest&) = delete;
  WeightsFileTest& operator=(const WeightsFileTest&) = delete;
  WeightsFileTest(WeightsFileTest&&) = delete;
  WeightsFileTest& operator=(WeightsFileTest&&) = delete;
  ~WeightsFileTest() override { std::filesystem::remove(path_); }

 protected:
  WeightsFileTest()
      : path_(std::filesystem::path(testing::TempDir()) /
              ("weights_" + std::to_string(getpid()) + ".safetensors")) {}

  void Write(const std::string& header, const std::vector<uint8_t>& data) {
    std::string bytes;
    for (uint32_t i = 0; i < 8; ++i) {
      bytes.push_back(static_cast<char>((header.size() >> (8 * i)) & 0xFFU));
    }
    bytes += header;
    for (const uint8_t value : data) {
      bytes.push_back(static_cast<char>(value));
    }
    std::ofstream(path_, std::ios::binary) << bytes;
  }

  std::filesystem::path path_;
};

TEST_F(WeightsFileTest, ReadsMxfp4CodesWithScalesAndFloatMatrices) {
  std::vector<uint8_t> data(64);
  data[0] = 7;
  data[33] = 0xB;
  data.push_back(127 + 3);
  data.push_back(127 - 4);
  for (const uint8_t byte : {0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0xC0}) {
    data.push_back(byte);
  }
  Write(
      R"({"m.weight.mxfp4":{"dtype":"U8","shape":[2,32],"data_offsets":[0,64]},)"
      R"("m.weight.scale":{"dtype":"U8","shape":[2,1],"data_offsets":[64,66]},)"
      R"("w":{"dtype":"F32","shape":[1,2,1],"data_offsets":[66,74]}})",
      data);
  const auto tensors = SafeTensors::Open(path_);
  ASSERT_TRUE(tensors.has_value()) << tensors.error();
  const Mxfp4Matrix m = ReadMxfp4(*tensors, "m");
  EXPECT_EQ(m.rows, 2U);
  EXPECT_EQ(m.cols, 32U);
  EXPECT_EQ(m.e, (std::vector<int32_t>{3, -4}));
  EXPECT_EQ(m.code[0], 7);
  EXPECT_EQ(m.code[33], 0xB);
  const Matrix w = ReadMatrix(*tensors, "w");
  EXPECT_EQ(w.rows, 1U);
  EXPECT_EQ(w.cols, 2U);
  EXPECT_EQ(w.bits, (std::vector<uint32_t>{0x3F800000U, 0xC0000000U}));
}

}  // namespace
}  // namespace hotarubi::compiler
