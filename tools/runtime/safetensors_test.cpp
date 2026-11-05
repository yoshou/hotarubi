#include "hotarubi/safetensors.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace hotarubi {
namespace {

class SafeTensorsTest : public testing::Test {
 public:
  SafeTensorsTest(const SafeTensorsTest&) = delete;
  SafeTensorsTest& operator=(const SafeTensorsTest&) = delete;
  SafeTensorsTest(SafeTensorsTest&&) = delete;
  SafeTensorsTest& operator=(SafeTensorsTest&&) = delete;
  ~SafeTensorsTest() override { std::filesystem::remove(path_); }

 protected:
  SafeTensorsTest()
      : path_(std::filesystem::path(testing::TempDir()) /
              (std::string(testing::UnitTest::GetInstance()
                               ->current_test_info()
                               ->name()) +
               "_" + std::to_string(getpid()) + ".safetensors")) {}

  void WriteFile(const std::string& header, const std::vector<uint8_t>& data) {
    std::ofstream file(path_, std::ios::binary);
    std::string bytes;
    for (uint32_t i = 0; i < 8; ++i) {
      bytes.push_back(static_cast<char>((header.size() >> (8 * i)) & 0xFFU));
    }
    bytes += header;
    for (const uint8_t value : data) {
      bytes.push_back(static_cast<char>(value));
    }
    file << bytes;
  }

  std::filesystem::path path_;
};

void Append32(std::vector<uint8_t>& data, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    data.push_back(static_cast<uint8_t>(value >> (8 * i)));
  }
}

TEST_F(SafeTensorsTest, ReadsTensorsOfEachSupportedType) {
  std::vector<uint8_t> data;
  Append32(data, 0x3FC00000U);
  Append32(data, 0xC0100000U);
  data.insert(data.end(),
              {0xC0, 0x3F, 0x00, 0x3C, 0x01, 0x80, 0x07, 0xFE, 0x81});
  WriteFile(R"({"__metadata__":{"seed":"11"},)"
            R"("f":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
            R"("b":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]},)"
            R"("h":{"dtype":"F16","shape":[2],"data_offsets":[10,14]},)"
            R"("u":{"dtype":"U8","shape":[1,2],"data_offsets":[14,16]},)"
            R"("i":{"dtype":"I8","shape":[1],"data_offsets":[16,17]}})",
            data);
  auto tensors = SafeTensors::Open(path_);
  ASSERT_TRUE(tensors.has_value()) << tensors.error();
  EXPECT_EQ(tensors->Float32Bits("f"),
            (std::vector<uint32_t>{0x3FC00000U, 0xC0100000U}));
  EXPECT_EQ(tensors->Float32Bits("b"), (std::vector<uint32_t>{0x3FC00000U}));
  EXPECT_EQ(tensors->Float32Bits("h"),
            (std::vector<uint32_t>{0x3F800000U, 0xB3800000U}));
  EXPECT_EQ(tensors->Bytes("u"), (std::vector<uint8_t>{0x07, 0xFE}));
  EXPECT_EQ(tensors->Int8s("i"), (std::vector<int8_t>{-127}));
  EXPECT_EQ(tensors->Info("u").shape, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(tensors->Metadata().at("seed"), "11");
  EXPECT_TRUE(tensors->Contains("f"));
  EXPECT_FALSE(tensors->Contains("g"));
  EXPECT_EQ(tensors->Names().size(), 5U);
}

TEST_F(SafeTensorsTest, RejectsMismatchedOffsets) {
  WriteFile(R"({"f":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",
            std::vector<uint8_t>(8));
  EXPECT_FALSE(SafeTensors::Open(path_).has_value());
}

TEST_F(SafeTensorsTest, RejectsDataBeyondTheFile) {
  WriteFile(R"({"f":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",
            std::vector<uint8_t>(8));
  EXPECT_FALSE(SafeTensors::Open(path_).has_value());
}

TEST_F(SafeTensorsTest, ReportsAMissingFile) {
  EXPECT_FALSE(SafeTensors::Open(path_).has_value());
}

TEST_F(SafeTensorsTest, RejectsMetadataThatIsNotAnObject) {
  WriteFile(R"({"__metadata__":[1]})", {});
  EXPECT_FALSE(SafeTensors::Open(path_).has_value());
}

TEST_F(SafeTensorsTest, WritesTensorsThatReadBack) {
  const std::vector<float> floats = {1.5F, -2.25F, 0.0F};
  const std::vector<uint32_t> words = {7, 0xFFFFFFFFU};
  const std::vector<int64_t> integers = {-3, int64_t{1} << 40};
  ASSERT_TRUE(WriteSafeTensors(path_,
                               {{"f", FloatTensor({3}, floats)},
                                {"u", Uint32Tensor({1, 2}, words)},
                                {"i", Int64Tensor({}, {integers.data(), 1})},
                                {"j", Int64Tensor({2}, integers)}},
                               {{"task", "pick up the cube"}})
                  .has_value());
  std::ifstream file(path_, std::ios::binary);
  std::vector<char> length(8);
  file.read(length.data(), 8);
  uint64_t header = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    header |= static_cast<uint64_t>(static_cast<unsigned char>(length[i]))
              << (8 * i);
  }
  EXPECT_EQ(header % 8, 0U);
  auto tensors = SafeTensors::Open(path_);
  ASSERT_TRUE(tensors.has_value()) << tensors.error();
  EXPECT_EQ(tensors->Float32Bits("f"),
            (std::vector<uint32_t>{0x3FC00000U, 0xC0100000U, 0}));
  EXPECT_EQ(tensors->Info("u").dtype, DType::kU32);
  EXPECT_EQ(tensors->Info("u").shape, (std::vector<uint64_t>{1, 2}));
  EXPECT_TRUE(tensors->Info("i").shape.empty());
  EXPECT_EQ(tensors->Int64s("i"), (std::vector<int64_t>{-3}));
  EXPECT_EQ(tensors->Int64s("j"), integers);
  EXPECT_EQ(tensors->Metadata().at("task"), "pick up the cube");
}

TEST_F(SafeTensorsTest, RejectsATensorWhoseDataDoesNotMatchItsShape) {
  const std::vector<float> floats = {1.0F, 2.0F};
  EXPECT_FALSE(WriteSafeTensors(path_, {{"f", FloatTensor({3}, floats)}}, {})
                   .has_value());
}

}  // namespace
}  // namespace hotarubi
