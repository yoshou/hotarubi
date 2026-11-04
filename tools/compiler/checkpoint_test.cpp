#include "checkpoint.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hotarubi::compiler {
namespace {

void Put(std::vector<std::byte>& out, uint64_t value, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFFU));
  }
}

void Put(std::vector<std::byte>& out, std::string_view text) {
  for (const char c : text) {
    out.push_back(static_cast<std::byte>(c));
  }
}

std::vector<std::byte> Zip(
    const std::vector<std::pair<std::string, std::string>>& files) {
  std::vector<std::byte> out;
  std::vector<std::byte> directory;
  for (const auto& [name, data] : files) {
    const size_t local = out.size();
    Put(out, 0x04034b50, 4);
    Put(out, 20, 2);
    Put(out, 0, 2);
    Put(out, 0, 2);
    Put(out, 0, 4);
    Put(out, 0, 4);
    Put(out, data.size(), 4);
    Put(out, data.size(), 4);
    Put(out, name.size(), 2);
    Put(out, 0, 2);
    Put(out, name);
    Put(out, data);
    Put(directory, 0x02014b50, 4);
    Put(directory, 20, 2);
    Put(directory, 20, 2);
    Put(directory, 0, 2);
    Put(directory, 0, 2);
    Put(directory, 0, 4);
    Put(directory, 0, 4);
    Put(directory, data.size(), 4);
    Put(directory, data.size(), 4);
    Put(directory, name.size(), 2);
    Put(directory, 0, 2);
    Put(directory, 0, 2);
    Put(directory, 0, 2);
    Put(directory, 0, 2);
    Put(directory, 0, 4);
    Put(directory, local, 4);
    Put(directory, name);
  }
  const size_t start = out.size();
  out.insert(out.end(), directory.begin(), directory.end());
  Put(out, 0x06054b50, 4);
  Put(out, 0, 4);
  Put(out, files.size(), 2);
  Put(out, files.size(), 2);
  Put(out, directory.size(), 4);
  Put(out, start, 4);
  Put(out, 0, 2);
  return out;
}

std::string Unicode(std::string_view text) {
  std::string out = "X";
  for (size_t i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((text.size() >> (8 * i)) & 0xFFU));
  }
  return out + std::string(text);
}

std::string Tensor(std::string_view storage, int offset,
                   std::pair<int, int> shape, std::pair<int, int> stride) {
  std::string out = "ctorch._utils\n_rebuild_tensor_v2\n(";
  out += storage;
  out += "K";
  out.push_back(static_cast<char>(offset));
  for (const auto& [a, b] : {shape, stride}) {
    out += "K";
    out.push_back(static_cast<char>(a));
    out += "K";
    out.push_back(static_cast<char>(b));
    out += "\x86";
  }
  out +=
      "\x89"
      "ccollections\nOrderedDict\n)RtR";
  return out;
}

std::string Pickle() {
  const std::string storage = "(" + Unicode("storage") +
                              "ctorch\nFloatStorage\n" + Unicode("0") +
                              Unicode("cpu") + "K\x06tQq\x02";
  std::string out = "\x80\x02}q";
  out.push_back('\0');
  out += "(" + Unicode("meta") + "K\x07" + Unicode("state_dict") +
         "ccollections\nOrderedDict\n)Rq\x01(" + Unicode("a") +
         Tensor(storage, 0, {2, 3}, {3, 1}) + Unicode("b") +
         Tensor("h\x02", 0, {3, 2}, {1, 3}) + "uu.";
  return out;
}

std::string Floats(const std::vector<uint32_t>& bits) {
  std::string out;
  for (const uint32_t value : bits) {
    for (size_t i = 0; i < 4; ++i) {
      out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU));
    }
  }
  return out;
}

TEST(CheckpointTest, ReadsTensorsFromTheStateDict) {
  const std::vector<uint32_t> values = {1, 2, 3, 4, 5, 6};
  auto checkpoint =
      Checkpoint::FromBytes(Zip({{"archive/data.pkl", Pickle()},
                                 {"archive/data/0", Floats(values)},
                                 {"archive/version", "3\n"}}),
                            "test");
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error();
  EXPECT_EQ(checkpoint->Names(), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(checkpoint->Info("a").shape, (std::vector<uint64_t>{2, 3}));
  EXPECT_EQ(checkpoint->Float32Bits("a"), values);
  EXPECT_EQ(checkpoint->Float32Bits("b"),
            (std::vector<uint32_t>{1, 4, 2, 5, 3, 6}));
  EXPECT_FALSE(checkpoint->Contains("meta"));
}

TEST(CheckpointTest, RejectsTensorsOutsideTheirStorage) {
  auto checkpoint =
      Checkpoint::FromBytes(Zip({{"archive/data.pkl", Pickle()},
                                 {"archive/data/0", Floats({1, 2, 3})}}),
                            "test");
  EXPECT_FALSE(checkpoint.has_value());
}

TEST(CheckpointTest, RejectsATruncatedPickle) {
  std::string pickle = Pickle();
  pickle.pop_back();
  auto checkpoint = Checkpoint::FromBytes(
      Zip({{"archive/data.pkl", pickle},
           {"archive/data/0", Floats({1, 2, 3, 4, 5, 6})}}),
      "test");
  EXPECT_FALSE(checkpoint.has_value());
}

TEST(CheckpointTest, RejectsSomethingThatIsNotAZip) {
  EXPECT_FALSE(
      Checkpoint::FromBytes(std::vector<std::byte>(100), "test").has_value());
}

}  // namespace
}  // namespace hotarubi::compiler
