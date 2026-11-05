#include "output.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/runtime.hpp"
#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kBase = 0x38700000;

Compiled Example() {
  DdrImage image(uint64_t{1} << 20);
  Program program(kBase);
  const uint64_t instructions = image.Alloc("instr", 4096);
  const uint64_t data = image.Alloc("data", 100);
  image.Put(data, std::vector<std::byte>(100, std::byte{0x5A}));
  program.Load({.ddr_offset = data, .mem = Mem::kFbuf, .onchip = 0}, "ld");
  program.Halt();
  image.Put(instructions, program.Bytes());
  std::vector<Stream> streams;
  streams.push_back({.name = "inference",
                     .program = std::move(program),
                     .offset = instructions});
  return {.streams = std::move(streams),
          .image = std::move(image),
          .parameters = {{"cameras", 2}}};
}

TEST(OutputTest, MetadataDescribesTheImage) {
  const Compiled compiled = Example();
  const auto meta = Json::Parse(MetaJson(compiled));
  ASSERT_TRUE(meta.has_value()) << meta.error();
  EXPECT_EQ(meta->Find("base")->Integer(), static_cast<int64_t>(kBase));
  EXPECT_EQ(meta->Find("instr_off")->Integer(), 0);
  EXPECT_EQ(meta->Find("n_instr")->Integer(), 2);
  EXPECT_EQ(meta->Find("image_bytes")->Integer(), 4096 + 100);
  EXPECT_EQ(meta->Find("cameras")->Integer(), 2);
  const Json* regions = meta->Find("regions");
  ASSERT_NE(regions, nullptr);
  ASSERT_EQ(regions->Members().size(), 2U);
  EXPECT_EQ(regions->Members()[1].first, "data");
  EXPECT_EQ(regions->Members()[1].second.Items()[0].Integer(), 4096);
  EXPECT_EQ(regions->Members()[1].second.Items()[1].Integer(), 100);
}

TEST(OutputTest, MetadataListsEveryStreamWhenThereAreSeveral) {
  Compiled compiled = Example();
  Program second(kBase);
  second.Halt();
  second.Halt();
  second.Halt();
  const uint64_t offset = compiled.image.Alloc("instr.second", 48);
  compiled.image.Put(offset, second.Bytes());
  compiled.streams.push_back(
      {.name = "second", .program = std::move(second), .offset = offset});
  const auto meta = Json::Parse(MetaJson(compiled));
  ASSERT_TRUE(meta.has_value()) << meta.error();
  EXPECT_EQ(meta->Find("instr_off")->Integer(), 0);
  EXPECT_EQ(meta->Find("n_instr")->Integer(), 2);
  const Json* streams = meta->Find("streams");
  ASSERT_NE(streams, nullptr);
  ASSERT_EQ(streams->Members().size(), 2U);
  EXPECT_EQ(streams->Members()[0].first, "inference");
  EXPECT_EQ(streams->Members()[0].second.Items()[0].Integer(), 0);
  EXPECT_EQ(streams->Members()[0].second.Items()[1].Integer(), 2);
  EXPECT_EQ(streams->Members()[1].first, "second");
  EXPECT_EQ(streams->Members()[1].second.Items()[0].Integer(),
            static_cast<int64_t>(offset));
  EXPECT_EQ(streams->Members()[1].second.Items()[1].Integer(), 3);
}

TEST(OutputTest, ASingleStreamHasNoStreamTable) {
  const auto meta = Json::Parse(MetaJson(Example()));
  ASSERT_TRUE(meta.has_value()) << meta.error();
  EXPECT_EQ(meta->Find("streams"), nullptr);
}

TEST(OutputTest, TheRuntimeLoadsWhatTheCompilerWrites) {
  const std::filesystem::path directory =
      std::filesystem::path(testing::TempDir()) /
      ("output_" + std::to_string(getpid()));
  const Compiled compiled = Example();
  ASSERT_TRUE(WriteCompiled(directory, compiled).has_value());
  const auto program = LoadProgram(directory);
  std::filesystem::remove_all(directory);
  ASSERT_TRUE(program.has_value()) << program.error();
  EXPECT_EQ(program->base, kBase);
  EXPECT_EQ(program->instruction_offset, 0U);
  EXPECT_EQ(program->instruction_count, 2U);
  ASSERT_EQ(program->image.size(), compiled.image.Bytes().size());
  EXPECT_TRUE(std::equal(program->image.begin(), program->image.end(),
                         compiled.image.Bytes().begin()));
  EXPECT_EQ(program->regions.at("data").offset, 4096U);
  EXPECT_EQ(program->regions.at("data").size, 100U);
  EXPECT_TRUE(program->streams.empty());
  EXPECT_EQ(program->parameters.at("cameras"), 2);
}

TEST(OutputTest, TheRuntimeLoadsEveryStreamTheCompilerWrites) {
  const std::filesystem::path directory =
      std::filesystem::path(testing::TempDir()) /
      ("streams_" + std::to_string(getpid()));
  Compiled compiled = Example();
  Program second(kBase);
  second.Halt();
  const uint64_t offset = compiled.image.Alloc("instr.second", 16);
  compiled.image.Put(offset, second.Bytes());
  compiled.streams.push_back(
      {.name = "second", .program = std::move(second), .offset = offset});
  ASSERT_TRUE(WriteCompiled(directory, compiled).has_value());
  const auto program = LoadProgram(directory);
  std::filesystem::remove_all(directory);
  ASSERT_TRUE(program.has_value()) << program.error();
  ASSERT_EQ(program->streams.size(), 2U);
  EXPECT_EQ(program->streams.at("inference").count, 2U);
  EXPECT_EQ(program->streams.at("second").offset, offset);
  EXPECT_EQ(program->streams.at("second").count, 1U);
}

}  // namespace
}  // namespace hotarubi::compiler
