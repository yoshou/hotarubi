#include "output.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
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

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

Compiled Example() {
  DdrImage image(uint64_t{1} << 20);
  Program program(kBase);
  const uint64_t instructions = image.Alloc("instr", 4096);
  const uint64_t data = image.Alloc("data", 100);
  image.Put(data, std::vector<std::byte>(100, std::byte{0x5A}));
  const Tile tile = program.Alloc(Mem::kFbuf, 1);
  program.Load({.ddr_offset = data, .mem = Mem::kFbuf, .onchip = tile.At()},
               "ld");
  program.Halt();
  program.Finalize();
  image.Put(instructions, program.Bytes());
  std::vector<Stream> streams;
  streams.push_back({.name = "inference",
                     .program = std::move(program),
                     .offset = instructions});
  ReserveTraceRegion(image, streams);
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
  EXPECT_EQ(meta->Find("image_bytes")->Integer(), 8192 + 64);
  EXPECT_EQ(meta->Find("cameras")->Integer(), 2);
  const Json* regions = meta->Find("regions");
  ASSERT_NE(regions, nullptr);
  ASSERT_EQ(regions->Members().size(), 3U);
  EXPECT_EQ(regions->Members()[1].first, "data");
  EXPECT_EQ(regions->Members()[1].second.Items()[0].Integer(), 4096);
  EXPECT_EQ(regions->Members()[1].second.Items()[1].Integer(), 100);
  EXPECT_EQ(regions->Members()[2].first, "trace");
  EXPECT_EQ(regions->Members()[2].second.Items()[0].Integer(), 8192);
  EXPECT_EQ(regions->Members()[2].second.Items()[1].Integer(), 64);
  const Json* listings = meta->Find("listings");
  ASSERT_NE(listings, nullptr);
  ASSERT_EQ(listings->Members().size(), 1U);
  EXPECT_EQ(listings->Members()[0].first, "inference");
  EXPECT_EQ(listings->Members()[0].second.String(), "inference.listing.json");
}

TEST(OutputTest,
     TheTraceRegionHoldsTwoRecordsPerInstructionOfTheLongestStream) {
  Compiled compiled = Example();
  Program second(kBase);
  for (int i = 0; i < 5; ++i) {
    second.Halt();
  }
  second.Finalize();
  compiled.streams.push_back(
      {.name = "second", .program = std::move(second), .offset = 0});
  EXPECT_EQ(TraceRegionBytes(compiled.streams), 5U * 32);
}

TEST(OutputTest, MetadataListsEveryStreamWhenThereAreSeveral) {
  Compiled compiled = Example();
  Program second(kBase);
  second.Halt();
  second.Halt();
  second.Halt();
  second.Finalize();
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
  const auto listing =
      Json::Parse(ReadText(directory / "inference.listing.json"));
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
  EXPECT_EQ(program->regions.at("trace").size, 64U);
  EXPECT_TRUE(program->streams.empty());
  EXPECT_EQ(program->parameters.at("cameras"), 2);
  ASSERT_TRUE(listing.has_value()) << listing.error();
  ASSERT_EQ(listing->Find("instructions")->Items().size(), 2U);
  EXPECT_EQ(listing->Find("instructions")->Items()[0].Find("note")->String(),
            "ld");
}

TEST(OutputTest, TheRuntimeLoadsEveryStreamTheCompilerWrites) {
  const std::filesystem::path directory =
      std::filesystem::path(testing::TempDir()) /
      ("streams_" + std::to_string(getpid()));
  Compiled compiled = Example();
  Program second(kBase);
  second.Halt();
  second.Finalize();
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
