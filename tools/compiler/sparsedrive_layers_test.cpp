#include "sparsedrive_layers.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "isa.hpp"
#include "program.hpp"
#include "sparsedrive_parts.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

constexpr uint64_t kBase = 0x38700000;

TEST(SparseDriveLayersTest, BandsCoverTheRows) {
  const std::vector<Band> bands = Bands(100, 32);
  ASSERT_EQ(bands.size(), 4U);
  EXPECT_EQ(bands[0].i0, 0U);
  EXPECT_EQ(bands[0].m, 32U);
  EXPECT_EQ(bands[3].t, 3U);
  EXPECT_EQ(bands[3].i0, 96U);
  EXPECT_EQ(bands[3].m, 4U);
}

TEST(SparseDriveLayersTest, RowsAreFp32Tokens) {
  const Rows rows{.name = "rows", .n = 10, .cols = 256, .off = 4096};
  EXPECT_EQ(rows.Tok(), 8U);
  EXPECT_EQ(rows.Tb(), TokenBytes(Mem::kFbuf, rows.Tok()));
  EXPECT_EQ(rows.Addr(3), 4096U + (3U * 1024U));
}

TEST(SparseDriveLayersTest, ParametersArePlacesInTheLoadedPbufImage) {
  Pbuf pbuf;
  Keys keys;
  keys["a"] = pbuf.Add(std::vector<uint32_t>(40, 1));
  keys["norm"] = pbuf.Norm(
      {.g = std::vector<uint32_t>(64, 2), .b = std::vector<uint32_t>(64, 3)});
  EXPECT_EQ(keys["a"], 0U);
  EXPECT_EQ(keys["norm"], 2U);
  EXPECT_EQ(pbuf.Words(), 6U);
  Program program(kBase);
  const Tile tile = program.Alloc(Mem::kPbuf, pbuf.Words());
  const Params params(tile, keys);
  EXPECT_EQ(params.At("norm").tile, tile.At().tile);
  EXPECT_EQ(params.At("norm").offset, 2U);
}

TEST(SparseDriveLayersDeathTest, ParametersRejectUnknownKeys) {
  Program program(kBase);
  const Tile tile = program.Alloc(Mem::kPbuf, 1);
  const Keys keys;
  const Params params(tile, keys);
  EXPECT_DEATH(static_cast<void>(params.At("missing")), "no parameter missing");
}

}  // namespace
}  // namespace hotarubi::compiler::sparsedrive
