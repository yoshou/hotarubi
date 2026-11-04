#include "ddr_image.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kCapacity = uint64_t{1} << 16;

TEST(DdrImageTest, AllocatesAlignedRegionsInOrder) {
  DdrImage image(kCapacity);
  EXPECT_EQ(image.Alloc("a", 100), 0U);
  EXPECT_EQ(image.Alloc("b", 10), DdrImage::kPage);
  EXPECT_EQ(image.Alloc("c", 1, 16), DdrImage::kPage + 16);
  EXPECT_EQ(image.End(), DdrImage::kPage + 17);
  ASSERT_EQ(image.Regions().size(), 3U);
  EXPECT_EQ(image.Regions()[1].first, "b");
  EXPECT_EQ(image.Region("c").offset, DdrImage::kPage + 16);
  EXPECT_EQ(image.Region("c").size, 1U);
}

TEST(DdrImageTest, ReuseReturnsTheRegionAlreadyNamed) {
  DdrImage image(kCapacity);
  image.Alloc("a", 100);
  image.SetReuse(true);
  EXPECT_EQ(image.Alloc("a", 60), 0U);
  EXPECT_EQ(image.End(), 100U);
  EXPECT_EQ(image.Alloc("b", 8), DdrImage::kPage);
  EXPECT_EQ(image.Region("a").size, 100U);
  EXPECT_EQ(image.Regions().size(), 2U);
}

TEST(DdrImageTest, WithoutReuseANameMovesToANewRegion) {
  DdrImage image(kCapacity);
  image.Alloc("a", 100);
  EXPECT_EQ(image.Alloc("a", 200), DdrImage::kPage);
  ASSERT_EQ(image.Regions().size(), 1U);
  EXPECT_EQ(image.Region("a").offset, DdrImage::kPage);
  EXPECT_EQ(image.Region("a").size, 200U);
}

TEST(DdrImageTest, DefineNamesARangeInsideTheImage) {
  DdrImage image(kCapacity);
  image.Alloc("scratch", 2 * DdrImage::kPage);
  image.Define("part", DdrImage::kPage, 100);
  ASSERT_EQ(image.Regions().size(), 2U);
  EXPECT_EQ(image.Regions()[1].first, "part");
  EXPECT_EQ(image.Region("part").offset, DdrImage::kPage);
  EXPECT_EQ(image.Region("part").size, 100U);
  EXPECT_EQ(image.End(), 2 * DdrImage::kPage);
}

TEST(DdrImageTest, PutWritesTheBytes) {
  DdrImage image(kCapacity);
  const uint64_t offset = image.Alloc("a", 4);
  image.Put(offset + 1,
            std::vector<std::byte>{std::byte{0x12}, std::byte{0x34}});
  const std::vector<std::byte> expected = {std::byte{0}, std::byte{0x12},
                                           std::byte{0x34}, std::byte{0}};
  EXPECT_EQ(std::vector<std::byte>(image.Bytes().begin(), image.Bytes().end()),
            expected);
}

TEST(DdrImageDeathTest, RejectsARegionPastTheCapacity) {
  DdrImage image(kCapacity);
  EXPECT_DEATH(image.Alloc("a", kCapacity + 1), "exceeds the image");
}

TEST(DdrImageDeathTest, RejectsReuseWithMoreBytes) {
  DdrImage image(kCapacity);
  image.Alloc("a", 100);
  image.SetReuse(true);
  EXPECT_DEATH(image.Alloc("a", 101), "reused with 101 bytes");
}

TEST(DdrImageDeathTest, RejectsDefiningPastTheEndOrATakenName) {
  DdrImage image(kCapacity);
  image.Alloc("a", 100);
  EXPECT_DEATH(image.Define("b", 50, 51), "cannot define region b");
  EXPECT_DEATH(image.Define("a", 0, 10), "cannot define region a");
}

TEST(DdrImageDeathTest, RejectsAnUnknownName) {
  const DdrImage image(kCapacity);
  EXPECT_DEATH(static_cast<void>(image.Region("a")), "no region named a");
}

}  // namespace
}  // namespace hotarubi::compiler
