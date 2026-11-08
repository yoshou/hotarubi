#include "vec_rom.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "rom.hpp"
#include "word.hpp"

namespace hotarubi::compiler::vec_rom {
namespace {

iss::Word128 ToWord128(const Instruction& instruction) {
  return {.low = instruction.words.at(0) |
                 (uint64_t{instruction.words.at(1)} << 32),
          .high = instruction.words.at(2) |
                  (uint64_t{instruction.words.at(3)} << 32)};
}

TEST(VecRomTest, MatchesTheRomOfTheRtlPackage) {
  const Rom rom = Build();
  for (uint32_t op = 0; op < kOperations; ++op) {
    EXPECT_EQ(rom.starts.at(op), iss::kRomStart.at(op)) << "operation " << op;
  }
  for (uint32_t i = 0; i < vprog::kRomWords; ++i) {
    EXPECT_EQ(ToWord128(rom.words.at(i)), iss::kRom.at(i)) << "word " << i;
  }
}

TEST(VecRomTest, PackageListsTheStartsAndEveryWord) {
  const std::string text = Package(Build());
  EXPECT_TRUE(text.starts_with("package vec_rom_pkg;\n"));
  EXPECT_NE(text.find("localparam logic [9:0] RomPc [16] = '{"),
            std::string::npos);
  EXPECT_NE(text.find("localparam logic [127:0] Rom [256] = '{"),
            std::string::npos);
  EXPECT_TRUE(text.ends_with("  };\n\nendpackage\n"));
}

}  // namespace
}  // namespace hotarubi::compiler::vec_rom
