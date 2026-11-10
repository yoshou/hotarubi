#include "vec_rom.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "hotarubi/isa.hpp"
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
  for (uint32_t i = 0; i < vprog::kRomWords; ++i) {
    EXPECT_EQ(ToWord128(rom.words.at(i)), iss::kRom.at(i)) << "word " << i;
  }
}

TEST(VecRomTest, PlacesEveryLibraryProgramAtItsEntry) {
  const Rom rom = Build();
  for (const isa::library::Program& program : isa::library::kPrograms) {
    const uint32_t start = program.entry * isa::vec::kEntryWords;
    const iss::Word128 first = ToWord128(rom.words.at(start));
    EXPECT_NE(first.Field(isa::program::kKind), isa::program::kKindNop)
        << program.name;
    EXPECT_NE(first, iss::Word128{}) << program.name;
  }
}

TEST(VecRomTest, PackageListsEveryWord) {
  const std::string text = Package(Build());
  EXPECT_TRUE(text.starts_with("package vec_rom_pkg;\n"));
  EXPECT_NE(text.find("localparam logic [127:0] Rom [256] = '{"),
            std::string::npos);
  EXPECT_TRUE(text.ends_with("  };\n\nendpackage\n"));
}

}  // namespace
}  // namespace hotarubi::compiler::vec_rom
