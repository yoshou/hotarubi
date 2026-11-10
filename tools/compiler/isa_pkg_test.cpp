#include "isa_pkg.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>

namespace hotarubi::compiler::isa_pkg {
namespace {

std::string ReadText(const char* path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

TEST(IsaPkgTest, MatchesTheRtlPackage) {
  EXPECT_EQ(Package(), ReadText(HOTARUBI_ISA_PKG));
}

TEST(IsaPkgTest, DefinesEveryEnumOnce) {
  const std::string text = Package();
  for (const char* type :
       {"mm_e", "pa_e", "c0_e", "b_e", "nl_e", "po_e", "red_e", "wr_e", "sf_e",
        "cmp_e", "cl_e", "sel_e", "ia_e", "rm_e"}) {
    const std::string declaration = std::string("} ") + type + ";";
    EXPECT_EQ(text.find(declaration), text.rfind(declaration)) << type;
    EXPECT_NE(text.find(declaration), std::string::npos) << type;
  }
}

}  // namespace
}  // namespace hotarubi::compiler::isa_pkg
