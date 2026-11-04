#include "hotarubi/json.hpp"

#include <gtest/gtest.h>

#include <string>

namespace hotarubi {
namespace {

TEST(JsonTest, ParsesNestedValues) {
  const auto json = Json::Parse(
      R"( {"base": 948961280, "steps": 10, "ok": true, "none": null,
           "ratio": -1.5e-3, "regions": {"instr": [0, 4194304]},
           "names": ["x_t.1", "v_t.0"]} )");
  ASSERT_TRUE(json.has_value()) << json.error();
  EXPECT_EQ(json->Find("base")->Integer().value(), 948961280);
  EXPECT_EQ(json->Find("steps")->Integer().value(), 10);
  EXPECT_TRUE(json->Find("ok")->Bool());
  EXPECT_TRUE(json->Find("none")->IsNull());
  EXPECT_DOUBLE_EQ(json->Find("ratio")->Number(), -1.5e-3);
  const Json* instr = json->Find("regions")->Find("instr");
  ASSERT_NE(instr, nullptr);
  ASSERT_EQ(instr->Items().size(), 2U);
  EXPECT_EQ(instr->Items()[1].Integer().value(), 4194304);
  EXPECT_EQ(json->Find("names")->Items()[0].String(), "x_t.1");
  EXPECT_EQ(json->Find("missing"), nullptr);
  EXPECT_FALSE(json->Find("ratio")->Integer().has_value());
}

TEST(JsonTest, DecodesEscapes) {
  const std::string u = std::string(1, '\\') + "u";
  const auto json = Json::Parse(R"("a\"b\\c\/d\n)" + u + "00e9" + u + "d83d" +
                                u + "de00" + "\"");
  ASSERT_TRUE(json.has_value()) << json.error();
  EXPECT_EQ(json->String(), "a\"b\\c/d\n\xC3\xA9\xF0\x9F\x98\x80");
}

TEST(JsonTest, RejectsMalformedText) {
  const std::string u = std::string(1, '\\') + "u";
  for (const std::string& text :
       {std::string("{"), std::string("[1,]"), std::string(R"({"a" 1})"),
        std::string(R"("unterminated)"), std::string(R"("\x")"),
        "\"" + u + "d83d\"", std::string("1 2"), std::string("tru"),
        std::string("-"), std::string("{\"a\":1,}")}) {
    EXPECT_FALSE(Json::Parse(text).has_value()) << text;
  }
}

TEST(JsonTest, DumpsWhatItParses) {
  const std::string text =
      R"({"a":[1,2.5,"x\n"],"b":{"c":false,"d":null},"e":-3})";
  const auto json = Json::Parse(text);
  ASSERT_TRUE(json.has_value()) << json.error();
  EXPECT_EQ(json->Dump(), text);
}

}  // namespace
}  // namespace hotarubi
