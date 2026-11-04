#include "hotarubi/json.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace hotarubi {
namespace {

constexpr int kMaxDepth = 64;
constexpr double kMaxExactInteger = 9007199254740992.0;

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  std::expected<Json, std::string> Document() {
    auto value = Value(0);
    if (!value) {
      return value;
    }
    SkipSpace();
    if (pos_ != text_.size()) {
      return Error("trailing characters");
    }
    return value;
  }

 private:
  [[nodiscard]] std::unexpected<std::string> Error(
      std::string_view what) const {
    return std::unexpected(std::format("JSON: {} at offset {}", what, pos_));
  }

  void SkipSpace() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  bool Consume(std::string_view literal) {
    if (text_.substr(pos_).starts_with(literal)) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  std::expected<Json, std::string> Value(int depth) {
    if (depth > kMaxDepth) {
      return Error("nesting too deep");
    }
    SkipSpace();
    if (pos_ >= text_.size()) {
      return Error("unexpected end");
    }
    const char c = text_[pos_];
    if (c == '{') {
      return ObjectValue(depth);
    }
    if (c == '[') {
      return ArrayValue(depth);
    }
    if (c == '"') {
      auto text = StringValue();
      if (!text) {
        return std::unexpected(text.error());
      }
      return Json(std::move(*text));
    }
    if (Consume("true")) {
      return Json(true);
    }
    if (Consume("false")) {
      return Json(false);
    }
    if (Consume("null")) {
      return Json();
    }
    return NumberValue();
  }

  std::expected<Json, std::string> ObjectValue(int depth) {
    ++pos_;
    Json::Object members;
    SkipSpace();
    if (Consume("}")) {
      return Json(std::move(members));
    }
    while (true) {
      SkipSpace();
      if (pos_ >= text_.size() || text_[pos_] != '"') {
        return Error("expected a member name");
      }
      auto key = StringValue();
      if (!key) {
        return std::unexpected(key.error());
      }
      SkipSpace();
      if (!Consume(":")) {
        return Error("expected ':'");
      }
      auto value = Value(depth + 1);
      if (!value) {
        return value;
      }
      members.emplace_back(std::move(*key), std::move(*value));
      SkipSpace();
      if (Consume("}")) {
        return Json(std::move(members));
      }
      if (!Consume(",")) {
        return Error("expected ',' or '}'");
      }
    }
  }

  std::expected<Json, std::string> ArrayValue(int depth) {
    ++pos_;
    Json::Array items;
    SkipSpace();
    if (Consume("]")) {
      return Json(std::move(items));
    }
    while (true) {
      auto value = Value(depth + 1);
      if (!value) {
        return value;
      }
      items.push_back(std::move(*value));
      SkipSpace();
      if (Consume("]")) {
        return Json(std::move(items));
      }
      if (!Consume(",")) {
        return Error("expected ',' or ']'");
      }
    }
  }

  std::expected<uint32_t, std::string> Hex4() {
    if (pos_ + 4 > text_.size()) {
      return Error("short \\u escape");
    }
    uint32_t value = 0;
    const auto [end, error] = std::from_chars(
        text_.data() + pos_, text_.data() + pos_ + 4, value, 16);
    if (error != std::errc() || end != text_.data() + pos_ + 4) {
      return Error("bad \\u escape");
    }
    pos_ += 4;
    return value;
  }

  static void AppendUtf8(std::string& out, uint32_t code) {
    if (code < 0x80) {
      out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (code >> 18)));
      out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  std::expected<uint32_t, std::string> UnicodeEscape() {
    auto code = Hex4();
    if (!code) {
      return code;
    }
    if (*code >= 0xDC00 && *code < 0xE000) {
      return Error("unpaired surrogate");
    }
    if (*code < 0xD800 || *code >= 0xDC00) {
      return code;
    }
    if (!Consume("\\u")) {
      return Error("unpaired surrogate");
    }
    auto low = Hex4();
    if (!low) {
      return low;
    }
    if (*low < 0xDC00 || *low >= 0xE000) {
      return Error("unpaired surrogate");
    }
    return 0x10000 + ((*code - 0xD800) << 10) + (*low - 0xDC00);
  }

  std::expected<void, std::string> Escape(std::string& out) {
    if (pos_ >= text_.size()) {
      return Error("unterminated string");
    }
    const char escape = text_[pos_++];
    switch (escape) {
      case '"':
      case '\\':
      case '/':
        out.push_back(escape);
        return {};
      case 'b':
        out.push_back('\b');
        return {};
      case 'f':
        out.push_back('\f');
        return {};
      case 'n':
        out.push_back('\n');
        return {};
      case 'r':
        out.push_back('\r');
        return {};
      case 't':
        out.push_back('\t');
        return {};
      case 'u': {
        auto code = UnicodeEscape();
        if (!code) {
          return std::unexpected(code.error());
        }
        AppendUtf8(out, *code);
        return {};
      }
      default:
        return Error("bad escape");
    }
  }

  std::expected<std::string, std::string> StringValue() {
    ++pos_;
    std::string out;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') {
        return out;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        return Error("control character in a string");
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (auto escaped = Escape(out); !escaped) {
        return std::unexpected(escaped.error());
      }
    }
    return Error("unterminated string");
  }

  std::expected<Json, std::string> NumberValue() {
    const size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           ((text_[pos_] >= '0' && text_[pos_] <= '9') || text_[pos_] == '.' ||
            text_[pos_] == 'e' || text_[pos_] == 'E' || text_[pos_] == '+' ||
            text_[pos_] == '-')) {
      ++pos_;
    }
    double value = 0;
    const auto [end, error] =
        std::from_chars(text_.data() + start, text_.data() + pos_, value);
    if (start == pos_ || error != std::errc() || end != text_.data() + pos_) {
      pos_ = start;
      return Error("bad value");
    }
    return Json(value);
  }

  std::string_view text_;
  size_t pos_ = 0;
};

void DumpString(std::string& out, std::string_view text) {
  out.push_back('"');
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += std::format("\\u{:04x}", static_cast<unsigned char>(c));
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void DumpValue(std::string& out, const Json& value) {
  if (value.IsNull()) {
    out += "null";
  } else if (value.IsBool()) {
    out += value.Bool() ? "true" : "false";
  } else if (value.IsNumber()) {
    const double number = value.Number();
    if (std::trunc(number) == number && std::fabs(number) < kMaxExactInteger) {
      out += std::format("{}", static_cast<int64_t>(number));
    } else {
      out += std::format("{}", number);
    }
  } else if (value.IsString()) {
    DumpString(out, value.String());
  } else if (value.IsArray()) {
    out.push_back('[');
    bool first = true;
    for (const Json& item : value.Items()) {
      if (!first) {
        out.push_back(',');
      }
      first = false;
      DumpValue(out, item);
    }
    out.push_back(']');
  } else {
    out.push_back('{');
    bool first = true;
    for (const auto& [key, member] : value.Members()) {
      if (!first) {
        out.push_back(',');
      }
      first = false;
      DumpString(out, key);
      out.push_back(':');
      DumpValue(out, member);
    }
    out.push_back('}');
  }
}

}  // namespace

std::expected<Json, std::string> Json::Parse(std::string_view text) {
  return Parser(text).Document();
}

std::expected<int64_t, std::string> Json::Integer() const {
  if (!IsNumber()) {
    return std::unexpected("JSON: not a number");
  }
  const double value = Number();
  if (std::trunc(value) != value || std::fabs(value) > kMaxExactInteger) {
    return std::unexpected(std::format("JSON: {} is not an integer", value));
  }
  return static_cast<int64_t>(value);
}

const Json* Json::Find(std::string_view key) const {
  if (!IsObject()) {
    return nullptr;
  }
  for (const auto& [name, value] : Members()) {
    if (name == key) {
      return &value;
    }
  }
  return nullptr;
}

std::string Json::Dump() const {
  std::string out;
  DumpValue(out, *this);
  return out;
}

}  // namespace hotarubi
