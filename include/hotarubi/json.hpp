#ifndef HOTARUBI_JSON_HPP_
#define HOTARUBI_JSON_HPP_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace hotarubi {

class Json {
 public:
  using Array = std::vector<Json>;
  using Member = std::pair<std::string, Json>;
  using Object = std::vector<Member>;

  Json() = default;
  explicit Json(bool value) : value_(value) {}
  explicit Json(double value) : value_(value) {}
  explicit Json(std::string value) : value_(std::move(value)) {}
  explicit Json(Array value) : value_(std::move(value)) {}
  explicit Json(Object value) : value_(std::move(value)) {}

  static std::expected<Json, std::string> Parse(std::string_view text);

  [[nodiscard]] bool IsNull() const {
    return std::holds_alternative<std::nullptr_t>(value_);
  }
  [[nodiscard]] bool IsBool() const {
    return std::holds_alternative<bool>(value_);
  }
  [[nodiscard]] bool IsNumber() const {
    return std::holds_alternative<double>(value_);
  }
  [[nodiscard]] bool IsString() const {
    return std::holds_alternative<std::string>(value_);
  }
  [[nodiscard]] bool IsArray() const {
    return std::holds_alternative<Array>(value_);
  }
  [[nodiscard]] bool IsObject() const {
    return std::holds_alternative<Object>(value_);
  }

  [[nodiscard]] bool Bool() const { return std::get<bool>(value_); }
  [[nodiscard]] double Number() const { return std::get<double>(value_); }
  [[nodiscard]] const std::string& String() const {
    return std::get<std::string>(value_);
  }
  [[nodiscard]] const Array& Items() const { return std::get<Array>(value_); }
  [[nodiscard]] const Object& Members() const {
    return std::get<Object>(value_);
  }

  [[nodiscard]] std::expected<int64_t, std::string> Integer() const;
  [[nodiscard]] const Json* Find(std::string_view key) const;
  [[nodiscard]] std::string Dump() const;

 private:
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object>
      value_ = nullptr;
};

}  // namespace hotarubi

#endif  // HOTARUBI_JSON_HPP_
