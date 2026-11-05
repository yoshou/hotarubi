#ifndef HOTARUBI_REQUIRE_HPP_
#define HOTARUBI_REQUIRE_HPP_

#include <string_view>

namespace hotarubi {

[[noreturn]] void Fail(std::string_view message);

inline void Require(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace hotarubi

#endif  // HOTARUBI_REQUIRE_HPP_
