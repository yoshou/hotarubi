#include "hotarubi/require.hpp"

#include <cstdio>
#include <cstdlib>
#include <print>
#include <string_view>

namespace hotarubi {

void Fail(std::string_view message) {
  std::println(stderr, "error: {}", message);
  std::abort();
}

}  // namespace hotarubi
