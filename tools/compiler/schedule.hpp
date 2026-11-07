#ifndef HOTARUBI_TOOLS_COMPILER_SCHEDULE_HPP_
#define HOTARUBI_TOOLS_COMPILER_SCHEDULE_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "program.hpp"
#include "timing.hpp"

namespace hotarubi::compiler {

struct ScheduleOptions {
  uint32_t window = 64;
  iss::TimingParameters timing = {};
};

[[nodiscard]] bool Conflicts(const Program& program, size_t earlier,
                             size_t later);
[[nodiscard]] std::vector<size_t> ScheduleOrder(const Program& program,
                                                const ScheduleOptions& options);
void Schedule(Program& program, const ScheduleOptions& options);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_SCHEDULE_HPP_
