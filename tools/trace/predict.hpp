#ifndef HOTARUBI_TOOLS_TRACE_PREDICT_HPP_
#define HOTARUBI_TOOLS_TRACE_PREDICT_HPP_

#include <cstddef>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/driver.hpp"
#include "input.hpp"
#include "timing.hpp"

namespace hotarubi::trace {

struct Prediction {
  std::vector<std::byte> records;
  Counters counters;
};

std::expected<Prediction, std::string> Predict(
    const LoadedProgram& loaded, std::string_view stream, bool statistics,
    const iss::TimingParameters& parameters = {});

std::string CountersJson(std::string_view stream, bool statistics,
                         const Counters& counters, std::string_view origin);

}  // namespace hotarubi::trace

#endif  // HOTARUBI_TOOLS_TRACE_PREDICT_HPP_
