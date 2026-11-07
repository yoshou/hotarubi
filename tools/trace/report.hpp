#ifndef HOTARUBI_TOOLS_TRACE_REPORT_HPP_
#define HOTARUBI_TOOLS_TRACE_REPORT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "analysis.hpp"
#include "hotarubi/json.hpp"
#include "input.hpp"
#include "timing.hpp"

namespace hotarubi::trace {

inline constexpr uint64_t kDefaultClockHz = 250000000;

struct ReportOptions {
  size_t depth = 1;
  uint64_t clock_hz = kDefaultClockHz;
  size_t gaps = 20;
};

std::string ReportText(const Analysis& analysis, const ReportOptions& options);
Json ReportJson(const Analysis& analysis, const ReportOptions& options);
std::string KernelsText(const Analysis& analysis, const ReportOptions& options);
std::string Perfetto(const Analysis& analysis, uint64_t clock_hz);
std::string DiffText(const Analysis& before, const Analysis& after,
                     const ReportOptions& options);

struct Calibration {
  iss::TimingParameters parameters;
  std::string report;
};

struct CalibrationInput {
  const Analysis* analysis = nullptr;
  const Stream* stream = nullptr;
};

Calibration Calibrate(const std::vector<CalibrationInput>& inputs);

}  // namespace hotarubi::trace

#endif  // HOTARUBI_TOOLS_TRACE_REPORT_HPP_
