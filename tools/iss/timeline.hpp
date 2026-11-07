#ifndef HOTARUBI_TOOLS_ISS_TIMELINE_HPP_
#define HOTARUBI_TOOLS_ISS_TIMELINE_HPP_

#include <array>
#include <cstdint>

#include "timing.hpp"
#include "trace_record.hpp"

namespace hotarubi::iss {

struct Span {
  uint64_t issue = 0;
  uint64_t start = 0;
  uint64_t completion = 0;
};

class Timeline {
 public:
  static constexpr int kEngines = 3;

  explicit Timeline(const TimingParameters& parameters = {})
      : start_cycles_(parameters.start_cycles) {}

  void Start();
  Span Issue(Source source, uint32_t wait, uint64_t cost,
             uint64_t earliest_start = 0);

  [[nodiscard]] uint64_t EngineFree(Source source) const {
    return free_.at(static_cast<size_t>(source));
  }
  [[nodiscard]] uint64_t Cycles() const;
  [[nodiscard]] uint64_t Busy(Source source) const {
    return busy_.at(static_cast<size_t>(source));
  }
  [[nodiscard]] uint64_t Issued() const { return issued_; }
  [[nodiscard]] uint64_t Stall() const;
  [[nodiscard]] uint64_t Starve() const { return start_cycles_; }

 private:
  uint64_t start_cycles_;
  uint64_t last_issue_ = 0;
  uint64_t issued_ = 0;
  std::array<uint64_t, kEngines> free_{};
  std::array<uint64_t, kEngines> busy_{};
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_TIMELINE_HPP_
