#ifndef HOTARUBI_TOOLS_ISS_TRACE_RECORD_HPP_
#define HOTARUBI_TOOLS_ISS_TRACE_RECORD_HPP_

#include <cstdint>

#include "word.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kTraceRecordBytes = 16;
inline constexpr uint32_t kStatisticsCounterBits = 24;
inline constexpr uint32_t kStatisticsSmallCounterBits = 8;

enum class Source : uint8_t {
  kDma = 0,
  kGemm = 1,
  kVec = 2,
  kSequencer = 3,
};

enum class RecordType : uint8_t {
  kCompletion = 0,
  kStatistics = 1,
};

struct Statistics {
  uint32_t a = 0;
  uint32_t b = 0;
  uint32_t c = 0;
  uint32_t d = 0;

  friend constexpr bool operator==(const Statistics&,
                                   const Statistics&) = default;
};

struct TraceRecord {
  uint32_t index = 0;
  uint32_t issue = 0;
  uint32_t completion = 0;
  Source source = Source::kSequencer;
  uint32_t opcode = 0;
  uint32_t wait = 0;
  RecordType type = RecordType::kCompletion;
  Statistics statistics;

  friend constexpr bool operator==(const TraceRecord&,
                                   const TraceRecord&) = default;
};

Source SourceOf(uint32_t opcode);
Word128 EncodeRecord(const TraceRecord& record);
TraceRecord DecodeRecord(const Word128& word);
Statistics Saturate(const Statistics& statistics);

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_TRACE_RECORD_HPP_
