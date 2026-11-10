#ifndef HOTARUBI_TOOLS_ISS_TRACE_RECORD_HPP_
#define HOTARUBI_TOOLS_ISS_TRACE_RECORD_HPP_

#include <cstdint>

#include "hotarubi/isa.hpp"
#include "word.hpp"

namespace hotarubi::iss {

enum class Source : uint8_t {
  kDma = isa::kEngineDma,
  kGemm = isa::kEngineGemm,
  kVec = isa::kEngineVec,
  kSequencer = isa::trace::kSourceSequencer,
};

enum class RecordType : uint8_t {
  kCompletion = isa::trace::kTypeCompletion,
  kStatistics = isa::trace::kTypeStatistics,
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
