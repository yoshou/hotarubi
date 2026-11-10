#include "trace_record.hpp"

#include <algorithm>
#include <cstdint>

#include "hotarubi/isa.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

void Set(Word128& word, isa::Field field, uint64_t value) {
  const uint32_t lsb = field.lsb;
  const uint32_t width = field.width;
  const uint64_t mask = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
  value &= mask;
  if (lsb >= 64) {
    word.high = (word.high & ~(mask << (lsb - 64))) | (value << (lsb - 64));
    return;
  }
  word.low = (word.low & ~(mask << lsb)) | (value << lsb);
  if (lsb + width > 64) {
    const uint32_t spill = 64 - lsb;
    const uint64_t high_mask = mask >> spill;
    word.high = (word.high & ~high_mask) | (value >> spill);
  }
}

uint32_t Clamp(uint32_t value, uint32_t bits) {
  return std::min(value, (uint32_t{1} << bits) - 1);
}

}  // namespace

Source SourceOf(uint32_t opcode) {
  switch (opcode) {
    case isa::kOpDmaLoad:
    case isa::kOpDmaStore:
      return Source::kDma;
    case isa::kOpGemm:
      return Source::kGemm;
    case isa::kOpVec:
      return Source::kVec;
    default:
      return Source::kSequencer;
  }
}

Statistics Saturate(const Statistics& statistics) {
  namespace t = isa::trace;
  return {.a = Clamp(statistics.a, t::kStatisticsA.width),
          .b = Clamp(statistics.b, t::kStatisticsB.width),
          .c = Clamp(statistics.c, t::kStatisticsC.width),
          .d = Clamp(statistics.d, t::kStatisticsD.width)};
}

Word128 EncodeRecord(const TraceRecord& record) {
  namespace t = isa::trace;
  Word128 word;
  Set(word, t::kIndex, record.index);
  Set(word, t::kType, static_cast<uint32_t>(record.type));
  if (record.type == RecordType::kCompletion) {
    Set(word, t::kIssue, record.issue);
    Set(word, t::kCompletion, record.completion);
    Set(word, t::kSource, static_cast<uint32_t>(record.source));
    Set(word, t::kOpcode, record.opcode);
    Set(word, t::kWait, record.wait);
    return word;
  }
  const Statistics statistics = Saturate(record.statistics);
  Set(word, t::kStatisticsSource, static_cast<uint32_t>(record.source));
  Set(word, t::kStatisticsA, statistics.a);
  Set(word, t::kStatisticsB, statistics.b);
  Set(word, t::kStatisticsC, statistics.c);
  Set(word, t::kStatisticsD, statistics.d);
  return word;
}

TraceRecord DecodeRecord(const Word128& word) {
  TraceRecord record;
  namespace t = isa::trace;
  record.index = word.Field(t::kIndex);
  record.type = static_cast<RecordType>(word.Field(t::kType));
  if (record.type == RecordType::kCompletion) {
    record.issue = word.Field(t::kIssue);
    record.completion = word.Field(t::kCompletion);
    record.source = static_cast<Source>(word.Field(t::kSource));
    record.opcode = word.Field(t::kOpcode);
    record.wait = word.Field(t::kWait);
    return record;
  }
  record.source = static_cast<Source>(word.Field(t::kStatisticsSource));
  record.statistics = {.a = word.Field(t::kStatisticsA),
                       .b = word.Field(t::kStatisticsB),
                       .c = word.Field(t::kStatisticsC),
                       .d = word.Field(t::kStatisticsD)};
  return record;
}

}  // namespace hotarubi::iss
