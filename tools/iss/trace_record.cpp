#include "trace_record.hpp"

#include <algorithm>
#include <cstdint>

#include "hotarubi/isa.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

void Set(Word128& word, uint32_t lsb, uint32_t width, uint64_t value) {
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
  return {.a = Clamp(statistics.a, kStatisticsCounterBits),
          .b = Clamp(statistics.b, kStatisticsCounterBits),
          .c = Clamp(statistics.c, kStatisticsCounterBits),
          .d = Clamp(statistics.d, kStatisticsSmallCounterBits)};
}

Word128 EncodeRecord(const TraceRecord& record) {
  Word128 word;
  Set(word, 0, 32, record.index);
  Set(word, 116, 4, static_cast<uint32_t>(record.type));
  if (record.type == RecordType::kCompletion) {
    Set(word, 32, 32, record.issue);
    Set(word, 64, 32, record.completion);
    Set(word, 96, 4, static_cast<uint32_t>(record.source));
    Set(word, 100, 8, record.opcode);
    Set(word, 108, 4, record.wait);
    return word;
  }
  const Statistics statistics = Saturate(record.statistics);
  Set(word, 32, 4, static_cast<uint32_t>(record.source));
  Set(word, 40, kStatisticsCounterBits, statistics.a);
  Set(word, 64, kStatisticsCounterBits, statistics.b);
  Set(word, 88, kStatisticsCounterBits, statistics.c);
  Set(word, 120, kStatisticsSmallCounterBits, statistics.d);
  return word;
}

TraceRecord DecodeRecord(const Word128& word) {
  TraceRecord record;
  record.index = word.Field(0, 32);
  record.type = static_cast<RecordType>(word.Field(116, 4));
  if (record.type == RecordType::kCompletion) {
    record.issue = word.Field(32, 32);
    record.completion = word.Field(64, 32);
    record.source = static_cast<Source>(word.Field(96, 4));
    record.opcode = word.Field(100, 8);
    record.wait = word.Field(108, 4);
    return record;
  }
  record.source = static_cast<Source>(word.Field(32, 4));
  record.statistics = {.a = word.Field(40, kStatisticsCounterBits),
                       .b = word.Field(64, kStatisticsCounterBits),
                       .c = word.Field(88, kStatisticsCounterBits),
                       .d = word.Field(120, kStatisticsSmallCounterBits)};
  return record;
}

}  // namespace hotarubi::iss
