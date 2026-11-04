#include <Vfp32_add.h>
#include <Vfp32_mul.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
#include <random>
#include <vector>

#include "fp32_ref.hpp"

namespace hotarubi::tb {
namespace {

constexpr int kLatency = 6;

uint32_t Make(uint32_t sign, uint32_t exponent, uint32_t fraction) {
  return (sign << 31) | (exponent << 23) | (fraction & 0x7FFFFFU);
}

struct Operands {
  uint32_t a = 0;
  uint32_t b = 0;
};

std::vector<Operands> AddCases(uint32_t seed, int count) {
  std::mt19937 rng(seed);
  std::vector<Operands> cases;
  for (int i = 0; i < count; ++i) {
    const uint32_t ea = 20 + (rng() % 200);
    const uint32_t spread = rng() % 4;
    uint32_t eb = ea;
    if (spread == 1) {
      eb = ea - 2 + (rng() % 5);
    } else if (spread == 2) {
      eb = 20 + (rng() % 200);
    }
    uint32_t fb = rng();
    if (rng() % 8 == 0) {
      fb = rng() % 4 == 0 ? 0U : (rng() & 0x7FFF00U);
    }
    cases.push_back(
        {.a = Make(rng() & 1U, ea, rng()), .b = Make(rng() & 1U, eb, fb)});
  }
  return cases;
}

std::vector<Operands> MulCases(uint32_t seed, int count) {
  std::mt19937 rng(seed);
  std::vector<Operands> cases;
  for (int i = 0; i < count; ++i) {
    const uint32_t ea = 1 + (rng() % 253);
    const uint32_t low = ea < 128 ? 128 - ea : 1;
    const uint32_t high = ea < 128 ? 253 : 380 - ea;
    const uint32_t eb = low + (rng() % (high - low + 1));
    cases.push_back(
        {.a = Make(rng() & 1U, ea, rng()), .b = Make(rng() & 1U, eb, rng())});
  }
  return cases;
}

std::vector<Operands> SpecialCases() {
  const uint32_t one = kFp32One;
  const uint32_t two = kFp32Two;
  const uint32_t min_normal = Make(0, 1, 0);
  const uint32_t subnormal = Make(0, 0, 0x400000);
  return {
      {.a = one, .b = Fp32Neg(one)},
      {.a = Fp32Neg(one), .b = Fp32Neg(one)},
      {.a = 0, .b = Fp32Neg(two)},
      {.a = kFp32SignBit, .b = kFp32SignBit},
      {.a = subnormal, .b = one},
      {.a = Fp32Neg(subnormal), .b = 0},
      {.a = Make(0, 1, 1), .b = Fp32Neg(min_normal)},
      {.a = Make(0, 1, 0x7FFFFF), .b = Fp32Neg(Make(0, 1, 0x7FFFFE))},
      {.a = Make(0, 127, 0x7FFFFF), .b = Make(0, 103, 0)},
      {.a = Make(0, 127, 0x7FFFFF), .b = Make(0, 103, 1)},
      {.a = Make(0, 150, 0), .b = Make(1, 100, 1)},
      {.a = Make(0, 63, 0x7FFFFF), .b = Make(0, 63, 0x7FFFFF)},
      {.a = Make(0, 63, 0), .b = Make(0, 63, 0)},
      {.a = Make(0, 62, 0x7FFFFF), .b = Make(0, 64, 0x400000)},
      {.a = Make(1, 30, 0x123456), .b = Make(0, 40, 0x654321)},
  };
}

template <typename Model>
std::vector<uint32_t> RunUnit(Model& model, const std::vector<Operands>& cases,
                              std::vector<int>& latencies) {
  std::vector<uint32_t> results;
  std::deque<uint64_t> issue_cycles;
  uint64_t cycle = 0;
  model.clk_i = 0;
  model.rst_ni = 0;
  model.in_valid_i = 0;
  model.eval();
  model.rst_ni = 1;
  model.eval();
  size_t next = 0;
  while (results.size() < cases.size() && cycle < (2 * cases.size()) + 100) {
    const bool issue = next < cases.size() && (cycle % 5) != 3;
    model.in_valid_i = issue ? 1 : 0;
    if (issue) {
      model.a_i = cases[next].a;
      model.b_i = cases[next].b;
      issue_cycles.push_back(cycle);
      ++next;
    }
    model.clk_i = 1;
    model.eval();
    ++cycle;
    model.clk_i = 0;
    model.eval();
    if (model.out_valid_o != 0) {
      latencies.push_back(static_cast<int>(cycle - issue_cycles.front()));
      issue_cycles.pop_front();
      results.push_back(model.y_o);
    }
  }
  return results;
}

TEST(Fp32UnitTest, AdderMatchesRoundToNearestEvenWithFlushToZero) {
  std::vector<Operands> cases = SpecialCases();
  for (const Operands& extra : AddCases(11, 200000)) {
    cases.push_back(extra);
  }
  Vfp32_add model;
  std::vector<int> latencies;
  const std::vector<uint32_t> results = RunUnit(model, cases, latencies);
  ASSERT_EQ(results.size(), cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    ASSERT_EQ(results[i], Fp32Add(cases[i].a, cases[i].b))
        << std::hex << cases[i].a << " + " << cases[i].b;
    ASSERT_EQ(latencies[i], kLatency);
  }
}

TEST(Fp32UnitTest, MultiplierMatchesRoundToNearestEvenWithFlushToZero) {
  std::vector<Operands> cases = SpecialCases();
  for (const Operands& extra : MulCases(12, 200000)) {
    cases.push_back(extra);
  }
  Vfp32_mul model;
  std::vector<int> latencies;
  const std::vector<uint32_t> results = RunUnit(model, cases, latencies);
  ASSERT_EQ(results.size(), cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    ASSERT_EQ(results[i], Fp32Mul(cases[i].a, cases[i].b))
        << std::hex << cases[i].a << " * " << cases[i].b;
  }
}

}  // namespace
}  // namespace hotarubi::tb
