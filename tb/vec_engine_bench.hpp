#ifndef HOTARUBI_TB_VEC_ENGINE_BENCH_HPP_
#define HOTARUBI_TB_VEC_ENGINE_BENCH_HPP_

#include <Vvec_engine.h>

#include <cstdint>
#include <memory>
#include <random>

#include "memories.hpp"
#include "vec_engine.hpp"

namespace hotarubi::tb {

struct VecEngineRun {
  uint64_t cycles = 0;
  uint64_t issue_cycles = 0;
  uint64_t drain_cycles = 0;
  uint64_t scalar_cycles = 0;
  uint32_t passes = 0;
};

class VecEngineBench {
 public:
  struct State;

  explicit VecEngineBench(uint32_t deny_percent = 0, uint32_t seed = 1);
  ~VecEngineBench();
  VecEngineBench(const VecEngineBench&) = delete;
  VecEngineBench& operator=(const VecEngineBench&) = delete;

  VecEngineRun Run(const iss::VecCommand& command, iss::Memories& memories);

 private:
  void Tick(iss::Memories& memories);

  std::unique_ptr<Vvec_engine> model_;
  std::unique_ptr<State> state_;
  std::mt19937 generator_;
  uint32_t deny_percent_;
};

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_VEC_ENGINE_BENCH_HPP_
