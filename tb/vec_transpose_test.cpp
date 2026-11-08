#include <Vvec_transpose.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace hotarubi::tb {
namespace {

using Word = std::array<uint32_t, 32>;

struct Transpose {
  uint32_t src = 128;
  uint32_t dst = 2048;
  uint32_t tiles = 1;
  uint32_t src_tile_stride = 8;
  uint32_t src_word_stride = 1;
  uint32_t dst_tile_stride = 8;
  uint32_t dst_word_stride = 1;
  bool zip4 = false;
};

struct Response {
  uint32_t cycle = 0;
  Word data{};
};

std::vector<Word> RandomMemory(uint32_t seed) {
  std::mt19937 generator(seed);
  std::vector<Word> memory(4096);
  for (auto& word : memory) {
    for (auto& value : word) {
      value = static_cast<uint32_t>(generator());
    }
  }
  return memory;
}

std::vector<Word> Transposed(const std::vector<Word>& memory,
                             const Transpose& config) {
  auto expected = memory;
  for (uint32_t q = 0; q < config.tiles; ++q) {
    for (uint32_t row = 0; row < 8; ++row) {
      const uint32_t dst = (config.dst + (q * config.dst_tile_stride) +
                            (row * config.dst_word_stride)) &
                           4095U;
      for (uint32_t matrix = 0; matrix < 4; ++matrix) {
        for (uint32_t column = 0; column < 8; ++column) {
          const uint32_t input_row =
              config.zip4 ? (column / 2) + (4 * (column % 2)) : column;
          const uint32_t src = (config.src + (q * config.src_tile_stride) +
                                (input_row * config.src_word_stride)) &
                               4095U;
          expected.at(dst).at((matrix * 8) + column) =
              memory.at(src).at((matrix * 8) + row);
        }
      }
    }
  }
  return expected;
}

void ReturnResponse(Vvec_transpose& model, std::deque<Response>& responses,
                    uint32_t cycle) {
  if (!responses.empty() && responses.front().cycle == cycle) {
    const Word data = responses.front().data;
    responses.pop_front();
    for (uint32_t i = 0; i < 32; ++i) {
      model.rd_data_i[i] = data.at(i);
    }
  }
}

void Execute(const Transpose& config, std::vector<Word>& memory,
             uint32_t seed) {
  std::mt19937 generator(seed);
  Vvec_transpose model;
  model.rst_ni = 0;
  model.clk_i = 0;
  model.start_i = 0;
  model.rd_grant_i = 0;
  model.wr_grant_i = 0;
  model.src_i = config.src;
  model.dst_i = config.dst;
  model.tiles_i = config.tiles;
  model.src_tile_stride_i = config.src_tile_stride;
  model.src_word_stride_i = config.src_word_stride;
  model.dst_tile_stride_i = config.dst_tile_stride;
  model.dst_word_stride_i = config.dst_word_stride;
  model.pack_zip4_i = config.zip4 ? 1 : 0;
  std::deque<Response> responses;
  uint32_t writes = 0;
  for (uint32_t cycle = 0; cycle < 100000; ++cycle) {
    model.clk_i = 0;
    model.rst_ni = cycle >= 3 ? 1 : 0;
    model.start_i = cycle == 4 ? 1 : 0;
    model.rd_grant_i = generator() % 4 == 0 ? 1 : 0;
    model.wr_grant_i = generator() % 8 == 0 ? 1 : 0;
    ReturnResponse(model, responses, cycle);
    model.eval();
    if (model.rd_req_o != 0 && model.rd_grant_i != 0) {
      responses.push_back(
          {.cycle = cycle + 3, .data = memory.at(model.rd_addr_o)});
    }
    if (model.wr_req_o != 0 && model.wr_grant_i != 0) {
      Word data{};
      for (uint32_t i = 0; i < 32; ++i) {
        data.at(i) = model.wr_data_o[i];
      }
      memory.at(model.wr_addr_o) = data;
      ++writes;
    }
    model.clk_i = 1;
    model.eval();
    if (cycle > 4 && model.busy_o == 0) {
      EXPECT_EQ(writes, 8 * config.tiles);
      model.final();
      return;
    }
  }
  model.final();
  FAIL() << "TRANSPOSE8 did not finish";
}

void CheckTranspose(const Transpose& config, uint32_t seed) {
  auto memory = RandomMemory(seed);
  const auto expected = Transposed(memory, config);
  Execute(config, memory, seed);
  EXPECT_EQ(memory, expected);
}

TEST(VecTransposeTest, PreservesBitsAndStridesUnderIndependentBackpressure) {
  for (const bool zip4 : {false, true}) {
    for (const uint32_t tiles : {0U, 1U, 3U, 32U, 64U}) {
      CheckTranspose({.tiles = tiles,
                      .dst_tile_stride = 1,
                      .dst_word_stride = tiles,
                      .zip4 = zip4},
                     29 + tiles);
      CheckTranspose({.tiles = tiles,
                      .src_tile_stride = 24,
                      .src_word_stride = 2,
                      .dst_tile_stride = 24,
                      .dst_word_stride = 3,
                      .zip4 = zip4},
                     31 + tiles);
      CheckTranspose({.tiles = tiles,
                      .src_tile_stride = 1,
                      .src_word_stride = 0,
                      .zip4 = zip4},
                     37 + tiles);
    }
    CheckTranspose({.src = 4090,
                    .dst = 1024,
                    .tiles = 3,
                    .src_tile_stride = 16,
                    .dst_tile_stride = 24,
                    .dst_word_stride = 3,
                    .zip4 = zip4},
                   41);
    CheckTranspose(
        {.src = 0, .dst = 8, .tiles = 511, .src_tile_stride = 0, .zip4 = zip4},
        53);
  }
}

TEST(VecTransposeTest, TransposingTwiceRestoresFourIndependentMatrices) {
  auto memory = RandomMemory(43);
  const auto original = memory;
  Execute({.dst = 1024, .tiles = 3}, memory, 43);
  Execute({.src = 1024, .tiles = 3}, memory, 47);
  for (uint32_t word = 0; word < 24; ++word) {
    EXPECT_EQ(memory.at(2048 + word), original.at(128 + word));
  }
}

}  // namespace
}  // namespace hotarubi::tb
