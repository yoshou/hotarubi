#include "smolvla.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/safetensors.hpp"
#include "isa.hpp"
#include "output.hpp"
#include "program.hpp"
#include "smolvla_parts.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint64_t kSharedBytes = uint64_t{256} << 20;
constexpr uint64_t kInstructionBytes = uint64_t{4} << 20;

}  // namespace

Compiled CompileSmolVla(const SmolVlaOptions& options, const SafeTensors& quant,
                        const SafeTensors& model) {
  Require(options.cameras >= 1 && options.steps >= 1,
          "SmolVLA needs at least one camera and one step");
  const smolvla::Dims dims{.cameras = options.cameras,
                           .lang = options.lang_tokens,
                           .steps = options.steps};
  DdrImage image(kSharedBytes);
  Program program(options.base);
  const uint64_t instructions = image.Alloc("instr", kInstructionBytes);
  const smolvla::Vision vision =
      smolvla::PlaceVision(image, quant, model, dims);
  const smolvla::Lm lm = smolvla::PlaceLm(image, quant, model, dims);
  const smolvla::Expert expert =
      smolvla::PlaceExpert(image, quant, model, dims);
  smolvla::EmitVision(program, vision, dims, lm.x[0]);
  smolvla::EmitLm(program, lm, dims);
  smolvla::EmitExpert(program, expert, lm, dims);
  program.Halt();
  const std::vector<std::byte> bytes = program.Bytes();
  Require(bytes.size() <= kInstructionBytes,
          std::format("{} instructions exceed the instruction region",
                      program.Size()));
  image.Put(instructions, bytes);
  std::vector<Stream> streams;
  streams.push_back({.name = "inference",
                     .program = std::move(program),
                     .offset = instructions});
  return {.streams = std::move(streams),
          .image = std::move(image),
          .parameters = {{"cameras", options.cameras},
                         {"n_lang", options.lang_tokens},
                         {"steps", options.steps}}};
}

}  // namespace hotarubi::compiler
