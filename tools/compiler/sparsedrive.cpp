#include "sparsedrive.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "checkpoint.hpp"
#include "ddr_image.hpp"
#include "isa.hpp"
#include "output.hpp"
#include "program.hpp"
#include "safetensors.hpp"
#include "sparsedrive_head.hpp"
#include "sparsedrive_kernels.hpp"
#include "sparsedrive_layers.hpp"
#include "sparsedrive_motion.hpp"
#include "sparsedrive_parts.hpp"
#include "vprog.hpp"

namespace hotarubi::compiler {
namespace {

namespace sd = sparsedrive;

constexpr uint64_t kSharedBytes = uint64_t{256} << 20;
constexpr uint64_t kStemGroupBytes = uint64_t{32} * 24 * 4 * 4;
constexpr uint32_t kFsScratch = 512;

struct FrameStartRegions {
  uint64_t meta = 0;
  uint64_t state = 0;
  uint64_t flag = 0;
  uint64_t tinv = 0;
  uint64_t col8 = 0;
  uint64_t sc = 0;
  uint64_t tpl_det = 0;
  uint64_t tpl_map = 0;
  uint64_t prog = 0;
  sd::CbufImage cbuf;
};

uint64_t SoftmaxWeightBytes(const sd::HeadParams& params) {
  return uint64_t{8} * ((params.config.na + 3) / 4) *
         (uint64_t{sd::kCameras} * params.config.scam / 8) * sd::kWordF;
}

FrameStartRegions PlaceFrameStart(DdrImage& image, sd::SparseHead& det,
                                  sd::SparseHead& map) {
  FrameStartRegions fs;
  fs.meta = image.Alloc("fs.meta", 2 * sd::kWordF);
  fs.state = image.Alloc("fs.state", 2 * sd::kWordF);
  fs.flag = image.Alloc("fs.flag", sd::kWordF);
  std::vector<sd::Word> tinv(5, sd::Word{});
  for (uint32_t lane = 0; lane < 8; ++lane) {
    tinv[sd::kFsNtwoPbuf][lane] = sd::FloatBits(-2);
  }
  fs.tinv = image.Alloc("fs.tinv", 5 * sd::kWordF);
  sd::PutWords(image, fs.tinv, tinv);
  fs.col8 = image.Alloc("fs.col8", sd::kWordF);
  sd::PutWords(image, fs.col8, {sd::FsCol8Word()});
  std::vector<sd::Word> scales(std::max<size_t>(1, det.scale_patch.size()),
                               sd::Word{});
  for (size_t i = 0; i < det.scale_patch.size(); ++i) {
    std::ranges::copy(det.scale_patch[i].second, scales[i].begin());
  }
  fs.sc = image.Alloc("fs.sc", scales.size() * sd::kWordF);
  sd::PutWords(image, fs.sc, scales);
  fs.tpl_det = image.Alloc("fs.tpl_det", 18 * sd::kWordF);
  sd::PutWords(image, fs.tpl_det, sd::FsDetTemplate());
  fs.tpl_map = image.Alloc("fs.tpl_map", 9 * sd::kWordF);
  fs.cbuf.Add("start", sd::FrameStartKernel(
                           det.ncam + map.ncam,
                           static_cast<uint32_t>(det.scale_patch.size())));
  fs.cbuf.Add("fill", sd::MaskKernel(true));
  const std::vector<std::byte> bytes = Serialize(fs.cbuf.Words());
  fs.prog = image.Alloc("fs.prog", bytes.size());
  image.Put(fs.prog, bytes);
  det.fs_flag = fs.flag;
  map.fs_flag = fs.flag;
  return fs;
}

void EmitFrameStart(Program& program, const FrameStartRegions& fs,
                    const sd::SparseHead& det, const sd::SparseHead& map,
                    const sd::MotionHead& motion, bool temp) {
  const auto ld = [&](uint64_t ddr, Mem mem, uint32_t at, uint32_t words,
                      const std::string& note) {
    program.Load(sd::Words(ddr, mem, at, words), note);
  };
  const auto st = [&](uint64_t ddr, uint32_t at, uint32_t words,
                      const std::string& note) {
    program.Store(sd::Words(ddr, Mem::kFbuf, at, words), note);
  };
  const auto ns = static_cast<uint32_t>(det.scale_patch.size());
  ld(fs.prog, Mem::kCbuf, vprog::kRomWords,
     static_cast<uint32_t>(fs.cbuf.Words().size()), "fs ld VEC programs");
  ld(fs.meta, Mem::kFbuf, sd::kFsTs, 2, "fs ld TS TG");
  ld(fs.state, Mem::kFbuf, sd::kFsPts, 2, "fs ld previous TS TG");
  ld(fs.col8, Mem::kFbuf, sd::kFsCol8, 1, "fs ld COL8");
  ld(det.kp_const, Mem::kFbuf, sd::kFsKp, det.ncam, "fs ld det kp_const");
  ld(map.kp_const, Mem::kFbuf, sd::kFsKp + det.ncam, map.ncam,
     "fs ld map kp_const");
  if (ns != 0) {
    ld(fs.sc, Mem::kFbuf, sd::kFsSc, ns, "fs ld scale");
  }
  ld(fs.tpl_det, Mem::kFbuf, sd::kFsDi, 18, "fs ld det template");
  ld(fs.tpl_map, Mem::kFbuf, sd::kFsMi, 9, "fs ld map template");
  ld(fs.tinv, Mem::kPbuf, 0, 5, "fs ld Tinv");
  program.Run(
      {.pc = fs.cbuf.Pc("start"), .src = 0, .p_base = 0, .imm = temp ? 1U : 0U},
      "fs run");
  st(det.proj_pb, sd::kFsDi, 18, "fs st det proj");
  st(map.proj_pb, sd::kFsMi, 9, "fs st map proj");
  for (const sd::SparseHead* h : {&det, &map}) {
    st(h->proj_dt, sd::kFsDt, 1, std::format("fs st {} dt", h->tag));
  }
  for (uint32_t i = 0; i < ns; ++i) {
    st(det.scale_patch[i].first, sd::kFsSc + i, 1,
       std::format("fs st scale {}", i));
  }
  st(det.kp_const, sd::kFsKp, det.ncam, "fs st det kp_const");
  st(map.kp_const, sd::kFsKp + det.ncam, map.ncam, "fs st map kp_const");
  st(fs.flag, sd::kFsFlag, 1, "fs st flag");
  st(fs.state, sd::kFsPts, 2, "fs st state");
  struct Fill {
    std::string name;
    uint64_t offset;
    uint32_t words;
    uint32_t value;
  };
  const std::array<Fill, 3> fills = {{{.name = "det prevconf",
                                       .offset = det.prevconf,
                                       .words = det.nw,
                                       .value = 0xFF800000U},
                                      {.name = "map prevconf",
                                       .offset = map.prevconf,
                                       .words = map.nw,
                                       .value = 0xFF800000U},
                                      {.name = "motion stlist",
                                       .offset = motion.stlist,
                                       .words = sd::kMotionQueries / 32,
                                       .value = sd::kNull}}};
  for (const Fill& fill : fills) {
    ld(fill.offset, Mem::kFbuf, kFsScratch, fill.words,
       std::format("fs ld {}", fill.name));
    program.Run({.pc = fs.cbuf.Pc("fill"),
                 .src = kFsScratch,
                 .src2 = sd::kFsFlag,
                 .dst = kFsScratch,
                 .rows_q = fill.words,
                 .imm = fill.value},
                std::format("fs {} if !mask", fill.name));
    st(fill.offset, kFsScratch, fill.words, std::format("fs st {}", fill.name));
  }
  program.Load(sd::Tokens(motion.pst.Addr(0), Mem::kFbuf, kFsScratch, 4,
                          motion.pst.Tok()),
               "fs ld plan_status");
  program.Run({.pc = fs.cbuf.Pc("fill"),
               .src = kFsScratch,
               .src2 = sd::kFsFlag,
               .dst = kFsScratch,
               .rows_q = 4,
               .imm = 0},
              "fs plan_status if !mask");
  program.Store(sd::Tokens(motion.pst.Addr(0), Mem::kFbuf, kFsScratch, 4,
                           motion.pst.Tok()),
                "fs st plan_status");
}

using Gathers = std::array<std::vector<size_t>, sd::kCameras>;

std::vector<std::vector<sd::FMap>> EmitEncoders(Program& program,
                                                DdrImage& image, uint64_t base,
                                                const sd::Backbone& backbone,
                                                uint64_t stem_in, bool again,
                                                Gathers& gathers) {
  std::vector<std::vector<sd::FMap>> features;
  for (uint32_t c = 0; c < sd::kCameras; ++c) {
    image.SetReuse(again || c > 0);
    features.push_back(sd::EmitEncoder(
        program, image, base, backbone,
        {.image = stem_in, .lists = "stem.lists", .gathers = &gathers[c]}, c));
  }
  image.SetReuse(false);
  return features;
}

}  // namespace

Compiled CompileSparseDrive(const SparseDriveOptions& options,
                            const SafeTensors& quant,
                            const Checkpoint& checkpoint) {
  const uint64_t base = options.base;
  DdrImage image(kSharedBytes);
  const sd::Sources sources{.quant = quant, .checkpoint = checkpoint};
  const sd::HeadWeights weights(quant, checkpoint);
  const sd::Backbone backbone = sd::ReadBackbone(sources);
  const uint64_t stem_in = image.Alloc("stem.in", sd::kStemImageBytes);
  std::vector<Stream> streams;
  streams.push_back({.name = "start", .program = Program(base), .offset = 0});
  streams.push_back(
      {.name = "continue", .program = Program(base), .offset = 0});
  std::array<Gathers, 2> gathers;
  const std::vector<std::vector<sd::FMap>> features = EmitEncoders(
      streams[0].program, image, base, backbone, stem_in, false, gathers[0]);
  sd::HeadParams det_params = sd::ReadHeadParams(weights, true);
  sd::HeadParams map_params = sd::ReadHeadParams(weights, false);
  const uint64_t heads_w = image.Alloc(
      "heads.w",
      std::max(SoftmaxWeightBytes(det_params), SoftmaxWeightBytes(map_params)));
  std::vector<std::vector<sd::Level>> levels;
  levels.reserve(features.size());
  for (const auto& camera : features) {
    std::vector<sd::Level> level;
    level.reserve(camera.size());
    for (const sd::FMap& map : camera) {
      level.push_back(
          {.phys = base + map.off[0], .h = map.h, .w = map.w, .tb = map.tb[0]});
    }
    levels.push_back(std::move(level));
  }
  sd::SparseHead det(image, base, std::move(det_params), levels, heads_w);
  sd::SparseHead map(image, base, std::move(map_params), levels, heads_w);
  sd::MotionHead motion(image, base, sd::ReadMotionParams(weights, sources),
                        det, map, features[0][3]);
  const FrameStartRegions fs = PlaceFrameStart(image, det, map);
  for (size_t s = 0; s < streams.size(); ++s) {
    Program& program = streams[s].program;
    const bool temp = s == 1;
    if (temp) {
      EmitEncoders(program, image, base, backbone, stem_in, true, gathers[s]);
    }
    EmitFrameStart(program, fs, det, map, motion, temp);
    det.EmitFrame(program, temp);
    map.EmitFrame(program, temp);
    motion.EmitFrame(program);
    program.Halt();
  }
  const uint64_t heads_w_bytes = image.Region("heads.w").size;
  Require((sd::kCameras - 1) * sd::kStemImageBytes <= heads_w_bytes,
          "camera inputs do not fit in the softmax weight area");
  for (uint32_t c = 1; c < sd::kCameras; ++c) {
    const uint64_t input = heads_w + ((c - 1) * sd::kStemImageBytes);
    image.Define(std::format("stem.in.cam{}", c), input, sd::kStemImageBytes);
    const std::vector<std::byte> lists = sd::StemListBytes(base, input);
    const uint64_t offset =
        image.Alloc(std::format("stem.lists.cam{}", c), lists.size());
    image.Put(offset, lists);
    for (size_t s = 0; s < streams.size(); ++s) {
      for (size_t k = 0; k < gathers[s][c].size(); ++k) {
        streams[s].program.SetDdrOffset(gathers[s][c][k],
                                        offset + (k * kStemGroupBytes));
      }
    }
  }
  for (Stream& stream : streams) {
    const std::vector<std::byte> bytes = stream.program.Bytes();
    stream.offset =
        image.Alloc(std::format("instr.{}", stream.name), bytes.size());
    image.Put(stream.offset, bytes);
  }
  return {.streams = std::move(streams),
          .image = std::move(image),
          .parameters = {{"cameras", sd::kCameras}}};
}

}  // namespace hotarubi::compiler
