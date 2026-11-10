#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "isa.hpp"
#include "program.hpp"
#include "sparsedrive_parts.hpp"
#include "tensors.hpp"
#include "vprog.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

using ::hotarubi::sparsedrive::kImageRows;
using ::hotarubi::sparsedrive::kStemPadding;
using ::hotarubi::sparsedrive::kStemRowFloats;
constexpr uint32_t kStemBlocks = 6;
constexpr uint32_t kStemWords = 24;
constexpr uint32_t kStemTaps = 21;
constexpr uint32_t kStemOut = 64;
constexpr uint32_t kStemH = 128;
constexpr uint32_t kStemW = 352;
constexpr uint32_t kMaxpoolTile = 96;
constexpr uint32_t kTileLimit = 960;
constexpr uint64_t kArenaBytes = uint64_t{32} << 20;
constexpr uint32_t kTopDownTile = 128;
constexpr uint32_t kFpnChannels = 256;

struct Stage {
  uint32_t blocks;
  uint32_t mid;
  uint32_t stride;
};

constexpr std::array<Stage, 4> kStages = {
    {{.blocks = 3, .mid = 64, .stride = 1},
     {.blocks = 4, .mid = 128, .stride = 2},
     {.blocks = 6, .mid = 256, .stride = 2},
     {.blocks = 3, .mid = 512, .stride = 2}}};

struct Face {
  int32_t dy = 0;
  std::optional<int32_t> dx;
};

struct Tap {
  uint32_t ky = 0;
  uint32_t kx = 0;
  uint32_t qoff = 0;
};

struct Src {
  const FMap* x = nullptr;
  const Conv* conv = nullptr;
  uint32_t stride = 1;
  std::vector<Face> faces;
  std::vector<std::vector<Tap>> taps;
  bool direct = false;

  [[nodiscard]] uint32_t FaceTokens(uint32_t mt) const {
    return conv->kh == 3 && stride == 1 ? mt + (2 * x->h) : mt;
  }
  [[nodiscard]] uint32_t FaceWords(uint32_t mt, size_t p) const {
    return FaceTokens(mt) / 4 * x->tok[p];
  }
};

Src MakeSrc(const FMap& x, const Conv& conv, uint32_t stride) {
  Require(conv.cin == x.c && (conv.kh == 1 || conv.kh == 3) &&
              (stride == 1 || stride == 2),
          std::format("conv on {} is not supported", x.name));
  Src src;
  src.x = &x;
  src.conv = &conv;
  src.stride = stride;
  if (conv.kh == 1) {
    src.faces = {{.dy = 0, .dx = 0}};
    src.taps = {{{.ky = 0, .kx = 0, .qoff = 0}}};
  } else if (stride == 1) {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      src.faces.push_back({.dy = dy, .dx = std::nullopt});
      std::vector<Tap> taps;
      for (int32_t dx = -1; dx <= 1; ++dx) {
        taps.push_back({.ky = static_cast<uint32_t>(dy + 1),
                        .kx = static_cast<uint32_t>(dx + 1),
                        .qoff = static_cast<uint32_t>(dx + 1) * x.h / 4});
      }
      src.taps.push_back(std::move(taps));
    }
  } else {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        src.faces.push_back({.dy = dy, .dx = dx});
        src.taps.push_back({{.ky = static_cast<uint32_t>(dy + 1),
                             .kx = static_cast<uint32_t>(dx + 1),
                             .qoff = 0}});
      }
    }
  }
  src.direct = conv.kh == 1 && stride == 1;
  return src;
}

std::vector<uint32_t> FaceListS1(uint64_t base, const FMap& x, size_t p,
                                 int64_t i0, uint32_t mt, int32_t dy) {
  const int64_t h = x.h;
  std::vector<uint32_t> list;
  list.reserve(mt + (2 * x.h));
  for (int64_t e = 0; e < mt + (2 * h); ++e) {
    const int64_t j = i0 - h + e;
    const bool ok = j >= 0 && std::cmp_less(j, x.Tokens()) &&
                    (j % h) + dy >= 0 && (j % h) + dy < h;
    list.push_back(ok ? Entry(base, x.Addr(p, static_cast<uint64_t>(j + dy)))
                      : kNull);
  }
  return list;
}

struct Strided {
  int32_t dy = 0;
  int32_t dx = 0;
  uint32_t stride = 1;
  uint32_t ho = 0;
  uint32_t wo = 0;
};

std::vector<uint32_t> FaceListStrided(uint64_t base, const FMap& x, size_t p,
                                      int64_t i0, uint32_t mt,
                                      const Strided& face) {
  std::vector<uint32_t> list;
  list.reserve(mt);
  for (int64_t o = i0; o < i0 + mt; ++o) {
    const int64_t xo = o / face.ho;
    const int64_t yo = o % face.ho;
    const int64_t yi = (int64_t{face.stride} * yo) + face.dy;
    const int64_t xi = (int64_t{face.stride} * xo) + face.dx;
    const bool ok = o < int64_t{face.ho} * face.wo && yi >= 0 &&
                    std::cmp_less(yi, x.h) && xi >= 0 && std::cmp_less(xi, x.w);
    list.push_back(
        ok ? Entry(base, x.Addr(p, static_cast<uint64_t>((xi * x.h) + yi)))
           : kNull);
  }
  return list;
}

std::vector<uint32_t> FaceList(uint64_t base, const Src& src, size_t p,
                               int64_t i0, uint32_t mt, size_t f, uint32_t ho,
                               uint32_t wo) {
  const Face& face = src.faces[f];
  if (!face.dx.has_value()) {
    return FaceListS1(base, *src.x, p, i0, mt, face.dy);
  }
  return FaceListStrided(base, *src.x, p, i0, mt,
                         {.dy = face.dy,
                          .dx = *face.dx,
                          .stride = src.stride,
                          .ho = ho,
                          .wo = wo});
}

struct ConvSpec {
  std::string name;
  std::vector<Src> srcs;
  std::vector<uint32_t> bias;
  const FMap* out_f32 = nullptr;
  const FMap* out_i8 = nullptr;
  const FMap* residual = nullptr;
  bool relu = true;
};

struct Job {
  size_t si = 0;
  size_t p = 0;
  size_t f = 0;
  size_t ti = 0;
};

class ConvEmitter {
 public:
  ConvEmitter(Program& program, DdrImage& image, uint64_t base,
              const ConvSpec& spec)
      : program_(program), image_(image), base_(base), spec_(spec) {}

  void Emit() {
    Plan();
    PlaceParams();
    PlaceLists();
    program_.Load(Words(bias_, Mem::kPbuf, 0, (n_ + kBlock - 1) / kBlock),
                  std::format("{} ld bias", spec_.name));
    LoadFaces(0);
    if (resident_) {
      for (size_t si = 0; si < spec_.srcs.size(); ++si) {
        program_.Load(
            Words(weights_[si], Mem::kWbuf, src_base_[si], src_words_[si]),
            std::format("{} ld W (resident) s{}", spec_.name, si));
      }
    } else {
      const uint64_t total = uint64_t{groups_} * jobs_.size();
      for (uint64_t q = 0; q < std::min<uint64_t>(slots_ - 1, total); ++q) {
        LoadWeight(q);
      }
    }
    if (spec_.residual != nullptr) {
      LoadResidual(0);
    }
    for (uint32_t g = 0; g < groups_; ++g) {
      Group(g);
    }
    PostVec(groups_ - 1);
    PostStore(groups_ - 1);
  }

 private:
  void Plan() {
    CheckShapes();
    ChooseTile();
    tiles_ = (n_out_ + mt_ - 1) / mt_;
    groups_ = tiles_ * tiles_n_;
    PlaceFaceBuffers();
    PlanJobs();
    PlanWeights();
  }

  void CheckShapes() {
    const Src& main = spec_.srcs[0];
    n_ = static_cast<uint32_t>(spec_.bias.size());
    ho_ = main.x->h / main.stride;
    wo_ = main.x->w / main.stride;
    n_out_ = ho_ * wo_;
    for (const Src& src : spec_.srcs) {
      Require(src.x->h / src.stride == ho_ && src.x->w / src.stride == wo_ &&
                  src.conv->w.rows == n_,
              std::format("{}: sources disagree", spec_.name));
    }
    for (const FMap* map : {spec_.out_f32, spec_.out_i8, spec_.residual}) {
      Require(
          map == nullptr || (map->h == ho_ && map->w == wo_ && map->c == n_),
          std::format("{}: output shape mismatch", spec_.name));
    }
    nt_ = std::min(n_, kPlane);
    tiles_n_ = n_ / nt_;
    if (spec_.residual != nullptr) {
      Require(spec_.residual->planes[0] == nt_, "residual plane mismatch");
    }
  }

  void PlaceFaceBuffers() {
    uint32_t a = 0;
    for (uint32_t parity = 0; parity < 2; ++parity) {
      for (size_t si = 0; si < spec_.srcs.size(); ++si) {
        const Src& src = spec_.srcs[si];
        for (size_t p = 0; p < src.x->planes.size(); ++p) {
          for (size_t f = 0; f < src.faces.size(); ++f) {
            face_base_[{parity, si, p, f}] = a;
            a += src.FaceWords(mt_, p);
          }
        }
      }
    }
    const uint32_t q_words = (mt_ / 4) * (nt_ / kBlock);
    qb_ = {a, a + q_words};
    Require(qb_[1] + q_words <= kAbufWords, "conv tile exceeds ABUF");
  }

  void PlanJobs() {
    for (size_t si = 0; si < spec_.srcs.size(); ++si) {
      const Src& src = spec_.srcs[si];
      first_job_.push_back(jobs_.size());
      for (size_t p = 0; p < src.x->planes.size(); ++p) {
        for (size_t f = 0; f < src.faces.size(); ++f) {
          for (size_t ti = 0; ti < src.taps[f].size(); ++ti) {
            jobs_.push_back({.si = si, .p = p, .f = f, .ti = ti});
          }
        }
      }
      jobs_per_src_.push_back(jobs_.size() - first_job_.back());
    }
    for (const Job& job : jobs_) {
      job_words_.push_back(nt_ / 8 * spec_.srcs[job.si].x->tok[job.p]);
    }
  }

  void PlanWeights() {
    uint32_t total = 0;
    for (size_t si = 0; si < spec_.srcs.size(); ++si) {
      src_base_.push_back(total);
      src_words_.push_back(tiles_n_ * static_cast<uint32_t>(jobs_per_src_[si]) *
                           (nt_ / 8 * spec_.srcs[si].x->tok[0]));
      total += src_words_.back();
    }
    resident_ = total <= kWbufWords;
    slot_words_ = *std::ranges::max_element(job_words_);
    slots_ = resident_ ? 0 : kWbufWords / slot_words_;
  }

  void ChooseTile() {
    for (const uint32_t ns : {3U, 2U}) {
      for (uint32_t mt = kTileLimit; mt >= kBlock; mt -= kBlock) {
        uint32_t ab = 0;
        for (const Src& src : spec_.srcs) {
          for (size_t p = 0; p < src.x->planes.size(); ++p) {
            ab +=
                static_cast<uint32_t>(src.faces.size()) * src.FaceWords(mt, p);
          }
        }
        ab *= 2;
        if (spec_.out_i8 != nullptr) {
          ab += 2 * (mt / 4) * (nt_ / kBlock);
        }
        const uint32_t fb = ns * (mt / 4) * (nt_ / 8);
        if (ab <= kAbufWords && fb <= kFbufWords && (ns == 2 || mt >= 128)) {
          mt_ = mt;
          ns_ = ns;
          return;
        }
      }
    }
    Fail(std::format("{}: no tile fits", spec_.name));
  }

  void PlaceParams() {
    for (size_t si = 0; si < spec_.srcs.size(); ++si) {
      const Src& src = spec_.srcs[si];
      std::vector<std::byte> data;
      for (uint32_t nt = 0; nt < tiles_n_; ++nt) {
        uint32_t c0 = 0;
        for (const uint32_t channels : src.x->planes) {
          for (size_t f = 0; f < src.faces.size(); ++f) {
            for (const Tap& tap : src.taps[f]) {
              const std::vector<std::byte> tile = PackWbuf(
                  ConvTile(*src.conv, nt * nt_, nt_,
                           (tap.ky * src.conv->kh) + tap.kx, c0, channels));
              data.insert(data.end(), tile.begin(), tile.end());
            }
          }
          c0 += channels;
        }
      }
      const uint64_t offset =
          image_.Alloc(std::format("{}.w{}", spec_.name, si), data.size());
      image_.Put(offset, data);
      weights_.push_back(offset);
    }
    const std::vector<std::byte> bias = PackPbuf(PaddedBias(spec_.bias));
    bias_ = image_.Alloc(std::format("{}.b", spec_.name), bias.size());
    image_.Put(bias_, bias);
  }

  void PlaceLists() {
    for (uint32_t t = 0; t < tiles_; ++t) {
      for (size_t si = 0; si < spec_.srcs.size(); ++si) {
        const Src& src = spec_.srcs[si];
        if (src.direct) {
          continue;
        }
        for (size_t p = 0; p < src.x->planes.size(); ++p) {
          for (size_t f = 0; f < src.faces.size(); ++f) {
            list_ids_[{t, si, p, f}] = lists_.Add(
                FaceList(base_, src, p, int64_t{t} * mt_, mt_, f, ho_, wo_));
          }
        }
      }
    }
    lists_.Place(image_, std::format("{}.idx", spec_.name));
  }

  [[nodiscard]] uint32_t FaceBase(uint32_t t, size_t si, size_t p,
                                  size_t f) const {
    return face_base_.at({t % 2, si, p, f});
  }

  void LoadFaces(uint32_t t) {
    for (size_t si = 0; si < spec_.srcs.size(); ++si) {
      const Src& src = spec_.srcs[si];
      for (size_t p = 0; p < src.x->planes.size(); ++p) {
        for (size_t f = 0; f < src.faces.size(); ++f) {
          const uint32_t dst = FaceBase(t, si, p, f);
          if (src.direct) {
            program_.Load(
                Tokens(src.x->Addr(p, uint64_t{t} * mt_), Mem::kAbuf, dst, mt_,
                       src.x->tok[p]),
                std::format("{} ld A t{} s{} p{}", spec_.name, t, si, p));
          } else {
            program_.Load(
                Tokens(lists_.Offset(list_ids_.at({t, si, p, f})), Mem::kAbuf,
                       dst, src.FaceTokens(mt_), src.x->tok[p], true),
                std::format("{} ld face t{} s{} p{} f{}", spec_.name, t, si, p,
                            f));
          }
        }
      }
    }
  }

  [[nodiscard]] uint64_t TileIndex(uint32_t nt, size_t k) const {
    const size_t si = jobs_[k].si;
    return (uint64_t{nt} * jobs_per_src_[si]) + (k - first_job_[si]);
  }

  void LoadWeight(uint64_t q) {
    const auto g = static_cast<uint32_t>(q / jobs_.size());
    const size_t k = q % jobs_.size();
    const uint32_t nt = g % tiles_n_;
    program_.Load(
        Words(
            weights_[jobs_[k].si] + (TileIndex(nt, k) * job_words_[k] * kWordA),
            Mem::kWbuf, static_cast<uint32_t>(q % slots_) * slot_words_,
            job_words_[k]),
        std::format("{} ld W g{} k{}", spec_.name, g, k));
  }

  void LoadResidual(uint32_t g) {
    const uint32_t t = g / tiles_n_;
    const uint32_t nt = g % tiles_n_;
    program_.Load(Tokens(spec_.residual->Addr(nt, uint64_t{t} * mt_),
                         Mem::kFbuf, C(g), mt_, nt_ / kBlock),
                  std::format("{} ld residual t{} nt{}", spec_.name, t, nt));
  }

  [[nodiscard]] uint32_t C(uint32_t g) const {
    return (g % ns_) * (mt_ / 4) * (nt_ / 8);
  }

  void PostVec(uint32_t g) {
    const uint32_t t = g / tiles_n_;
    const uint32_t nt = g % tiles_n_;
    const uint32_t c = C(g);
    const std::string where = std::format("t{} nt{}", t, nt);
    program_.Vec({.op = VecOp::kAddP,
                  .src = c,
                  .dst = c,
                  .rows_q = mt_ / 4,
                  .n8 = nt_ / 8,
                  .p_base = nt * (nt_ / kBlock)},
                 std::format("{} bias {}", spec_.name, where));
    if (spec_.relu && spec_.out_f32 != nullptr) {
      program_.Vec({.op = VecOp::kRelu,
                    .src = c,
                    .dst = c,
                    .rows_q = mt_ / 4,
                    .n8 = nt_ / 8},
                   std::format("{} relu {}", spec_.name, where));
    }
    if (spec_.out_i8 != nullptr) {
      const bool relu = spec_.relu && spec_.out_f32 == nullptr;
      program_.Vec({.op = relu ? VecOp::kRelu : VecOp::kQuant,
                    .src = c,
                    .dst = qb_[g % 2],
                    .rows_q = mt_ / 4,
                    .n8 = nt_ / 8,
                    .q = true},
                   std::format("{} {}+Q {}", spec_.name,
                               relu ? "RELU" : "QUANT", where));
    }
  }

  void PostStore(uint32_t g) {
    const uint32_t t = g / tiles_n_;
    const uint32_t nt = g % tiles_n_;
    const uint32_t m = std::min(mt_, n_out_ - (t * mt_));
    if (spec_.out_f32 != nullptr) {
      program_.Store(Tokens(spec_.out_f32->Addr(nt, uint64_t{t} * mt_),
                            Mem::kFbuf, C(g), m, nt_ / kBlock),
                     std::format("{} st f32 t{} nt{}", spec_.name, t, nt));
    }
    if (spec_.out_i8 != nullptr) {
      program_.Store(Tokens(spec_.out_i8->Addr(nt, uint64_t{t} * mt_),
                            Mem::kAbuf, qb_[g % 2], m, nt_ / kBlock),
                     std::format("{} st i8 t{} nt{}", spec_.name, t, nt));
    }
  }

  void Group(uint32_t g) {
    const uint32_t t = g / tiles_n_;
    const uint32_t nt = g % tiles_n_;
    if (g > 0) {
      PostVec(g - 1);
    }
    for (size_t k = 0; k < jobs_.size(); ++k) {
      const Job& job = jobs_[k];
      const Src& src = spec_.srcs[job.si];
      const Tap& tap = src.taps[job.f][job.ti];
      const uint32_t kb = src.x->tok[job.p];
      const uint64_t q = (uint64_t{g} * jobs_.size()) + k;
      const uint32_t w_base =
          resident_
              ? src_base_[job.si] +
                    static_cast<uint32_t>(TileIndex(nt, k) * job_words_[k])
              : static_cast<uint32_t>(q % slots_) * slot_words_;
      program_.Gemm(
          {.a_base = FaceBase(t, job.si, job.p, job.f) + (tap.qoff * kb),
           .a_stride = kb,
           .kb = kb,
           .m = mt_,
           .w_base = w_base,
           .w_stride = kb,
           .n8 = nt_ / 8,
           .c_base = C(g),
           .acc = k > 0 || spec_.residual != nullptr},
          std::format("{} gemm t{} nt{} s{} p{} f{} tap{}{}", spec_.name, t, nt,
                      job.si, job.p, job.f, tap.ky, tap.kx));
      if (k == 0) {
        if (nt == 0 && t + 1 < tiles_) {
          LoadFaces(t + 1);
        }
        if (spec_.residual != nullptr && g + 1 < groups_) {
          LoadResidual(g + 1);
        }
        if (g > 0) {
          PostStore(g - 1);
        }
      }
      if (!resident_) {
        const uint64_t next = q + slots_ - 1;
        if (next < uint64_t{groups_} * jobs_.size()) {
          LoadWeight(next);
        }
      }
    }
  }

  using Key = std::array<size_t, 4>;

  Program& program_;
  DdrImage& image_;
  uint64_t base_;
  const ConvSpec& spec_;
  uint32_t n_ = 0;
  uint32_t ho_ = 0;
  uint32_t wo_ = 0;
  uint32_t n_out_ = 0;
  uint32_t nt_ = 0;
  uint32_t tiles_n_ = 0;
  uint32_t mt_ = 0;
  uint32_t ns_ = 0;
  uint32_t tiles_ = 0;
  uint32_t groups_ = 0;
  std::map<Key, uint32_t> face_base_;
  std::array<uint32_t, 2> qb_{};
  std::vector<Job> jobs_;
  std::vector<size_t> first_job_;
  std::vector<size_t> jobs_per_src_;
  std::vector<uint32_t> job_words_;
  std::vector<uint32_t> src_base_;
  std::vector<uint32_t> src_words_;
  bool resident_ = false;
  uint32_t slot_words_ = 0;
  uint32_t slots_ = 0;
  std::vector<uint64_t> weights_;
  uint64_t bias_ = 0;
  Lists lists_;
  std::map<Key, size_t> list_ids_;
};

void EmitConv(Program& program, DdrImage& image, uint64_t base,
              const ConvSpec& spec) {
  ConvEmitter(program, image, base, spec).Emit();
}

uint32_t StemGroupColumn(uint32_t gi) { return (16 * (gi / 4)) + (gi % 4); }

std::vector<uint32_t> StemLists(uint64_t base, uint64_t image) {
  constexpr uint32_t kQuads = kStemH / 4;
  constexpr uint32_t kGroups = kStemW / 4;
  std::vector<uint32_t> out(
      static_cast<size_t>(kGroups) * kQuads * kStemWords * 4, kNull);
  for (uint32_t gi = 0; gi < kGroups; ++gi) {
    const uint32_t xb = StemGroupColumn(gi);
    const uint32_t phase = xb % 2;
    const uint32_t s = (2 * xb) + (2 * phase);
    Require(s % 4 == 0, "stem window is not aligned");
    for (uint32_t n = 0; n < kStemTaps; ++n) {
      const uint32_t c = n / 7;
      const uint32_t ky = n % 7;
      for (uint32_t y = 0; y < kStemH; ++y) {
        const int64_t row = int64_t{ky} + (2 * int64_t{y}) - kStemPadding;
        if (row < 0 || std::cmp_greater_equal(row, kImageRows)) {
          continue;
        }
        const size_t at =
            ((((static_cast<size_t>(gi) * kQuads) + (y / 4)) * kStemWords + n) *
             4) +
            (y % 4);
        out[at] = Entry(base, image +
                                  (((((phase * 3) + c) * uint64_t{kImageRows}) +
                                    static_cast<uint64_t>(row)) *
                                   kStemRowFloats * 4) +
                                  (uint64_t{s} * 4));
      }
    }
  }
  return out;
}

struct StemParams {
  uint64_t hi = 0;
  uint64_t lo = 0;
  uint64_t bias = 0;
  uint64_t lists = 0;
  uint64_t program = 0;
  uint32_t program_words = 0;
};

std::vector<std::byte> ListBytes(const std::vector<uint32_t>& list) {
  std::vector<std::byte> bytes;
  bytes.reserve(list.size() * 4);
  for (const uint32_t entry : list) {
    for (uint32_t i = 0; i < 4; ++i) {
      bytes.push_back(static_cast<std::byte>(entry >> (8 * i)));
    }
  }
  return bytes;
}

uint64_t PlaceBytes(DdrImage& image, std::string_view name,
                    const std::vector<std::byte>& bytes) {
  const uint64_t offset = image.Alloc(name, bytes.size());
  image.Put(offset, bytes);
  return offset;
}

StemParams PlaceStem(DdrImage& image, uint64_t base, const Backbone& backbone,
                     const CameraInput& input) {
  Require(backbone.stem.cols == kBlock * kStemBlocks,
          "conv1 does not have 192 columns");
  const W8Planes planes = SplitW8(backbone.stem);
  StemParams params;
  params.hi = PlaceBytes(image, "stem.w_hi", PackWbuf(planes.hi));
  params.lo = PlaceBytes(image, "stem.w_lo", PackWbuf(planes.lo));
  params.bias = PlaceBytes(image, "stem.b", PackPbuf(backbone.stem_bias));
  params.lists =
      PlaceBytes(image, input.lists, ListBytes(StemLists(base, input.image)));
  const std::vector<Instruction> kernel = StemQuantKernel();
  params.program = PlaceBytes(image, "stem.prog", Serialize(kernel));
  params.program_words = static_cast<uint32_t>(kernel.size());
  return params;
}

void EmitStem(Program& program, const StemParams& params, const FMap& out,
              std::vector<size_t>* gathers) {
  constexpr uint32_t kQuads = kStemH / 4;
  constexpr uint32_t kFaceWords = kQuads * kStemWords * 4;
  constexpr uint32_t kAWords = kQuads * kStemWords;
  constexpr std::array<uint32_t, 2> kA = {0, kAWords};
  constexpr uint32_t kCWords = kQuads * (kStemOut / 8);
  constexpr uint32_t kSlots = 3;
  constexpr uint32_t kWWords = (kStemOut / 8) * kStemBlocks;
  program.Load(
      Words(params.program, Mem::kCbuf, vprog::kRomWords, params.program_words),
      "stem ld VEC program");
  program.Load(Words(params.bias, Mem::kPbuf, 0, kStemOut / kBlock),
               "stem ld bias");
  program.Load(Words(params.hi, Mem::kWbuf, 0, kWWords), "stem ld W hi");
  program.Load(Words(params.lo, Mem::kWbuf, kWWords, kWWords), "stem ld W lo");
  uint32_t column = 0;
  for (uint32_t gi = 0; gi < kStemW / 4; ++gi) {
    const uint32_t xb = StemGroupColumn(gi);
    const uint32_t a = kA[gi % 2];
    const size_t gather =
        program.Load(Tokens(params.lists + (uint64_t{gi} * kFaceWords * 4),
                            Mem::kFbuf, 0, kFaceWords, 1, true),
                     std::format("stem gather g{}", gi));
    if (gathers != nullptr) {
      gathers->push_back(gather);
    }
    for (uint32_t g = 0; g < 4; ++g) {
      program.Run({.pc = vprog::kRomWords,
                   .src = g,
                   .dst = a + (kStemBlocks * g),
                   .rows_q = kQuads,
                   .n8 = 4 * kStemWords,
                   .reads =
                       std::vector<Range>{
                           {.mem = Mem::kFbuf, .lo = 0, .hi = kFaceWords}},
                   .writes =
                       std::vector<Range>{
                           {.mem = Mem::kAbuf, .lo = a, .hi = a + kAWords}}},
                  std::format("stem quant g{} c{}", gi, g));
    }
    for (uint32_t g = 0; g < 4; ++g) {
      const uint32_t x = xb + (4 * g);
      const uint32_t c = kFaceWords + ((column++ % kSlots) * kCWords);
      for (const bool lo : {false, true}) {
        program.Gemm({.a_base = a + (kStemBlocks * g),
                      .a_stride = kStemWords,
                      .kb = kStemBlocks,
                      .m = kStemH,
                      .w_base = lo ? kWWords : 0,
                      .w_stride = kStemBlocks,
                      .n8 = kStemOut / 8,
                      .c_base = c,
                      .wmode = lo ? WeightMode::kW8Lo : WeightMode::kW8Hi,
                      .acc = lo},
                     std::format("stem gemm {} x{}", lo ? "lo" : "hi", x));
      }
      program.Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = c,
                   .rows_q = kQuads,
                   .n8 = kStemOut / 8},
                  std::format("stem bias x{}", x));
      program.Vec({.op = VecOp::kRelu,
                   .src = c,
                   .dst = c,
                   .rows_q = kQuads,
                   .n8 = kStemOut / 8},
                  std::format("stem relu x{}", x));
      program.Store(Tokens(out.Addr(0, uint64_t{x} * kStemH), Mem::kFbuf, c,
                           kStemH, kStemOut / kBlock),
                    std::format("stem st x{}", x));
    }
  }
}

void EmitMaxpool(Program& program, DdrImage& image, uint64_t base,
                 const FMap& x, const FMap& out) {
  const uint32_t ho = x.h / 2;
  const uint32_t wo = x.w / 2;
  const uint32_t n_out = ho * wo;
  const uint32_t tiles = (n_out + kMaxpoolTile - 1) / kMaxpoolTile;
  const uint32_t face_words = (kMaxpoolTile / 4) * x.QuadWords(0);
  const uint32_t q_words = (kMaxpoolTile / 4) * (x.c / kBlock);
  Require(2 * 9 * face_words <= kFbufWords, "maxpool faces exceed FBUF");
  Lists lists;
  std::vector<size_t> ids;
  for (uint32_t t = 0; t < tiles; ++t) {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        ids.push_back(lists.Add(FaceListStrided(
            base, x, 0, int64_t{t} * kMaxpoolTile, kMaxpoolTile,
            {.dy = dy, .dx = dx, .stride = 2, .ho = ho, .wo = wo})));
      }
    }
  }
  lists.Place(image, "maxpool.idx");
  const auto face = [&](uint32_t t, uint32_t f) {
    return ((t % 2) * 9 * face_words) + (f * face_words);
  };
  const auto load = [&](uint32_t t) {
    for (uint32_t f = 0; f < 9; ++f) {
      program.Load(Tokens(lists.Offset(ids[(t * 9) + f]), Mem::kFbuf,
                          face(t, f), kMaxpoolTile, x.tok[0], true),
                   std::format("maxpool ld face t{} f{}", t, f));
    }
  };
  load(0);
  for (uint32_t t = 0; t < tiles; ++t) {
    for (uint32_t f = 1; f < 9; ++f) {
      program.Vec({.op = VecOp::kMax,
                   .src = face(t, 0),
                   .src2 = face(t, f),
                   .dst = face(t, 0),
                   .rows_q = kMaxpoolTile / 4,
                   .n8 = x.c / 8},
                  std::format("maxpool max t{} f{}", t, f));
    }
    program.Vec({.op = VecOp::kQuant,
                 .src = face(t, 0),
                 .dst = (t % 2) * q_words,
                 .rows_q = kMaxpoolTile / 4,
                 .n8 = x.c / 8},
                std::format("maxpool quant t{}", t));
    if (t + 1 < tiles) {
      load(t + 1);
    }
    program.Store(
        Tokens(out.Addr(0, uint64_t{t} * kMaxpoolTile), Mem::kAbuf,
               (t % 2) * q_words,
               std::min(kMaxpoolTile, n_out - (t * kMaxpoolTile)), x.tok[0]),
        std::format("maxpool st t{}", t));
  }
}

void EmitQuantMap(Program& program, const FMap& x, const FMap& out,
                  std::string_view name) {
  constexpr uint32_t kTile = kTopDownTile;
  const uint32_t tiles = (x.Tokens() + kTile - 1) / kTile;
  const uint32_t words = x.QuadWords(0);
  const std::array<uint32_t, 2> xs = {0, (kTile / 4) * words};
  const std::array<uint32_t, 2> qb = {0, (kTile / 4) * x.tok[0]};
  program.Load(Tokens(x.Addr(0, 0), Mem::kFbuf, xs[0], kTile, x.tok[0]),
               std::format("{} ld t0", name));
  for (uint32_t t = 0; t < tiles; ++t) {
    program.Vec({.op = VecOp::kQuant,
                 .src = xs[t % 2],
                 .dst = qb[t % 2],
                 .rows_q = kTile / 4,
                 .n8 = words},
                std::format("{} quant t{}", name, t));
    if (t + 1 < tiles) {
      program.Load(Tokens(x.Addr(0, uint64_t{t + 1} * kTile), Mem::kFbuf,
                          xs[(t + 1) % 2], kTile, x.tok[0]),
                   std::format("{} ld t{}", name, t + 1));
    }
    program.Store(
        Tokens(out.Addr(0, uint64_t{t} * kTile), Mem::kAbuf, qb[t % 2],
               std::min(kTile, x.Tokens() - (t * kTile)), x.tok[0]),
        std::format("{} st t{}", name, t));
  }
}

void EmitUpsampleAdd(Program& program, DdrImage& image, uint64_t base,
                     const FMap& a, const FMap& b, const FMap* out_f32,
                     const FMap& out_i8, std::string_view name) {
  constexpr uint32_t kTile = kTopDownTile;
  constexpr uint32_t kSlot = (kTile / 4) * 32;
  constexpr std::array<uint32_t, 2> kXa = {0, kSlot};
  constexpr std::array<uint32_t, 2> kXb = {2 * kSlot, 3 * kSlot};
  constexpr std::array<uint32_t, 2> kQb = {0, (kTile / 4) * 8};
  const uint32_t tiles = (a.Tokens() + kTile - 1) / kTile;
  Lists lists;
  for (uint32_t t = 0; t < tiles; ++t) {
    std::vector<uint32_t> list;
    list.reserve(kTile);
    for (uint32_t o = t * kTile; o < (t + 1) * kTile; ++o) {
      const uint32_t xo = o / a.h;
      const uint32_t yo = o % a.h;
      list.push_back(
          o < a.Tokens()
              ? Entry(base, b.Addr(0, (uint64_t{xo / 2} * b.h) + (yo / 2)))
              : kNull);
    }
    lists.Add(std::move(list));
  }
  lists.Place(image, std::format("{}.idx", name));
  const auto load = [&](uint32_t t) {
    program.Load(Tokens(a.Addr(0, uint64_t{t} * kTile), Mem::kFbuf, kXa[t % 2],
                        kTile, 8),
                 std::format("{} ld A t{}", name, t));
    program.Load(
        Tokens(lists.Offset(t), Mem::kFbuf, kXb[t % 2], kTile, 8, true),
        std::format("{} ld up(B) t{}", name, t));
  };
  load(0);
  for (uint32_t t = 0; t < tiles; ++t) {
    const uint32_t m = std::min(kTile, a.Tokens() - (t * kTile));
    program.Vec({.op = VecOp::kAdd,
                 .src = kXa[t % 2],
                 .src2 = kXb[t % 2],
                 .dst = kXa[t % 2],
                 .rows_q = kTile / 4,
                 .n8 = 32},
                std::format("{} add t{}", name, t));
    if (t + 1 < tiles) {
      load(t + 1);
    }
    program.Vec({.op = VecOp::kQuant,
                 .src = kXa[t % 2],
                 .dst = kQb[t % 2],
                 .rows_q = kTile / 4,
                 .n8 = 32},
                std::format("{} quant t{}", name, t));
    if (out_f32 != nullptr) {
      program.Store(Tokens(out_f32->Addr(0, uint64_t{t} * kTile), Mem::kFbuf,
                           kXa[t % 2], m, 8),
                    std::format("{} st f32 t{}", name, t));
    }
    program.Store(Tokens(out_i8.Addr(0, uint64_t{t} * kTile), Mem::kAbuf,
                         kQb[t % 2], m, 8),
                  std::format("{} st i8 t{}", name, t));
  }
}

class EncoderEmitter {
 public:
  EncoderEmitter(Program& program, DdrImage& image, uint64_t base,
                 const Backbone& backbone)
      : program_(program),
        image_(image),
        base_(base),
        backbone_(backbone),
        arena_(image, "enc.arena", kArenaBytes) {}

  std::vector<FMap> Emit(const CameraInput& input, uint32_t camera) {
    FMap conv1 =
        MakeMap("stem.conv1", kStemH, kStemW, kStemOut, MapKind::kFloat);
    PlaceMap(conv1, image_, std::nullopt, &arena_);
    {
      const Program::Scope scope(program_, "stem");
      EmitStem(program_, PlaceStem(image_, base_, backbone_, input), conv1,
               input.gathers);
    }
    FMap x8 =
        MakeMap("stem.out", kStemH / 2, kStemW / 2, kStemOut, MapKind::kInt8);
    PlaceMap(x8, image_);
    {
      const Program::Scope scope(program_, "maxpool");
      EmitMaxpool(program_, image_, base_, conv1, x8);
    }
    std::vector<FMap> c = Resnet(std::move(x8));
    const Program::Scope scope(program_, "fpn");
    return Fpn(c, camera);
  }

 private:
  [[nodiscard]] const Conv& Weights(const std::string& name) const {
    const auto found = backbone_.convs.find(name);
    Require(found != backbone_.convs.end(), std::format("no conv {}", name));
    return found->second;
  }

  std::vector<FMap> Resnet(FMap x8) {
    std::vector<FMap> outputs;
    FMap x32;
    uint32_t h = kStemH / 2;
    uint32_t w = kStemW / 2;
    for (size_t i = 0; i < kStages.size(); ++i) {
      const Stage& stage = kStages[i];
      const uint32_t cout = 4 * stage.mid;
      const uint32_t s0 = stage.stride;
      arena_.Reset();
      FMap t1_0 = MakeMap(std::format("layer{}.t1", i + 1), h, w, stage.mid,
                          MapKind::kInt8);
      PlaceMap(t1_0, image_, std::nullopt, &arena_);
      FMap t2_0 = MakeMap(std::format("layer{}.t2", i + 1), h / s0, w / s0,
                          stage.mid, MapKind::kInt8);
      PlaceMap(t2_0, image_, std::nullopt, &arena_);
      std::array<FMap, 2> pool32;
      std::array<FMap, 2> pool8;
      for (uint32_t k = 0; k < 2; ++k) {
        pool32[k] = MakeMap(std::format("layer{}.out{}", i + 1, k), h / s0,
                            w / s0, cout, MapKind::kFloat);
        PlaceMap(pool32[k], image_, std::nullopt, &arena_);
        pool8[k] = MakeMap(std::format("layer{}.out8{}", i + 1, k), h / s0,
                           w / s0, cout, MapKind::kInt8);
        PlaceMap(pool8[k], image_, std::nullopt,
                 k == (stage.blocks - 1) % 2 ? nullptr : &arena_);
      }
      for (uint32_t j = 0; j < stage.blocks; ++j) {
        const uint32_t s = j == 0 ? s0 : 1;
        const std::string tag = std::format("layer{}.{}", i + 1, j);
        const Program::Scope scope(program_, tag);
        FMap t1 = MakeMap(tag + ".t1", h, w, stage.mid, MapKind::kInt8);
        AliasMap(t1, t1_0);
        EmitConv(program_, image_, base_,
                 {.name = tag + ".conv1",
                  .srcs = {MakeSrc(x8, Weights(tag + ".conv1"), 1)},
                  .bias = Weights(tag + ".conv1").bias,
                  .out_i8 = &t1});
        FMap t2 = MakeMap(tag + ".t2", h / s, w / s, stage.mid, MapKind::kInt8);
        AliasMap(t2, t2_0);
        EmitConv(program_, image_, base_,
                 {.name = tag + ".conv2",
                  .srcs = {MakeSrc(t1, Weights(tag + ".conv2"), s)},
                  .bias = Weights(tag + ".conv2").bias,
                  .out_i8 = &t2});
        FMap y32 = MakeMap(tag + ".out", h / s, w / s, cout, MapKind::kFloat);
        AliasMap(y32, pool32[j % 2]);
        FMap y8 = MakeMap(tag + ".out8", h / s, w / s, cout, MapKind::kInt8);
        AliasMap(y8, pool8[j % 2]);
        const Conv& conv3 = Weights(tag + ".conv3");
        if (j == 0) {
          const Conv& down = Weights(tag + ".downsample.0");
          EmitConv(program_, image_, base_,
                   {.name = tag + ".conv3",
                    .srcs = {MakeSrc(t2, conv3, 1), MakeSrc(x8, down, s)},
                    .bias = AddBits(conv3.bias, down.bias),
                    .out_f32 = &y32,
                    .out_i8 = &y8});
        } else {
          EmitConv(program_, image_, base_,
                   {.name = tag + ".conv3",
                    .srcs = {MakeSrc(t2, conv3, 1)},
                    .bias = conv3.bias,
                    .out_f32 = &y32,
                    .out_i8 = &y8,
                    .residual = &x32});
        }
        x32 = std::move(y32);
        x8 = std::move(y8);
        h /= s;
        w /= s;
      }
      outputs.push_back(x8);
    }
    return outputs;
  }

  std::vector<FMap> Fpn(const std::vector<FMap>& c, uint32_t camera) {
    arena_.Reset();
    std::vector<FMap> lateral;
    for (size_t i = 0; i < c.size(); ++i) {
      lateral.push_back(MakeMap(std::format("lat{}", i), c[i].h, c[i].w,
                                kFpnChannels, MapKind::kFloat));
      PlaceMap(lateral.back(), image_, std::nullopt, &arena_);
    }
    for (size_t i = 0; i < c.size(); ++i) {
      const Conv& conv = Weights(std::format("lateral{}", i));
      EmitConv(program_, image_, base_,
               {.name = std::format("lateral{}", i),
                .srcs = {MakeSrc(c[i], conv, 1)},
                .bias = conv.bias,
                .out_f32 = &lateral[i],
                .relu = false});
    }
    std::vector<FMap> sum;
    for (size_t i = 0; i < c.size(); ++i) {
      sum.push_back(MakeMap(std::format("fsum{}", i), c[i].h, c[i].w,
                            kFpnChannels, MapKind::kInt8));
      PlaceMap(sum.back(), image_, std::nullopt, &arena_);
    }
    EmitQuantMap(program_, lateral[3], sum[3], "fsum3");
    for (size_t i = 3; i-- > 0;) {
      EmitUpsampleAdd(program_, image_, base_, lateral[i], lateral[i + 1],
                      i > 0 ? &lateral[i] : nullptr, sum[i],
                      std::format("topdown{}", i));
    }
    std::vector<FMap> features;
    for (size_t i = 0; i < c.size(); ++i) {
      features.push_back(MakeMap(camera == 0
                                     ? std::format("F{}", i)
                                     : std::format("F{}.cam{}", i, camera),
                                 c[i].h, c[i].w, kFpnChannels, MapKind::kInt8));
      PlaceMap(features.back(), image_,
               camera == 0 ? std::nullopt : std::optional<uint64_t>(0));
    }
    for (size_t i = 0; i < c.size(); ++i) {
      const Conv& conv = Weights(std::format("fpn{}", i));
      EmitConv(program_, image_, base_,
               {.name = std::format("fpn{}", i),
                .srcs = {MakeSrc(sum[i], conv, 1)},
                .bias = conv.bias,
                .out_i8 = &features[i],
                .relu = false});
    }
    return features;
  }

  Program& program_;
  DdrImage& image_;
  uint64_t base_;
  const Backbone& backbone_;
  Arena arena_;
};

}  // namespace

std::vector<std::byte> StemListBytes(uint64_t base, uint64_t image) {
  return ListBytes(StemLists(base, image));
}

std::vector<Instruction> StemQuantKernel() {
  vprog::Assembler assembler;
  assembler.LoadImmediate(12, kStemWords);
  assembler.LoadImmediate(28, 7);
  assembler.Emit({.rx = 1,
                  .rd = 3,
                  .rq = 4,
                  .rnw = 12,
                  .rn8 = 5,
                  .rimm2 = 28,
                  .map = {.stride = 2, .x = vprog::Map::kGroup},
                  .wr = vprog::Write::kAbuf,
                  .cmp = vprog::Cmp::kNe,
                  .cmp_left = vprog::CmpLeft::kColumn,
                  .sel = vprog::Sel::kZero});
  assembler.Ret();
  return assembler.Words(vprog::kRomWords);
}

Backbone ReadBackbone(const Sources& sources) {
  Backbone backbone{
      .stem = ReadMxint8(sources.quant, "img_backbone.conv1.weight"),
      .stem_bias = sources.quant.Float32Bits("img_backbone.conv1.bias"),
      .convs = {}};
  for (size_t i = 0; i < kStages.size(); ++i) {
    for (uint32_t j = 0; j < kStages[i].blocks; ++j) {
      const std::string tag = std::format("layer{}.{}", i + 1, j);
      for (const auto& [conv, kh] :
           {std::pair{"conv1", 1U}, {"conv2", 3U}, {"conv3", 1U}}) {
        backbone.convs.emplace(
            std::format("{}.{}", tag, conv),
            ReadConv(sources, std::format("img_backbone.{}.{}", tag, conv), kh,
                     true));
      }
      if (j == 0) {
        backbone.convs.emplace(
            tag + ".downsample.0",
            ReadConv(sources, std::format("img_backbone.{}.downsample.0", tag),
                     1, true));
      }
    }
  }
  for (size_t i = 0; i < kStages.size(); ++i) {
    backbone.convs.emplace(
        std::format("lateral{}", i),
        ReadConv(sources, std::format("img_neck.lateral_convs.{}.conv", i), 1,
                 false));
    backbone.convs.emplace(
        std::format("fpn{}", i),
        ReadConv(sources, std::format("img_neck.fpn_convs.{}.conv", i), 3,
                 false));
  }
  return backbone;
}

void EmitSingleConv(Program& program, DdrImage& image, uint64_t base,
                    const std::string& name, const FMap& x, const Conv& conv,
                    uint32_t stride, const FMap* out_f32, const FMap* out_i8,
                    bool relu) {
  EmitConv(program, image, base,
           {.name = name,
            .srcs = {MakeSrc(x, conv, stride)},
            .bias = conv.bias,
            .out_f32 = out_f32,
            .out_i8 = out_i8,
            .relu = relu});
}

std::vector<FMap> EmitEncoder(Program& program, DdrImage& image, uint64_t base,
                              const Backbone& backbone,
                              const CameraInput& input, uint32_t camera) {
  return EncoderEmitter(program, image, base, backbone).Emit(input, camera);
}

}  // namespace hotarubi::compiler::sparsedrive
