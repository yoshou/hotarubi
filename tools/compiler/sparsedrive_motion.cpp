#include "sparsedrive_motion.hpp"

#include <array>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/sparsedrive_layout.hpp"
#include "isa.hpp"
#include "kernels.hpp"
#include "program.hpp"
#include "sparsedrive_head.hpp"
#include "sparsedrive_kernels.hpp"
#include "sparsedrive_layers.hpp"
#include "sparsedrive_parts.hpp"
#include "tensors.hpp"
#include "vprog.hpp"
#include "weights.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

constexpr uint32_t kNa = 900;
constexpr uint32_t kQuantQuads = 32;
constexpr uint32_t kChunkKeys = kQueue * 32;
constexpr std::string_view kModule = "head.motion_plan_head";
constexpr std::array<int64_t, 8> kHalves = {0, 1, 2, 3, 4, 6, 8, 12};

std::vector<Op> MotionOrder() {
  const std::vector<Op> block = {Op::kTempGnn,  Op::kGnn,  Op::kNorm,
                                 Op::kCrossGnn, Op::kNorm, Op::kFfn,
                                 Op::kNorm};
  std::vector<Op> order;
  for (int i = 0; i < 3; ++i) {
    order.insert(order.end(), block.begin(), block.end());
  }
  order.push_back(Op::kRefine);
  return order;
}

std::string_view Name(Op op) {
  switch (op) {
    case Op::kGnn:
      return "gnn";
    case Op::kTempGnn:
      return "temp_gnn";
    case Op::kCrossGnn:
      return "cross_gnn";
    case Op::kNorm:
      return "norm";
    case Op::kDeformable:
      return "deformable";
    case Op::kFfn:
      return "ffn";
    case Op::kRefine:
      return "refine";
  }
  return "";
}

exact::Real F32(uint32_t bits) { return exact::Real::Binary32(bits); }

std::vector<uint32_t> SineEmbed(uint32_t x, uint32_t y) {
  constexpr uint32_t kHalf = 128;
  const uint32_t scale =
      exact::Real::Binary64(exact::Pi().ToBinary64()).Scaled(1).ToBinary32();
  std::vector<uint32_t> out;
  out.reserve(size_t{2} * kHalf);
  for (const uint32_t coordinate : {y, x}) {
    const exact::Real embed = F32((F32(coordinate) * F32(scale)).ToBinary32());
    for (uint32_t i = 0; i < kHalf; ++i) {
      const exact::Real dim =
          F32(exact::Pow(exact::Real::Integer(10000),
                         exact::Real::Ratio(i / 2, kHalf / 2))
                  .ToBinary32());
      const exact::Real angle = F32((embed / dim).ToBinary32());
      out.push_back(i % 2 == 0 ? exact::Sin(angle).ToBinary32()
                               : exact::Cos(angle).ToBinary32());
    }
  }
  return out;
}

std::vector<exact::Real> ExactLinear(const std::vector<uint32_t>& input,
                                     const Mxfp4Matrix& w,
                                     const std::vector<uint32_t>& bias) {
  const Mxint8Matrix a = EncodeMxint8(
      {.rows = 1, .cols = static_cast<uint32_t>(input.size()), .bits = input});
  const uint32_t blocks = a.cols / 32;
  std::vector<exact::Real> out;
  out.reserve(w.rows);
  for (uint32_t n = 0; n < w.rows; ++n) {
    exact::Real sum = F32(bias[n]);
    for (uint32_t b = 0; b < blocks; ++b) {
      int64_t dot = 0;
      for (uint32_t k = 0; k < 32; ++k) {
        const uint8_t code =
            w.code[(static_cast<size_t>(n) * w.cols) + (size_t{b} * 32) + k];
        const int64_t half =
            (code & 8U) != 0 ? -kHalves[code & 7U] : kHalves[code & 7U];
        dot += int64_t{a.q[(b * 32) + k]} * half;
      }
      sum = sum + exact::Real::Integer(dot).Scaled(
                      a.e[b] + w.e[(static_cast<size_t>(n) * blocks) + b] - 7);
    }
    out.push_back(sum);
  }
  return out;
}

std::vector<uint32_t> PlanModeQuery(const HeadWeights& weights) {
  const std::string module = std::format("{}.plan_anchor_encoder", kModule);
  const std::vector<uint32_t> anchor =
      weights.Values(std::format("{}.plan_anchor", kModule));
  const Mxfp4Matrix w1 = CanonicalMxfp4(weights.Fp4Rows(module + ".0"), 0, 256);
  const Mxfp4Matrix w2 = CanonicalMxfp4(weights.Fp4Rows(module + ".3"), 0, 256);
  const std::vector<uint32_t> b1 = weights.Values(module + ".0.bias");
  const std::vector<uint32_t> b2 = weights.Values(module + ".3.bias");
  const LayerNorm ln = weights.Norm(module + ".2");
  const exact::Real eps = F32(LnEpsBits());
  std::vector<uint32_t> out;
  for (uint32_t q = 0; q < kPlanQueries; ++q) {
    const size_t at = ((static_cast<size_t>(q) * 6) + 5) * 2;
    const std::vector<exact::Real> h =
        ExactLinear(SineEmbed(anchor[at], anchor[at + 1]), w1, b1);
    std::vector<exact::Real> relu;
    exact::Real mean = exact::Real::Integer(0);
    for (const exact::Real& value : h) {
      const exact::Real r = F32(value.ToBinary32());
      relu.push_back(r.Negative() ? exact::Real::Integer(0) : r);
      mean = mean + relu.back();
    }
    mean = mean / exact::Real::Integer(kD);
    exact::Real var = exact::Real::Integer(0);
    for (const exact::Real& value : relu) {
      var = var + ((value - mean) * (value - mean));
    }
    var = var / exact::Real::Integer(kD);
    const exact::Real rstd = exact::Real::Integer(1) / exact::Sqrt(var + eps);
    std::vector<uint32_t> normed;
    normed.reserve(kD);
    for (uint32_t i = 0; i < kD; ++i) {
      normed.push_back(
          ((((relu[i] - mean) * rstd) * F32(ln.g[i])) + F32(ln.b[i]))
              .ToBinary32());
    }
    for (const exact::Real& value : ExactLinear(normed, w2, b2)) {
      out.push_back(value.ToBinary32());
    }
  }
  return out;
}

}  // namespace

MotionParams ReadMotionParams(const HeadWeights& weights,
                              const Sources& sources) {
  MotionParams params;
  params.fc_before = weights.Linear(std::format("{}.fc_before", kModule));
  params.fc_after = weights.Linear(std::format("{}.fc_after", kModule));
  const std::vector<Op> order = MotionOrder();
  for (size_t i = 0; i < order.size(); ++i) {
    const std::string m = std::format("{}.layers.{}", kModule, i);
    MotionLayer layer;
    layer.op = order[i];
    switch (order[i]) {
      case Op::kTempGnn:
      case Op::kGnn:
      case Op::kCrossGnn: {
        layer.embed = static_cast<uint32_t>(
            weights.Ckpt().Info(m + ".attn.in_proj_bias").shape[0] / 3);
        layer.hd = layer.embed / kHeads;
        layer.attn =
            weights.Attention(m + ".attn", layer.embed, layer.hd == 64);
        break;
      }
      case Op::kNorm:
        layer.ln = weights.Norm(m);
        break;
      case Op::kFfn:
        layer.ln = weights.Norm(m + ".pre_norm");
        layer.fc1 = weights.Linear(m + ".layers.0.0");
        layer.fc2 = weights.Linear(m + ".layers.1");
        layer.idf = weights.Linear(m + ".identity_fc");
        break;
      case Op::kRefine:
        for (const auto& [key, branch, normed] :
             {std::tuple{"mcls", "motion_cls_branch", true},
              {"mreg", "motion_reg_branch", false},
              {"pcls", "plan_cls_branch", true},
              {"preg", "plan_reg_branch", false},
              {"pst", "plan_status_branch", false}}) {
          const std::string b = std::format("{}.{}", m, branch);
          if (normed) {
            layer.branch[key] = {weights.Linear(b + ".0"),
                                 weights.Linear(b + ".3")};
            layer.branch_ln[key] = {weights.Norm(b + ".2"),
                                    weights.Norm(b + ".5")};
            layer.branch_out[key] = weights.Linear(b + ".6", 32);
          } else {
            layer.branch[key] = {weights.Linear(b + ".0"),
                                 weights.Linear(b + ".2")};
            layer.branch_out[key] = weights.Linear(b + ".4", 32);
          }
        }
        break;
      case Op::kDeformable:
        Fail("the motion head has no deformable aggregation");
    }
    params.layers.push_back(std::move(layer));
  }
  const std::string mae = std::format("{}.motion_anchor_encoder", kModule);
  params.mae0 = weights.Linear(mae + ".0");
  params.mae_ln = weights.Norm(mae + ".2");
  params.mae1 = weights.Linear(mae + ".3");
  params.motion_anchor =
      weights.Values(std::format("{}.motion_anchor", kModule));
  params.plan_mode_query = PlanModeQuery(weights);
  params.ego_anchor =
      weights.Values(std::format("{}.instance_queue.ego_anchor", kModule));
  for (uint32_t i = 0; i < 2; ++i) {
    params.ego_conv[i] = ReadConv(
        sources,
        std::format("{}.instance_queue.ego_feature_encoder.{}", kModule, 2 * i),
        3, true);
  }
  return params;
}

MotionHead::MotionHead(DdrImage& image, uint64_t base, MotionParams params,
                       SparseHead& det, SparseHead& map, const FMap& f3)
    : image_(image),
      base_(base),
      p_(std::move(params)),
      det_(det),
      map_(map),
      f3_(f3) {
  for (uint32_t i = 0; i < 3; ++i) {
    x[i] = PlaceRows(image, std::format("{}.x{}", tag, i), kMotionQueries, kD);
  }
  e = PlaceRows(image, tag + ".E", kMotionQueries, kD);
  xe = PlaceRows(image, tag + ".xE", kMotionQueries, kD);
  anc = PlaceRows(image, tag + ".anchor", kMotionQueries, 32, 0);
  tx = PlaceRows(image, tag + ".temp_x", kTemporalKeys, kD, 0);
  tanc = PlaceRows(image, tag + ".temp_anchor", kTemporalKeys, 32, 0);
  tlist = image.Alloc(tag + ".tlist", uint64_t{kTemporalKeys} * 4);
  te = PlaceRows(image, tag + ".temp_E", kTemporalKeys, kD, 0);
  xsel = PlaceRows(image, tag + ".xsel", kSelectedPadded, kD, 0);
  esel = PlaceRows(image, tag + ".esel", kSelectedPadded, kD, 0);
  mxsel = PlaceRows(image, tag + ".mxsel", kMapSelectedPadded, kD, 0);
  mesel = PlaceRows(image, tag + ".mesel", kMapSelectedPadded, kD, 0);
  sel = image.Alloc(tag + ".sel", uint64_t{kSelectedPadded} * 4 * 2);
  msel = image.Alloc(tag + ".msel", uint64_t{kMapSelectedPadded} * 4 * 2);
  mask = image.Alloc(tag + ".mask",
                     uint64_t{kMotionQueries / 32} * mask_cs * kWordF);
  PutEntries(image, mask,
             std::vector<uint32_t>(
                 static_cast<size_t>(kMotionQueries / 32) * mask_cs * 32,
                 NegativeHugeBits()));
  tx_old = PlaceRows(image, tag + ".temp_x_old", kTemporalKeys, kD, 0);
  tanc_old = PlaceRows(image, tag + ".temp_anchor_old", kTemporalKeys, 32, 0);
  talist = image.Alloc(tag + ".talist", uint64_t{kTemporalKeys} * 4);
  st = PlaceRows(image, tag + ".state", kMotionQueries, 32, 0);
  std::vector<uint32_t> state(static_cast<size_t>(kMotionQueries) * 32);
  for (uint32_t i = 0; i < kMotionQueries; ++i) {
    state[(static_cast<size_t>(i) * 32) + 2] = FloatBits(i);
  }
  PutEntries(image, st.off, state);
  stlist = image.Alloc(tag + ".stlist", uint64_t{kMotionQueries} * 4);
  PutEntries(image, stlist, std::vector<uint32_t>(kMotionQueries, kNullEntry));
  ego_anc0 = PlaceRows(image, tag + ".ego_anchor0", 4, 32, 0);
  PutRows(image, ego_anc0, 1, static_cast<uint32_t>(p_.ego_anchor.size()),
          p_.ego_anchor);
  iq_.Add("state", IqueueState(det.ncls));
  iq_.Add("list", IqueueList());
  iq_.Add("stlist", IqueueStlist(kNa));
  iq_.Add("ego", IqueueEgo());
  const auto place_program = [&](const std::string& name,
                                 const CbufImage& cbuf) {
    const std::vector<std::byte> bytes = Serialize(cbuf.Words());
    const uint64_t offset = image.Alloc(name, bytes.size());
    image.Put(offset, bytes);
    return offset;
  };
  iqprog_ = place_program(tag + ".iqprog", iq_);
  const auto word = [&](std::string_view name, const Word& w) {
    const uint64_t offset =
        image.Alloc(std::format("{}.iq_{}", tag, name), kWordF);
    PutWords(image, offset, {w});
    return offset;
  };
  iq_cs_ = word("cs", IqueueConstWord(kNa, kQueue));
  iq_cx_ =
      word("cx", IqueueListWord(Entry(base_, tx_old.Addr(0)),
                                static_cast<uint32_t>(tx_old.Tb() / 16),
                                Entry(base_, x[0].Addr(0)),
                                static_cast<uint32_t>(x[0].Tb() / 16), kNa));
  iq_ca_ =
      word("ca", IqueueListWord(Entry(base_, tanc_old.Addr(0)),
                                static_cast<uint32_t>(tanc_old.Tb() / 16),
                                Entry(base_, anc.Addr(0)),
                                static_cast<uint32_t>(anc.Tb() / 16), kNa));
  iq_cl_ = word("cl", IqueueStlistWord(Entry(base_, st.Addr(0)),
                                       static_cast<uint32_t>(st.Tb() / 16),
                                       det.C().ntemp, kNa));
  iq_t_ = word("t", IqueueTWord());
  iq_zero_ = word("zero", Word{});
  mp_.Add("ego", EgoMeanKernel());
  mp_.Add("seld", InstanceBankKernel(det.C().nap, det.C().na, 0,
                                     BankMode::kSelect, kSelected - 1));
  mp_.Add("selm", InstanceBankKernel(map.C().nap, map.C().na, 0,
                                     BankMode::kSelect, kMapSelected));
  mpprog_ = place_program(tag + ".mpprog", mp_);
  std::vector<uint32_t> lists(size_t{2} * kSelectedPadded, kNullEntry);
  lists[kSelected - 1] = Entry(base_, x[0].Addr(900));
  lists[kSelectedPadded + kSelected - 1] = Entry(base_, e.Addr(900));
  PutEntries(image, sel, lists);
  PutEntries(image, msel,
             std::vector<uint32_t>(size_t{2} * kMapSelectedPadded, kNullEntry));
  ma_.Add("clsidx", MotionClsIdx(det.C().nap, det.ncls));
  ma_.Add("rot", MotionRot());
  ma_.Add("bc", MotionBcast(kModes));
  ma_.Add("compact", MotionCompact());
  ma_.Add("embed", MotionEmbed());
  maprog_ = place_program(tag + ".maprog", ma_);
  matab_ = PlaceRows(image, tag + ".matab", 12, 32, 0);
  std::vector<uint32_t> table(static_cast<size_t>(12) * 32);
  for (uint32_t c = 0; c < 10; ++c) {
    for (uint32_t m = 0; m < kModes; ++m) {
      const size_t last =
          ((((static_cast<size_t>(c) * kModes) + m) * 12) + 11) * 2;
      table[(static_cast<size_t>(c) * 32) + m] = p_.motion_anchor[last];
      table[(static_cast<size_t>(c) * 32) + 8 + m] = p_.motion_anchor[last + 1];
    }
  }
  PutEntries(image, matab_.off, table);
  ma_list_ = image.Alloc(tag + ".ma_list", uint64_t{det.C().nap / 32} * kWordF);
  ma_const_ = image.Alloc(tag + ".ma_const", kWordF);
  PutWords(image, ma_const_,
           {IbankConstWord(Entry(base_, matab_.Addr(0)),
                           static_cast<uint32_t>(matab_.Tb() / 16), 0, 1,
                           det.C().na, 0)});
  for (uint32_t m = 0; m < kModes; ++m) {
    rxm_[m] =
        PlaceRows(image, std::format("{}.rx{}", tag, m), det.C().nap, 32, 0);
  }
  for (uint32_t m = 0; m < kModes; ++m) {
    rym_[m] =
        PlaceRows(image, std::format("{}.ry{}", tag, m), det.C().nap, 32, 0);
  }
  std::vector<uint32_t> glist(2 * static_cast<size_t>(kModeQueriesPadded),
                              kNullEntry);
  for (uint32_t a = 0; a < det.C().na; ++a) {
    for (uint32_t m = 0; m < kModes; ++m) {
      glist[(a * kModes) + m] = Entry(base_, rxm_[m].Addr(a));
      glist[kModeQueriesPadded + (a * kModes) + m] =
          Entry(base_, rym_[m].Addr(a));
    }
  }
  ma_glist_ = image.Alloc(tag + ".ma_glist", uint64_t{kModeQueriesPadded} * 8);
  PutEntries(image, ma_glist_, glist);
  const std::vector<std::byte> sin_table = PackTbuf(SinTbuf());
  ma_tb_ = image.Alloc(tag + ".ma_tbuf", sin_table.size());
  image.Put(ma_tb_, sin_table);
  const std::vector<std::byte> embed = PackPbuf(EmbedPbuf());
  ma_pb_ = image.Alloc(tag + ".ma_pbuf", embed.size());
  image.Put(ma_pb_, embed);
  sine = PlaceRows(image, tag + ".sine", kModeQueriesPadded, kD, 0);
  modeq = PlaceRows(image, tag + ".modeq", kModeQueriesPadded, kD, 0);
  pmq = PlaceRows(image, tag + ".plan_modeq", kPlanQueriesPadded, kD, 0);
  PutRows(image, pmq, kPlanQueries, kD, p_.plan_mode_query);
  rep6_ = image.Alloc(tag + ".rep6", uint64_t{kModeQueriesPadded} * 4);
  rep_ego_ = image.Alloc(tag + ".rep_ego", uint64_t{kPlanQueriesPadded} * 4);
  mcls = PlaceRows(image, tag + ".motion_cls", kModeQueriesPadded, 32, 0);
  mreg = PlaceRows(image, tag + ".motion_reg", kModeQueriesPadded, 32, 0);
  pcls = PlaceRows(image, tag + ".plan_cls", kPlanQueriesPadded, 32, 0);
  preg = PlaceRows(image, tag + ".plan_reg", kPlanQueriesPadded, 32, 0);
  pst = PlaceRows(image, tag + ".plan_status", 32, 32, 0);
  qb_ = image.Alloc(tag + ".Q", uint64_t{kMotionQueries / 4} * 16 * kWordA);
  kb_ = image.Alloc(tag + ".K", uint64_t{kTemporalKeys / 4} * 16 * kWordA);
  vb_ = image.Alloc(tag + ".V", uint64_t{kTemporalKeys / 4} * 16 * kWordA);
  ob_ = image.Alloc(tag + ".O", uint64_t{kMotionQueries / 4} * 64 * kWordF);
  ego_t_ = MakeMap(tag + ".ego_t", f3.h, f3.w, kD, MapKind::kInt8);
  PlaceMap(ego_t_, image, 256);
  ego_o_ = MakeMap(tag + ".ego_out", f3.h / 2, f3.w / 2, kD, MapKind::kFloat);
  PlaceMap(ego_o_, image, 64);
  PlaceParams();
}

namespace {

constexpr std::array<std::string_view, 5> kBranches = {"mcls", "mreg", "pcls",
                                                       "preg", "pst"};

void PlaceLayerWeights(DdrImage& image, MotionLayer& layer,
                       const std::string& at) {
  switch (layer.op) {
    case Op::kTempGnn:
    case Op::kGnn:
    case Op::kCrossGnn:
      for (size_t i = 0; i < 4; ++i) {
        layer.attn[i].Place(image, at + std::string(1, "qkvo"[i]));
      }
      break;
    case Op::kFfn:
      layer.fc1.Place(image, at + "fc1");
      layer.fc2.Place(image, at + "fc2");
      layer.idf.Place(image, at + "idf");
      break;
    case Op::kRefine:
      for (const std::string_view key : kBranches) {
        auto& lins = layer.branch.find(key)->second;
        for (size_t i = 0; i < lins.size(); ++i) {
          lins[i].Place(image, std::format("{}{}{}", at, key, i));
        }
        layer.branch_out.find(key)->second.Place(
            image, std::format("{}{}_out", at, key));
      }
      break;
    case Op::kNorm:
    case Op::kDeformable:
      break;
  }
}

void AddRefineBiases(const MotionLayer& layer, const MotionParams& params,
                     const std::string& note, Pbuf& pb, Keys& keys) {
  keys["mae b0"] = pb.Add(params.mae0.Bias());
  keys["mae ln"] = pb.Norm(params.mae_ln);
  keys["mae b1"] = pb.Add(params.mae1.Bias());
  for (const std::string_view key : kBranches) {
    const auto& lins = layer.branch.find(key)->second;
    for (size_t i = 0; i < lins.size(); ++i) {
      keys[std::format("{} {}{}", note, key, i)] = pb.Add(lins[i].Bias());
    }
    keys[std::format("{} {}_out", note, key)] =
        pb.Add(layer.branch_out.find(key)->second.Bias());
  }
  for (const std::string_view key : {"mcls", "pcls"}) {
    const auto& norms = layer.branch_ln.find(key)->second;
    for (size_t i = 0; i < norms.size(); ++i) {
      keys[std::format("{} {}_ln{}", note, key, i)] = pb.Norm(norms[i]);
    }
  }
}

void AddLayerBiases(const MotionLayer& layer, const MotionParams& params,
                    const std::string& note, Pbuf& pb, Keys& keys) {
  switch (layer.op) {
    case Op::kTempGnn:
    case Op::kGnn:
    case Op::kCrossGnn:
      for (size_t i = 0; i < 4; ++i) {
        keys[std::format("{} {}", note, "qkvo"[i])] =
            pb.Add(layer.attn[i].Bias());
      }
      break;
    case Op::kNorm:
      keys[note] = pb.Norm(layer.ln);
      break;
    case Op::kFfn:
      keys[note + " ln"] = pb.Norm(layer.ln);
      keys[note + " fc1"] = pb.Add(layer.fc1.Bias());
      keys[note + " fc2"] = pb.Add(AddBits(layer.fc2.Bias(), layer.idf.Bias()));
      break;
    case Op::kRefine:
      AddRefineBiases(layer, params, note, pb, keys);
      break;
    case Op::kDeformable:
      break;
  }
}

}  // namespace

void MotionHead::PlaceParams() {
  p_.fc_before.Place(image_, tag + ".fc_before");
  p_.fc_after.Place(image_, tag + ".fc_after");
  p_.mae0.Place(image_, tag + ".mae.l0");
  p_.mae1.Place(image_, tag + ".mae.l1");
  for (size_t l = 0; l < p_.layers.size(); ++l) {
    PlaceLayerWeights(image_, p_.layers[l], std::format("{}.L{}.", tag, l));
  }
  for (size_t l = 0; l < p_.layers.size(); ++l) {
    const MotionLayer& layer = p_.layers[l];
    const std::string note = std::format("{} L{} {}", tag, l, Name(layer.op));
    Pbuf pb;
    Keys keys;
    AddLayerBiases(layer, p_, note, pb, keys);
    pb.Place(image_, std::format("{}.pbuf.L{}", tag, l));
    pb_layers_.emplace_back(std::move(pb), std::move(keys));
  }
  std::vector<uint32_t> rep6(kModeQueriesPadded, kNullEntry);
  for (uint32_t j = 0; j < kModeQueries; ++j) {
    rep6[j] = Entry(base_, xe.Addr(j / kModes));
  }
  PutEntries(image_, rep6_, rep6);
  std::vector<uint32_t> rep(kPlanQueriesPadded, kNullEntry);
  for (uint32_t j = 0; j < kPlanQueries; ++j) {
    rep[j] = Entry(base_, xe.Addr(900));
  }
  PutEntries(image_, rep_ego_, rep);
}

Place MotionHead::Pb(std::string_view key) const {
  return Params(pb_, *keys_).At(key);
}

void MotionHead::LoadRows(const Rows& rows, uint32_t j, uint32_t m, Place dst,
                          std::string_view note) {
  program_->Load(Tokens(rows.Addr(j), Mem::kFbuf, dst, m, rows.Tok()), note);
}

Tile MotionHead::LoadRows(const Rows& rows, uint32_t j, uint32_t m,
                          std::string_view note) {
  const Tile tile = program_->Alloc(Mem::kFbuf, m / 4 * (rows.cols / 8));
  LoadRows(rows, j, m, tile.At(), note);
  return tile;
}

void MotionHead::StoreRows(const Rows& rows, uint32_t j, uint32_t m, Place src,
                           std::string_view note) {
  program_->Store(Tokens(rows.Addr(j), Mem::kFbuf, src, m, rows.Tok()), note);
}

void MotionHead::QuantRows(const Rows& rows, uint32_t n, Place abuf,
                           std::string_view note, const Rows* second,
                           std::optional<Place> abuf_plain) {
  const uint32_t words = rows.cols / 8;
  QuantizeHook add;
  if (second != nullptr) {
    add = [&](const Tile& x, uint32_t quad, uint32_t quads) {
      const uint32_t t = quad / kQuantQuads;
      const Tile b = LoadRows(*second, 4 * quad, 4 * quads,
                              std::format("{} ld 2nd t{}", note, t));
      if (abuf_plain.has_value()) {
        program_->Vec({.op = VecOp::kCopy,
                       .src = x.At(),
                       .dst = *abuf_plain + (quad * (words / 4)),
                       .rows_q = quads,
                       .n8 = words,
                       .q = true},
                      std::format("{} Q(plain) t{}", note, t));
      }
      program_->Vec({.op = VecOp::kAdd,
                     .src = x.At(),
                     .src2 = b.At(),
                     .dst = x.At(),
                     .rows_q = quads,
                     .n8 = words},
                    std::format("{} add t{}", note, t));
    };
  }
  EmitQuantize(*program_,
               {.x = rows.Addr(0),
                .quads = n / 4,
                .words = words,
                .tile_quads = kQuantQuads,
                .tokens = true,
                .dst = abuf},
               note, add);
}

void MotionHead::ProjStore(const Lin& lin, const std::vector<Piece>& pieces,
                           uint32_t astride, uint32_t n, uint64_t out,
                           const std::string& key, std::string_view note) {
  const uint32_t blocks = lin.Groups() / 4;
  const Tile weights = lin.Load(*program_, note);
  for (const Band& band : Bands(n, 64)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile c = program_->Alloc(Mem::kFbuf, rq * lin.Groups());
    lin.Gemm(*program_, pieces, q0 * astride, astride, band.m, c.At(), weights,
             std::format("{} t{}", note, band.t));
    const Tile q = program_->Alloc(Mem::kAbuf, rq * blocks);
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = q.At(),
                   .rows_q = rq,
                   .n8 = lin.Groups(),
                   .p_base = Pb(key),
                   .q = true},
                  std::format("{} bias+Q t{}", note, band.t));
    program_->Store(Words(out + (uint64_t{q0} * blocks * kWordA), Mem::kAbuf,
                          q.At(), rq * blocks),
                    std::format("{} st t{}", note, band.t));
  }
}

void MotionHead::ProjStream(const Lin& lin, const Rows& rows_a,
                            const Rows* rows_b, uint32_t n, uint64_t out,
                            const std::string& key, std::string_view note) {
  const uint32_t blocks = lin.Groups() / 4;
  const Tile weights = lin.Load(*program_, note);
  for (const Band& band : Bands(n, 64)) {
    const uint32_t rq = band.m / 4;
    const Tile a = LoadRows(rows_a, band.i0, band.m,
                            std::format("{} ld a t{}", note, band.t));
    const Tile qa = program_->Alloc(Mem::kAbuf, rq * kKbD);
    program_->Vec({.op = VecOp::kCopy,
                   .src = a.At(),
                   .dst = qa.At(),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .q = true},
                  std::format("{} Q(a) t{}", note, band.t));
    std::vector<Piece> pieces;
    if (rows_b != nullptr) {
      const Tile b = LoadRows(*rows_b, band.i0, band.m,
                              std::format("{} ld b t{}", note, band.t));
      const Tile qb = program_->Alloc(Mem::kAbuf, rq * kKbD);
      program_->Vec({.op = VecOp::kCopy,
                     .src = b.At(),
                     .dst = qb.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .q = true},
                    std::format("{} Q(b) t{}", note, band.t));
      pieces = {{.a = qa.At(), .kb_offset = 0, .kb = kKbD},
                {.a = qb.At(), .kb_offset = kKbD, .kb = kKbD}};
    } else {
      pieces = {{.a = qa.At(), .kb_offset = 0, .kb = kKbD}};
    }
    const Tile c = program_->Alloc(Mem::kFbuf, rq * lin.Groups());
    lin.Gemm(*program_, pieces, 0, kKbD, band.m, c.At(), weights,
             std::format("{} t{}", note, band.t));
    const Tile q = program_->Alloc(Mem::kAbuf, rq * blocks);
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = q.At(),
                   .rows_q = rq,
                   .n8 = lin.Groups(),
                   .p_base = Pb(key),
                   .q = true},
                  std::format("{} bias+Q t{}", note, band.t));
    program_->Store(Words(out + (uint64_t{band.i0 / 4} * blocks * kWordA),
                          Mem::kAbuf, q.At(), rq * blocks),
                    std::format("{} st t{}", note, band.t));
  }
}

void MotionHead::ProjThrough(const Lin& lin, const Lin& pre, const Rows& rows,
                             uint32_t n, uint64_t out, const std::string& key,
                             std::string_view note) {
  constexpr uint32_t kChunk = 1024;
  const uint32_t blocks = lin.Groups() / 4;
  const uint32_t mid = pre.Groups() / 4;
  for (const Band& chunk : Bands(n, kChunk)) {
    const Tile hidden = program_->Alloc(Mem::kAbuf, chunk.m / 4 * mid);
    const Tile first =
        pre.Load(*program_, std::format("{} pre ch{}", note, chunk.t));
    for (const Band& band : Bands(chunk.m, 64)) {
      const uint32_t rq = band.m / 4;
      const Tile a =
          LoadRows(rows, chunk.i0 + band.i0, band.m,
                   std::format("{} ld ch{} t{}", note, chunk.t, band.t));
      const Tile qa = program_->Alloc(Mem::kAbuf, rq * kKbD);
      program_->Vec({.op = VecOp::kCopy,
                     .src = a.At(),
                     .dst = qa.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .q = true},
                    std::format("{} Q(a) ch{} t{}", note, chunk.t, band.t));
      const Tile c = program_->Alloc(Mem::kFbuf, rq * pre.Groups());
      pre.Gemm(*program_, pre.Input(qa.At()), 0, kKbD, band.m, c.At(), first,
               std::format("{} pre ch{} t{}", note, chunk.t, band.t));
      program_->Vec({.op = VecOp::kCopy,
                     .src = c.At(),
                     .dst = hidden.At((band.i0 / 4) * mid),
                     .rows_q = rq,
                     .n8 = pre.Groups(),
                     .q = true},
                    std::format("{} Q(pre) ch{} t{}", note, chunk.t, band.t));
    }
    const Tile second =
        lin.Load(*program_, std::format("{} ch{}", note, chunk.t));
    for (const Band& band : Bands(chunk.m, 64)) {
      const uint32_t rq = band.m / 4;
      const Tile c = program_->Alloc(Mem::kFbuf, rq * lin.Groups());
      lin.Gemm(*program_, lin.Input(hidden.At()), (band.i0 / 4) * mid, mid,
               band.m, c.At(), second,
               std::format("{} ch{} t{}", note, chunk.t, band.t));
      const Tile q = program_->Alloc(Mem::kAbuf, rq * blocks);
      program_->Vec({.op = VecOp::kAddP,
                     .src = c.At(),
                     .dst = q.At(),
                     .rows_q = rq,
                     .n8 = lin.Groups(),
                     .p_base = Pb(key),
                     .q = true},
                    std::format("{} bias+Q ch{} t{}", note, chunk.t, band.t));
      program_->Store(
          Words(out + (uint64_t{(chunk.i0 + band.i0) / 4} * blocks * kWordA),
                Mem::kAbuf, q.At(), rq * blocks),
          std::format("{} st ch{} t{}", note, chunk.t, band.t));
    }
  }
}

void MotionHead::AttentionCore(const MotionLayer& layer, std::string_view note,
                               uint32_t keys, uint32_t valid, bool chunk_local,
                               bool masked) {
  EmitAttention(
      *program_,
      {.q = qb_,
       .k = kb_,
       .v = vb_,
       .o = ob_,
       .heads = kHeads,
       .kv_heads = kHeads,
       .rows = kMotionQueries,
       .chunk_quads = 8,
       .embed = layer.embed,
       .kv_blocks = layer.embed / 32,
       .hd = layer.hd,
       .keys = chunk_local ? kChunkKeys : keys,
       .valid = {{.quad = 0, .keys = chunk_local ? kChunkKeys : valid}},
       .chunk_local = chunk_local,
       .scale = layer.hd != 64,
       .mask = masked ? std::optional<uint64_t>(mask) : std::nullopt,
       .mask_stride = mask_cs},
      note);
}

void MotionHead::OutProj(const MotionLayer& layer, std::string_view note,
                         bool decoupled) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  const uint32_t kbq = layer.embed / 32;
  const uint32_t nwq = layer.embed / 8;
  const Tile ao = program_->Alloc(Mem::kAbuf, kMotionQueries / 4 * kbq);
  for (const Band& band : Bands(kMotionQueries, 128)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile o = program_->Alloc(Mem::kFbuf, rq * nwq);
    program_->Load(Words(ob_ + (uint64_t{q0} * nwq * kWordF), Mem::kFbuf,
                         o.At(), rq * nwq),
                   std::format("{} ld O t{}", note, band.t));
    program_->Vec({.op = VecOp::kCopy,
                   .src = o.At(),
                   .dst = ao.At(q0 * kbq),
                   .rows_q = rq,
                   .n8 = nwq,
                   .q = true},
                  std::format("{} Q(O) t{}", note, band.t));
  }
  const Lin& op = layer.attn[3];
  const Tile weights = op.Load(*program_, std::format("{} out_proj", note));
  const Place pbo = Pb(std::format("{} o", note));
  if (decoupled) {
    const std::array<Tile, 2> af = {
        program_->Alloc(Mem::kAbuf, kMotionQueries / 4 * kKbD),
        program_->Alloc(Mem::kAbuf, kMotionQueries / 4 * kKbD)};
    for (uint32_t half = 0; half < 2; ++half) {
      const Rows& res = half == 0 ? xr : e;
      for (const Band& band : Bands(kMotionQueries, 64)) {
        const uint32_t q0 = band.i0 / 4;
        const uint32_t rq = band.m / 4;
        const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
        program_->Gemm({.a_base = ao.At(q0 * kbq),
                        .a_stride = kbq,
                        .kb = kbq,
                        .m = band.m,
                        .w_base = weights.At(half * 32 * op.Kb()),
                        .w_stride = op.Kb(),
                        .n8 = 32,
                        .c_base = c.At()},
                       std::format("{} out_proj h{} t{}", note, half, band.t));
        const Tile identity =
            LoadRows(res, band.i0, band.m,
                     std::format("{} ld identity h{} t{}", note, half, band.t));
        program_->Vec({.op = VecOp::kAddP,
                       .src = c.At(),
                       .dst = c.At(),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .p_base = pbo + (half * (kNwD / 4))},
                      std::format("{} bias h{} t{}", note, half, band.t));
        program_->Vec({.op = VecOp::kAdd,
                       .src = c.At(),
                       .src2 = identity.At(),
                       .dst = c.At(),
                       .rows_q = rq,
                       .n8 = kNwD},
                      std::format("{} + identity h{} t{}", note, half, band.t));
        program_->Vec({.op = VecOp::kCopy,
                       .src = c.At(),
                       .dst = af.at(half).At(q0 * kKbD),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .q = true},
                      std::format("{} Q h{} t{}", note, half, band.t));
      }
    }
    const Tile wa =
        p_.fc_after.Load(*program_, std::format("{} fc_after", note));
    for (const Band& band : Bands(kMotionQueries, 128)) {
      const Tile c =
          program_->Alloc(Mem::kFbuf, band.m / 4 * p_.fc_after.Groups());
      p_.fc_after.Gemm(*program_,
                       {{.a = af[0].At(), .kb_offset = 0, .kb = kKbD},
                        {.a = af[1].At(), .kb_offset = kKbD, .kb = kKbD}},
                       (band.i0 / 4) * kKbD, kKbD, band.m, c.At(), wa,
                       std::format("{} fc_after t{}", note, band.t));
      StoreRows(xo, band.i0, band.m, c.At(),
                std::format("{} st x t{}", note, band.t));
    }
  } else {
    for (const Band& band : Bands(kMotionQueries, 64)) {
      const uint32_t q0 = band.i0 / 4;
      const uint32_t rq = band.m / 4;
      const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
      op.Gemm(*program_, op.Input(ao.At()), q0 * kbq, kbq, band.m, c.At(),
              weights, std::format("{} out_proj t{}", note, band.t));
      const Tile identity = LoadRows(
          xr, band.i0, band.m, std::format("{} ld identity t{}", note, band.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c.At(),
                     .dst = c.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = pbo},
                    std::format("{} bias t{}", note, band.t));
      program_->Vec({.op = VecOp::kAdd,
                     .src = c.At(),
                     .src2 = identity.At(),
                     .dst = c.At(),
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} + identity t{}", note, band.t));
      StoreRows(xo, band.i0, band.m, c.At(),
                std::format("{} st x t{}", note, band.t));
    }
  }
  cur_x = (cur_x + 1) % 3;
}

void MotionHead::TempGnn(const MotionLayer& layer, std::string_view note) {
  ProjStream(layer.attn[0], x[cur_x], &e, kMotionQueries, qb_,
             std::format("{} q", note), std::format("{} q", note));
  ProjStream(layer.attn[1], tx, &te, kTemporalKeys, kb_,
             std::format("{} k", note), std::format("{} k", note));
  ProjThrough(layer.attn[2], p_.fc_before, tx, kTemporalKeys, vb_,
              std::format("{} v", note), std::format("{} v", note));
  AttentionCore(layer, note, kTemporalKeys, kTemporalKeys, true, true);
  OutProj(layer, note, true);
}

void MotionHead::Gnn(const MotionLayer& layer, std::string_view note) {
  ProjStream(layer.attn[0], x[cur_x], &e, kMotionQueries, qb_,
             std::format("{} q", note), std::format("{} q", note));
  ProjStream(layer.attn[1], xsel, &esel, kSelectedPadded, kb_,
             std::format("{} k", note), std::format("{} k", note));
  ProjThrough(layer.attn[2], p_.fc_before, xsel, kSelectedPadded, vb_,
              std::format("{} v", note), std::format("{} v", note));
  AttentionCore(layer, note, kSelectedPadded, kSelected, false, false);
  OutProj(layer, note, true);
}

void MotionHead::CrossGnn(const MotionLayer& layer, std::string_view note) {
  const Tile aq = program_->Alloc(Mem::kAbuf, kMotionQueries / 4 * kKbD);
  const Tile ak = program_->Alloc(Mem::kAbuf, kMapSelectedPadded / 4 * kKbD);
  const Tile av = program_->Alloc(Mem::kAbuf, kMapSelectedPadded / 4 * kKbD);
  QuantRows(x[cur_x], kMotionQueries, aq.At(), std::format("{} Q(x+E)", note),
            &e);
  QuantRows(mxsel, kMapSelectedPadded, ak.At(),
            std::format("{} Q(mx+mE)", note), &mesel, av.At());
  ProjStore(layer.attn[0], {{.a = aq.At(), .kb_offset = 0, .kb = kKbD}}, kKbD,
            kMotionQueries, qb_, std::format("{} q", note),
            std::format("{} q", note));
  ProjStore(layer.attn[1], {{.a = ak.At(), .kb_offset = 0, .kb = kKbD}}, kKbD,
            kMapSelectedPadded, kb_, std::format("{} k", note),
            std::format("{} k", note));
  ProjStore(layer.attn[2], {{.a = av.At(), .kb_offset = 0, .kb = kKbD}}, kKbD,
            kMapSelectedPadded, vb_, std::format("{} v", note),
            std::format("{} v", note));
  AttentionCore(layer, note, kMapSelectedPadded, kMapSelected, false, false);
  OutProj(layer, note, false);
}

void MotionHead::Norm(std::string_view note) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  for (const Band& band : Bands(kMotionQueries, 128)) {
    const uint32_t rq = band.m / 4;
    const Tile in =
        LoadRows(xr, band.i0, band.m, std::format("{} ld t{}", note, band.t));
    const Tile out = program_->Alloc(Mem::kFbuf, rq * kNwD);
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = in.At(),
                   .dst = out.At(),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = Pb(note),
                   .imm = LnEpsBits()},
                  std::format("{} LN t{}", note, band.t));
    StoreRows(xo, band.i0, band.m, out.At(),
              std::format("{} st t{}", note, band.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void MotionHead::Ffn(const MotionLayer& layer, std::string_view note) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  const Tile aff = program_->Alloc(Mem::kAbuf, kMotionQueries / 4 * kKbD);
  const Tile ah = program_->Alloc(Mem::kAbuf, kMotionQueries / 4 * 16);
  for (const Band& band : Bands(kMotionQueries, 128)) {
    const uint32_t rq = band.m / 4;
    const Tile in =
        LoadRows(xr, band.i0, band.m, std::format("{} ld x t{}", note, band.t));
    const Tile scratch = program_->Alloc(Mem::kFbuf, rq * kNwD);
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = in.At(),
                   .dst = aff.At((band.i0 / 4) * kKbD),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = Pb(std::format("{} ln", note)),
                   .imm = LnEpsBits(),
                   .q = true,
                   .scratch = scratch.At()},
                  std::format("{} LN+Q t{}", note, band.t));
  }
  const Tile w1 = layer.fc1.Load(*program_, std::format("{} fc1", note));
  const Tile w2 = layer.fc2.Load(*program_, std::format("{} fc2", note));
  for (const Band& band : Bands(kMotionQueries, 128)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile h = program_->Alloc(Mem::kFbuf, rq * layer.fc1.Groups());
    layer.fc1.Gemm(*program_, layer.fc1.Input(aff.At()), q0 * kKbD, kKbD,
                   band.m, h.At(), w1, std::format("{} fc1 t{}", note, band.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = h.At(),
                   .dst = h.At(),
                   .rows_q = rq,
                   .n8 = 64,
                   .p_base = Pb(std::format("{} fc1", note))},
                  std::format("{} fc1 bias t{}", note, band.t));
    program_->Vec({.op = VecOp::kRelu,
                   .src = h.At(),
                   .dst = ah.At(q0 * 16),
                   .rows_q = rq,
                   .n8 = 64,
                   .q = true},
                  std::format("{} fc1 relu+Q t{}", note, band.t));
  }
  const Tile wi = layer.idf.Load(*program_, std::format("{} idf", note));
  for (const Band& band : Bands(kMotionQueries, 128)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
    layer.fc2.Gemm(*program_, layer.fc2.Input(ah.At()), q0 * 16, 16, band.m,
                   c.At(), w2, std::format("{} fc2 t{}", note, band.t));
    layer.idf.Gemm(*program_, layer.idf.Input(aff.At()), q0 * kKbD, kKbD,
                   band.m, c.At(), wi,
                   std::format("{} identity_fc t{}", note, band.t), true);
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = Pb(std::format("{} fc2", note))},
                  std::format("{} bias t{}", note, band.t));
    StoreRows(xo, band.i0, band.m, c.At(),
              std::format("{} st x t{}", note, band.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void MotionHead::Mlp(const Tile& a_in, uint32_t n, const MotionLayer& layer,
                     const std::string& branch, bool relu_ln,
                     const Rows& rows_out, uint32_t row0,
                     std::string_view note) {
  constexpr uint32_t kTile = 224;
  const std::string bkey = std::format("{} {}", note, branch);
  const auto& lins = layer.branch.find(branch)->second;
  Tile src = a_in;
  for (size_t i = 0; i < lins.size(); ++i) {
    const Tile dst = program_->Alloc(Mem::kAbuf, n / 4 * kKbD);
    const std::string name = std::format("{}{}", bkey, i);
    const Tile weights = lins[i].Load(*program_, name);
    for (const Band& band : Bands(n, kTile)) {
      const uint32_t q0 = band.i0 / 4;
      const uint32_t rq = band.m / 4;
      const Tile c = program_->Alloc(Mem::kFbuf, rq * kNwD);
      lins[i].Gemm(*program_, lins[i].Input(src.At()), q0 * kKbD, kKbD, band.m,
                   c.At(), weights, std::format("{} t{}", name, band.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c.At(),
                     .dst = c.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = Pb(name)},
                    std::format("{} bias t{}", name, band.t));
      if (relu_ln) {
        program_->Vec({.op = VecOp::kRelu,
                       .src = c.At(),
                       .dst = c.At(),
                       .rows_q = rq,
                       .n8 = kNwD},
                      std::format("{} relu t{}", name, band.t));
        const Tile scratch = program_->Alloc(Mem::kFbuf, rq * kNwD);
        program_->Vec({.op = VecOp::kLayerNorm,
                       .src = c.At(),
                       .dst = dst.At(q0 * kKbD),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .p_base = Pb(std::format("{} {}_ln{}", note, branch, i)),
                       .imm = LnEpsBits(),
                       .q = true,
                       .scratch = scratch.At()},
                      std::format("{} LN+Q t{}", name, band.t));
      } else {
        program_->Vec({.op = VecOp::kRelu,
                       .src = c.At(),
                       .dst = dst.At(q0 * kKbD),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .q = true},
                      std::format("{} relu+Q t{}", name, band.t));
      }
    }
    src = dst;
  }
  const Lin& out = layer.branch_out.find(branch)->second;
  const Tile weights = out.Load(*program_, std::format("{} out", bkey));
  for (const Band& band : Bands(n, kTile)) {
    const uint32_t q0 = band.i0 / 4;
    const uint32_t rq = band.m / 4;
    const Tile c = program_->Alloc(Mem::kFbuf, rq * out.Groups());
    out.Gemm(*program_, out.Input(src.At()), q0 * kKbD, kKbD, band.m, c.At(),
             weights, std::format("{} out t{}", bkey, band.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = rq,
                   .n8 = out.Groups(),
                   .p_base = Pb(std::format("{}_out", bkey))},
                  std::format("{} out bias t{}", bkey, band.t));
    StoreRows(rows_out, row0 + band.i0, band.m, c.At(),
              std::format("{} out st t{}", bkey, band.t));
  }
}

void MotionHead::Refine(const MotionLayer& layer, std::string_view note) {
  const Rows& xr = x[cur_x];
  for (const Band& band : Bands(kMotionQueries, 128)) {
    const Tile rows =
        LoadRows(xr, band.i0, band.m, std::format("{} ld x t{}", note, band.t));
    const Tile embed =
        LoadRows(e, band.i0, band.m, std::format("{} ld E t{}", note, band.t));
    program_->Vec({.op = VecOp::kAdd,
                   .src = rows.At(),
                   .src2 = embed.At(),
                   .dst = rows.At(),
                   .rows_q = band.m / 4,
                   .n8 = kNwD},
                  std::format("{} x+E t{}", note, band.t));
    StoreRows(xe, band.i0, band.m, rows.At(),
              std::format("{} st xE t{}", note, band.t));
  }
  for (const Band& chunk : Bands(kModeQueriesPadded, kMotionQueries)) {
    const uint32_t words = chunk.m / 4 * kKbD;
    const Tile qs = program_->Alloc(Mem::kAbuf, words);
    for (const Band& band : Bands(chunk.m, 128)) {
      const Tile rows =
          LoadRows(sine, chunk.i0 + band.i0, band.m,
                   std::format("{} ld sine ch{} t{}", note, chunk.t, band.t));
      program_->Vec({.op = VecOp::kCopy,
                     .src = rows.At(),
                     .dst = qs.At((band.i0 / 4) * kKbD),
                     .rows_q = band.m / 4,
                     .n8 = kNwD,
                     .q = true},
                    std::format("{} Q(sine) t{}", note, band.t));
    }
    const Tile w0 = p_.mae0.Load(*program_, std::format("{} mae0", note));
    const Tile w1 = p_.mae1.Load(*program_, std::format("{} mae1", note));
    const Tile hidden = program_->Alloc(Mem::kAbuf, words);
    const Tile query = program_->Alloc(Mem::kAbuf, words);
    for (const Band& band : Bands(chunk.m, 128)) {
      const uint32_t q0 = band.i0 / 4;
      const uint32_t rq = band.m / 4;
      const Tile c0 = program_->Alloc(Mem::kFbuf, rq * kNwD);
      p_.mae0.Gemm(*program_, p_.mae0.Input(qs.At()), q0 * kKbD, kKbD, band.m,
                   c0.At(), w0,
                   std::format("{} mae0 ch{} t{}", note, chunk.t, band.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c0.At(),
                     .dst = c0.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = Pb("mae b0")},
                    std::format("{} mae0 bias t{}", note, band.t));
      program_->Vec({.op = VecOp::kRelu,
                     .src = c0.At(),
                     .dst = c0.At(),
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} mae0 relu t{}", note, band.t));
      const Tile scratch = program_->Alloc(Mem::kFbuf, rq * kNwD);
      program_->Vec({.op = VecOp::kLayerNorm,
                     .src = c0.At(),
                     .dst = hidden.At(q0 * kKbD),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = Pb("mae ln"),
                     .imm = LnEpsBits(),
                     .q = true,
                     .scratch = scratch.At()},
                    std::format("{} mae0 LN+Q t{}", note, band.t));
      const Tile c1 = program_->Alloc(Mem::kFbuf, rq * kNwD);
      p_.mae1.Gemm(*program_, p_.mae1.Input(hidden.At()), q0 * kKbD, kKbD,
                   band.m, c1.At(), w1,
                   std::format("{} mae1 ch{} t{}", note, chunk.t, band.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c1.At(),
                     .dst = c1.At(),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = Pb("mae b1")},
                    std::format("{} mae1 bias t{}", note, band.t));
      const Tile rep = program_->Alloc(Mem::kFbuf, rq * kNwD);
      program_->Load(
          Tokens(rep6_ + (uint64_t{chunk.i0 + band.i0} * 4), Mem::kFbuf,
                 rep.At(), band.m, kKbD, true),
          std::format("{} ld xE rep6 ch{} t{}", note, chunk.t, band.t));
      program_->Vec({.op = VecOp::kAdd,
                     .src = c1.At(),
                     .src2 = rep.At(),
                     .dst = c1.At(),
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} + xE t{}", note, band.t));
      program_->Vec({.op = VecOp::kCopy,
                     .src = c1.At(),
                     .dst = query.At(q0 * kKbD),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .q = true},
                    std::format("{} Q(motion_query) t{}", note, band.t));
    }
    Mlp(query, chunk.m, layer, "mcls", true, mcls, chunk.i0, note);
    Mlp(query, chunk.m, layer, "mreg", false, mreg, chunk.i0, note);
  }
  constexpr uint32_t kPlanQuads = kPlanQueriesPadded / 4;
  const Tile plan = LoadRows(pmq, 0, kPlanQueriesPadded,
                             std::format("{} ld plan_modeq", note));
  const Tile ego = program_->Alloc(Mem::kFbuf, kPlanQuads * kNwD);
  program_->Load(
      Tokens(rep_ego_, Mem::kFbuf, ego.At(), kPlanQueriesPadded, kKbD, true),
      std::format("{} ld xE[900] x18", note));
  program_->Vec({.op = VecOp::kAdd,
                 .src = plan.At(),
                 .src2 = ego.At(),
                 .dst = plan.At(),
                 .rows_q = kPlanQuads,
                 .n8 = kNwD},
                std::format("{} plan_query", note));
  const Tile plan_q = program_->Alloc(Mem::kAbuf, kPlanQuads * kKbD);
  program_->Vec({.op = VecOp::kCopy,
                 .src = plan.At(),
                 .dst = plan_q.At(),
                 .rows_q = kPlanQuads,
                 .n8 = kNwD,
                 .q = true},
                std::format("{} Q(plan_query)", note));
  Mlp(plan_q, kPlanQueriesPadded, layer, "pcls", true, pcls, 0, note);
  Mlp(plan_q, kPlanQueriesPadded, layer, "preg", false, preg, 0, note);
  const Tile ego_q = program_->Alloc(Mem::kAbuf, kPlanQuads * kKbD);
  program_->Vec({.op = VecOp::kCopy,
                 .src = ego.At(),
                 .dst = ego_q.At(),
                 .rows_q = kPlanQuads,
                 .n8 = kNwD,
                 .q = true},
                std::format("{} Q(xE[900])", note));
  Mlp(ego_q, kPlanQueriesPadded, layer, "pst", false, pst, 0, note);
}

void MotionHead::Queue() {
  constexpr uint32_t kQ = kMotionQueries / 4;
  constexpr uint32_t kNw = kMotionQueries / 32;
  program_->Load(Words(det_.pjprog, Mem::kCbuf, Place::Fixed(vprog::kRomWords),
                       static_cast<uint32_t>(det_.pj.Words().size())),
                 std::format("{} iq ld VEC program (anchor proj)", tag));
  const Tile project = program_->Alloc(Mem::kPbuf, 18);
  program_->Load(Words(det_.proj_pb, Mem::kPbuf, project.At(), 18),
                 std::format("{} iq ld PBUF (T)", tag));
  const Tile dt = program_->Alloc(Mem::kFbuf, 1);
  program_->Load(Words(iq_zero_, Mem::kFbuf, dt.At(), 1),
                 std::format("{} iq ld dt = 0", tag));
  for (const Band& band : Bands(kTemporalKeys, 1024)) {
    const Tile raw = LoadRows(tanc, band.i0, band.m,
                              std::format("{} iq ld anchors t{}", tag, band.t));
    const Tile out = program_->Alloc(Mem::kFbuf, band.m);
    program_->Run({.pc = det_.pj.Pc("project"),
                   .src = raw.At(),
                   .src2 = dt.At(),
                   .dst = out.At(),
                   .rows_q = band.m / 4,
                   .n8 = 4,
                   .p_base = project.At(),
                   .imm_places = {},
                   .reads = {Whole(raw), Whole(dt), Whole(out), Whole(project)},
                   .writes = {Whole(out)}},
                  std::format("{} iq anchor proj t{}", tag, band.t));
    StoreRows(tanc_old, band.i0, band.m, out.At(),
              std::format("{} iq st anchors t{}", tag, band.t));
  }
  for (const Band& band : Bands(kTemporalKeys, 128)) {
    const Tile rows = LoadRows(tx, band.i0, band.m,
                               std::format("{} iq ld temp_x t{}", tag, band.t));
    StoreRows(tx_old, band.i0, band.m, rows.At(),
              std::format("{} iq st temp_x t{}", tag, band.t));
  }
  program_->Load(Words(iqprog_, Mem::kCbuf, Place::Fixed(vprog::kRomWords),
                       static_cast<uint32_t>(iq_.Words().size())),
                 std::format("{} ld VEC programs (instance queue)", tag));
  const Tile ego =
      LoadRows(ego_anc0, 0, 4, std::format("{} iq ld ego anchor", tag));
  const Tile status =
      LoadRows(pst, 0, 4, std::format("{} iq ld plan_status", tag));
  program_->Run({.pc = iq_.Pc("ego"),
                 .src2 = status.At(),
                 .dst = ego.At(1),
                 .imm_places = {},
                 .reads = {Part(status, 0, 1), Part(ego, 1, 1)},
                 .writes = {Part(ego, 1, 1)}},
                std::format("{} iq ego anchor VY", tag));
  StoreRows(anc, 900, 4, ego.At(), std::format("{} iq st ego anchor", tag));
  const Tile t = program_->Alloc(Mem::kPbuf, 1);
  program_->Load(Words(iq_t_, Mem::kPbuf, t.At(), 1),
                 std::format("{} iq ld PBUF (t)", tag));
  const Tile state = program_->Alloc(Mem::kFbuf, 4 * kQ);
  program_->Load(
      Tokens(stlist, Mem::kFbuf, state.At(), kMotionQueries, 1, true),
      std::format("{} iq gather state", tag));
  const Tile own = program_->Alloc(Mem::kFbuf, 8 * kQ);
  LoadRows(det_.cls, 0, kMotionQueries, own.At(),
           std::format("{} iq ld cls", tag));
  LoadRows(st, 0, kMotionQueries, own.At(4 * kQ),
           std::format("{} iq ld own state", tag));
  const Tile masks = program_->Alloc(Mem::kFbuf, 2 * kQ);
  const Tile cs = program_->Alloc(Mem::kFbuf, 1);
  program_->Load(Words(iq_cs_, Mem::kFbuf, cs.At(), 1),
                 std::format("{} iq ld const", tag));
  program_->Run({.pc = iq_.Pc("state"),
                 .src = state.At(),
                 .src2 = own.At(),
                 .dst = masks.At(),
                 .rows_q = kQ,
                 .n8 = det_.ncls,
                 .p_base = t.At(),
                 .imm_places = {{.place = cs.At(), .shift = 0}},
                 .reads = {Whole(state), Whole(own), Whole(cs), Whole(t)},
                 .writes = {Whole(state), Whole(own), Whole(masks)}},
                std::format("{} iq state + mask", tag));
  StoreRows(st, 0, kMotionQueries, state.At(),
            std::format("{} iq st state", tag));
  for (uint32_t at = 0; at < 2; ++at) {
    program_->Store({.ddr_offset = mask + (uint64_t{at} * kWordF),
                     .mem = Mem::kFbuf,
                     .onchip = masks.At(at * kQ),
                     .rows = kQ,
                     .row_words = 1,
                     .ddr_stride = 18 * kWordF},
                    std::format("{} iq st mask {}", tag, at == 0 ? "A" : "B"));
  }
  for (const auto& [word, list, name] :
       {std::tuple{iq_cx_, tlist, "temp_x"},
        std::tuple{iq_ca_, talist, "temp_anchor"}}) {
    const Tile cl = program_->Alloc(Mem::kFbuf, 1);
    program_->Load(Words(word, Mem::kFbuf, cl.At(), 1),
                   std::format("{} iq ld const ({})", tag, name));
    const Tile out = program_->Alloc(Mem::kFbuf, kQ / 2);
    program_->Run({.pc = iq_.Pc("list"),
                   .src = state.At(),
                   .src2 = own.At(),
                   .dst = out.At(),
                   .rows_q = kQ,
                   .imm_places = {{.place = cl.At(), .shift = 0}},
                   .reads = {Whole(state), Whole(own), Whole(cl), Whole(out)},
                   .writes = {Whole(own), Whole(out)}},
                  std::format("{} iq list {}", tag, name));
    program_->Store(Words(list, Mem::kFbuf, out.At(), kQ / 2),
                    std::format("{} iq st list {}", tag, name));
  }
  const Tile selected = program_->Alloc(Mem::kFbuf, kNw);
  program_->Load(Words(det_.selidx, Mem::kFbuf, selected.At(), kNw),
                 std::format("{} iq ld selidx", tag));
  const Tile index = program_->Alloc(Mem::kFbuf, kNw);
  program_->Load(Words(det_.ib_idx, Mem::kFbuf, index.At(), kNw),
                 std::format("{} iq ld idx", tag));
  const Tile cl = program_->Alloc(Mem::kFbuf, 1);
  program_->Load(Words(iq_cl_, Mem::kFbuf, cl.At(), 1),
                 std::format("{} iq ld const (stlist)", tag));
  const Tile out = program_->Alloc(Mem::kFbuf, 2 * kNw);
  program_->Run(
      {.pc = iq_.Pc("stlist"),
       .src = selected.At(),
       .src2 = index.At(),
       .dst = out.At(),
       .rows_q = kNw,
       .imm_places = {{.place = cl.At(), .shift = 0}},
       .reads = {Whole(selected), Whole(index), Whole(cl), Whole(out)},
       .writes = {Whole(out)}},
      std::format("{} iq stlist", tag));
  program_->Store(Words(stlist, Mem::kFbuf, out.At(), kNw),
                  std::format("{} iq st stlist", tag));
  for (const Band& band : Bands(kTemporalKeys, 128)) {
    const Tile rows = program_->Alloc(Mem::kFbuf, band.m / 4 * kNwD);
    program_->Load(Tokens(tlist + (uint64_t{band.i0} * 4), Mem::kFbuf,
                          rows.At(), band.m, kKbD, true),
                   std::format("{} gather temp_x t{}", tag, band.t));
    StoreRows(tx, band.i0, band.m, rows.At(),
              std::format("{} st temp_x t{}", tag, band.t));
  }
  for (const Band& band : Bands(kTemporalKeys, 1024)) {
    const Tile rows = program_->Alloc(Mem::kFbuf, band.m);
    program_->Load(Tokens(talist + (uint64_t{band.i0} * 4), Mem::kFbuf,
                          rows.At(), band.m, 1, true),
                   std::format("{} gather temp_anchor t{}", tag, band.t));
    StoreRows(tanc, band.i0, band.m, rows.At(),
              std::format("{} st temp_anchor t{}", tag, band.t));
  }
}

void MotionHead::MotionAnchor() {
  const uint32_t nap = det_.C().nap;
  const uint32_t q = nap / 4;
  const uint32_t nw = nap / 32;
  const Access table = Fixed(Mem::kTbuf, 0, kTbufWords);
  program_->Load(Words(maprog_, Mem::kCbuf, Place::Fixed(vprog::kRomWords),
                       static_cast<uint32_t>(ma_.Words().size())),
                 std::format("{} ld VEC programs (motion anchor)", tag));
  const Tile cls = LoadRows(det_.cls, 0, nap, std::format("{} ma ld cls", tag));
  const Tile work = program_->Alloc(Mem::kFbuf, (4 * q) + 2);
  const Tile list = program_->Alloc(Mem::kFbuf, nw);
  const Tile cst = program_->Alloc(Mem::kFbuf, 1);
  program_->Load(Words(ma_const_, Mem::kFbuf, cst.At(), 1),
                 std::format("{} ma ld const", tag));
  program_->Run({.pc = ma_.Pc("clsidx"),
                 .src = cls.At(),
                 .src2 = work.At(),
                 .dst = list.At(),
                 .rows_q = q,
                 .n8 = det_.ncls,
                 .imm_places = {{.place = cst.At(), .shift = 0}},
                 .reads = {Whole(cls), Whole(work), Whole(list), Whole(cst)},
                 .writes = {Whole(work), Whole(list)}},
                std::format("{} ma clsidx", tag));
  program_->Store(Words(ma_list_, Mem::kFbuf, list.At(), nw),
                  std::format("{} ma st list", tag));
  const Tile anchors = LoadRows(det_.anc[det_.cur_a], 0, nap,
                                std::format("{} ma ld anchor", tag));
  const Tile rows = program_->Alloc(Mem::kFbuf, 4 * q);
  program_->Load(Tokens(ma_list_, Mem::kFbuf, rows.At(), nap, 1, true),
                 std::format("{} ma gather table rows", tag));
  const Tile rotated = program_->Alloc(Mem::kFbuf, 4 * q);
  program_->Run({.pc = ma_.Pc("rot"),
                 .src = anchors.At(),
                 .src2 = rows.At(),
                 .dst = rotated.At(),
                 .rows_q = q,
                 .imm_places = {},
                 .reads = {Whole(anchors), Whole(rows), Whole(rotated)},
                 .writes = {Whole(anchors), Whole(rows), Whole(rotated)}},
                std::format("{} ma rot", tag));
  for (uint32_t m = 0; m < kModes; ++m) {
    const Tile modes = program_->Alloc(Mem::kFbuf, 8 * q);
    program_->Run({.pc = ma_.Pc("bc"),
                   .src = rotated.At(),
                   .dst = modes.At(),
                   .rows_q = q,
                   .n8 = m,
                   .imm_places = {},
                   .reads = {Whole(rotated)},
                   .writes = {Whole(modes)}},
                  std::format("{} ma bcast m{}", tag, m));
    StoreRows(rxm_[m], 0, nap, modes.At(),
              std::format("{} ma st rx{}", tag, m));
    StoreRows(rym_[m], 0, nap, modes.At(4 * q),
              std::format("{} ma st ry{}", tag, m));
  }
  program_->Load(
      {.ddr_offset = ma_tb_,
       .mem = Mem::kTbuf,
       .onchip = Place::Fixed(0),
       .rows = 1,
       .row_words = kTbufWords,
       .ddr_stride = static_cast<uint32_t>(uint64_t{kTbufWords} * kWordT)},
      std::format("{} ma ld TBUF (sin)", tag));
  const Tile embed = program_->Alloc(Mem::kPbuf, 32);
  program_->Load(Words(ma_pb_, Mem::kPbuf, embed.At(), 32),
                 std::format("{} ma ld PBUF", tag));
  constexpr uint32_t kTq = 64;
  for (const Band& band : Bands(kModeQueriesPadded, 4 * kTq)) {
    const uint32_t count = band.m / 4;
    const Tile px = program_->Alloc(Mem::kFbuf, band.m);
    program_->Load(Tokens(ma_glist_ + (uint64_t{band.i0} * 4), Mem::kFbuf,
                          px.At(), band.m, 1, true),
                   std::format("{} ma gather posx t{}", tag, band.t));
    const Tile py = program_->Alloc(Mem::kFbuf, band.m);
    program_->Load(Tokens(ma_glist_ + (uint64_t{kModeQueriesPadded} * 4) +
                              (uint64_t{band.i0} * 4),
                          Mem::kFbuf, py.At(), band.m, 1, true),
                   std::format("{} ma gather posy t{}", tag, band.t));
    const Tile ply = program_->Alloc(Mem::kFbuf, count);
    program_->Run({.pc = ma_.Pc("compact"),
                   .src = py.At(),
                   .dst = ply.At(),
                   .rows_q = count,
                   .imm_places = {},
                   .reads = {Whole(py)},
                   .writes = {Whole(ply)}},
                  std::format("{} ma compact y t{}", tag, band.t));
    const Tile plx = program_->Alloc(Mem::kFbuf, count);
    program_->Run({.pc = ma_.Pc("compact"),
                   .src = px.At(),
                   .dst = plx.At(),
                   .rows_q = count,
                   .imm_places = {},
                   .reads = {Whole(px)},
                   .writes = {Whole(plx)}},
                  std::format("{} ma compact x t{}", tag, band.t));
    const Tile out = program_->Alloc(Mem::kFbuf, count * kNwD);
    program_->Run(
        {.pc = ma_.Pc("embed"),
         .src = ply.At(),
         .src2 = plx.At(),
         .dst = out.At(),
         .rows_q = count,
         .p_base = embed.At(),
         .imm_places = {},
         .reads = {Whole(ply), Whole(plx), Whole(out), Whole(embed), table},
         .writes = {Whole(out)}},
        std::format("{} ma embed t{}", tag, band.t));
    StoreRows(sine, band.i0, band.m, out.At(),
              std::format("{} ma st sine t{}", tag, band.t));
  }
  program_->Load(
      {.ddr_offset = det_.tb,
       .mem = Mem::kTbuf,
       .onchip = Place::Fixed(0),
       .rows = 1,
       .row_words = kTbufWords,
       .ddr_stride = static_cast<uint32_t>(uint64_t{kTbufWords} * kWordT)},
      std::format("{} ma ld TBUF (silu)", tag));
}

void MotionHead::PrepNpu() {
  program_->Load(Words(mpprog_, Mem::kCbuf, Place::Fixed(vprog::kRomWords),
                       static_cast<uint32_t>(mp_.Words().size())),
                 std::format("{} ld VEC programs (prep)", tag));
  constexpr uint32_t kEgoTokens = 44;
  const Tile ego = program_->Alloc(Mem::kFbuf, kEgoTokens / 4 * kNwD);
  program_->Load(
      Tokens(ego_o_.Addr(0, 0), Mem::kFbuf, ego.At(), kEgoTokens, kKbD),
      std::format("{} ego ld", tag));
  const Tile mean = program_->Alloc(Mem::kFbuf, kNwD);
  program_->Run({.pc = mp_.Pc("ego"),
                 .src = ego.At(),
                 .dst = mean.At(),
                 .imm = exact::Real::Binary64(
                            exact::Real::Ratio(1, kEgoTokens).ToBinary64())
                            .ToBinary32(),
                 .imm_places = {},
                 .reads = {Whole(ego), Whole(mean)},
                 .writes = {Whole(mean)}},
                std::format("{} ego mean", tag));
  StoreRows(x[0], 900, 4, mean.At(),
            std::format("{} ego st rows 900..903", tag));
  for (int which = 0; which < 2; ++which) {
    SparseHead& h = which == 0 ? det_ : map_;
    const std::string name = which == 0 ? "seld" : "selm";
    const uint64_t list = which == 0 ? sel : msel;
    const uint32_t nl = which == 0 ? kSelectedPadded : kMapSelectedPadded;
    const std::string note = which == 0 ? "sel det" : "sel map";
    const uint32_t nap = h.C().nap;
    const uint32_t nw = nap / 32;
    const uint32_t work = nap;
    const uint32_t out = work + (6 * nw) + 3;
    const uint32_t cst = out + (2 * nw);
    const uint32_t nlw = nl / 32;
    const auto key = std::pair{which, h.cur_x};
    if (!mp_const_.contains(key)) {
      const uint64_t offset = image_.Alloc(
          std::format("{}.mp_const.{}{}", tag, name, h.cur_x), kWordF);
      PutWords(
          image_, offset,
          {IbankConstWord(Entry(base_, h.x[h.cur_x].Addr(0)),
                          static_cast<uint32_t>(h.x[h.cur_x].Tb() / 16),
                          Entry(base_, h.e.Addr(0)),
                          static_cast<uint32_t>(h.e.Tb() / 16), h.C().na, 0)});
      mp_const_[key] = offset;
    }
    const Tile area = program_->Alloc(Mem::kFbuf, cst + 1);
    LoadRows(h.cls, 0, nap, area.At(), std::format("{} {} ld cls", tag, note));
    program_->Load(Words(h.ib_idx, Mem::kFbuf, area.At(work + nw), nw),
                   std::format("{} {} ld idx", tag, note));
    program_->Load(
        Words(h.ib_lane, Mem::kFbuf, area.At(work + (6 * nw) + 2), 1),
        std::format("{} {} ld lane", tag, note));
    program_->Load(Words(list, Mem::kFbuf, area.At(out), nlw),
                   std::format("{} {} ld list x", tag, note));
    program_->Load(
        Words(list + (uint64_t{nl} * 4), Mem::kFbuf, area.At(out + nw), nlw),
        std::format("{} {} ld list E", tag, note));
    program_->Load(Words(mp_const_.at(key), Mem::kFbuf, area.At(cst), 1),
                   std::format("{} {} ld const", tag, note));
    program_->Run(
        {.pc = mp_.Pc(name),
         .src = area.At(),
         .src2 = area.At(work),
         .dst = area.At(out),
         .rows_q = nap / 4,
         .n8 = h.ncls,
         .imm_places = {{.place = area.At(cst), .shift = 0}},
         .reads = {Part(area, 0, cst + 1), Fixed(Mem::kTbuf, 0, kTbufWords)},
         .writes = {Part(area, work, cst - work)}},
        std::format("{} {} run", tag, note));
    program_->Store(Words(list, Mem::kFbuf, area.At(out), nlw),
                    std::format("{} {} st list x", tag, note));
    program_->Store(
        Words(list + (uint64_t{nl} * 4), Mem::kFbuf, area.At(out + nw), nlw),
        std::format("{} {} st list E", tag, note));
  }
  for (const Band& band : Bands(900, 128)) {
    const uint32_t m4 = (band.m + 3) / 4 * 4;
    const Tile rows =
        LoadRows(det_.anc[det_.cur_a], band.i0, m4,
                 std::format("{} ld det anchor t{}", tag, band.t));
    StoreRows(anc, band.i0, m4, rows.At(),
              std::format("{} st anchor t{}", tag, band.t));
  }
  MotionAnchor();
}

void MotionHead::Prep() {
  EmitSingleConv(*program_, image_, base_, tag + ".ego0", f3_, p_.ego_conv[0],
                 1, nullptr, &ego_t_, false);
  EmitSingleConv(*program_, image_, base_, tag + ".ego1", ego_t_,
                 p_.ego_conv[1], 2, &ego_o_, nullptr, true);
  PrepNpu();
  const Rows& xd = det_.x[det_.cur_x];
  for (const Band& band : Bands(900, 128)) {
    const uint32_t m4 = (band.m + 3) / 4 * 4;
    const Tile rows =
        LoadRows(xd, band.i0, m4, std::format("{} ld det x t{}", tag, band.t));
    StoreRows(x[0], band.i0, m4, rows.At(),
              std::format("{} st x t{}", tag, band.t));
  }
  cur_x = 0;
  Queue();
  det_.EncodeAnchors(*program_, anc, e, kMotionQueries,
                     std::format("{} E", tag));
  det_.EncodeAnchors(*program_, tanc, te, kTemporalKeys,
                     std::format("{} temp_E", tag));
  const std::array<std::tuple<uint64_t, const Rows*, uint32_t>, 4> gathers = {
      {{sel, &xsel, kSelectedPadded},
       {sel + (uint64_t{kSelectedPadded} * 4), &esel, kSelectedPadded},
       {msel, &mxsel, kMapSelectedPadded},
       {msel + (uint64_t{kMapSelectedPadded} * 4), &mesel,
        kMapSelectedPadded}}};
  for (size_t i = 0; i < gathers.size(); ++i) {
    const auto& [list, rows, n] = gathers[i];
    const Tile gathered = program_->Alloc(Mem::kFbuf, n / 4 * kNwD);
    program_->Load(Tokens(list, Mem::kFbuf, gathered.At(), n, kKbD, true),
                   std::format("{} gather sel {}", tag, i));
    StoreRows(*rows, 0, n, gathered.At(), std::format("{} st sel {}", tag, i));
  }
}

void MotionHead::EmitFrame(Program& program) {
  program_ = &program;
  {
    const Program::Scope scope(program, "prep");
    Prep();
  }
  for (size_t l = 0; l < p_.layers.size(); ++l) {
    const MotionLayer& layer = p_.layers[l];
    const std::string note = std::format("{} L{} {}", tag, l, Name(layer.op));
    const Program::Scope scope(program,
                               std::format("L{}.{}", l, Name(layer.op)));
    pb_ = pb_layers_[l].first.Load(program, note);
    keys_ = &pb_layers_[l].second;
    switch (layer.op) {
      case Op::kTempGnn:
        TempGnn(layer, note);
        break;
      case Op::kGnn:
        Gnn(layer, note);
        break;
      case Op::kCrossGnn:
        CrossGnn(layer, note);
        break;
      case Op::kNorm:
        Norm(note);
        break;
      case Op::kFfn:
        Ffn(layer, note);
        break;
      case Op::kRefine:
        Refine(layer, note);
        break;
      case Op::kDeformable:
        break;
    }
  }
  const Program::Scope scope(program, "end");
  const Rows& xr = x[cur_x];
  const std::string note = std::format("{} cache ego feature", tag);
  const Tile feature = program.Alloc(Mem::kFbuf, kD / 32);
  program.Load(Words(xr.Addr(900), Mem::kFbuf, feature.At(), kD / 32),
               std::format("{} ld", note));
  program.Store(Words(tx.Addr((900 * kQueue) + kQueue - 1), Mem::kFbuf,
                      feature.At(), kD / 32),
                std::format("{} st", note));
  program.Signal(::hotarubi::sparsedrive::kMotionDoneSignal,
                 ::hotarubi::sparsedrive::kMotionDoneSignal,
                 std::format("{} SIGNAL motion done", tag));
  program.WaitHost(::hotarubi::sparsedrive::kMotionDoneSignal,
                   std::format("{} WAIT_HOST", tag));
}

}  // namespace hotarubi::compiler::sparsedrive
