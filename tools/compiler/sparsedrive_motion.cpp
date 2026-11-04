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
#include "exact.hpp"
#include "isa.hpp"
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
  PutEntries(image, stlist, std::vector<uint32_t>(kMotionQueries, kNull));
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
  std::vector<uint32_t> lists(size_t{2} * kSelectedPadded, kNull);
  lists[kSelected - 1] = Entry(base_, x[0].Addr(900));
  lists[kSelectedPadded + kSelected - 1] = Entry(base_, e.Addr(900));
  PutEntries(image, sel, lists);
  PutEntries(image, msel,
             std::vector<uint32_t>(size_t{2} * kMapSelectedPadded, kNull));
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
                              kNull);
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
  std::vector<uint32_t> rep6(kModeQueriesPadded, kNull);
  for (uint32_t j = 0; j < kModeQueries; ++j) {
    rep6[j] = Entry(base_, xe.Addr(j / kModes));
  }
  PutEntries(image_, rep6_, rep6);
  std::vector<uint32_t> rep(kPlanQueriesPadded, kNull);
  for (uint32_t j = 0; j < kPlanQueries; ++j) {
    rep[j] = Entry(base_, xe.Addr(900));
  }
  PutEntries(image_, rep_ego_, rep);
}

void MotionHead::LoadRows(const Rows& rows, uint32_t j, uint32_t m,
                          uint32_t dst, std::string_view note) {
  program_->Load(Tokens(rows.Addr(j), Mem::kFbuf, dst, m, rows.Tok()), note);
}

void MotionHead::StoreRows(const Rows& rows, uint32_t j, uint32_t m,
                           uint32_t src, std::string_view note) {
  program_->Store(Tokens(rows.Addr(j), Mem::kFbuf, src, m, rows.Tok()), note);
}

void MotionHead::QuantRows(const Rows& rows, uint32_t n, uint32_t abuf,
                           std::string_view note, const Rows* second,
                           std::optional<uint32_t> abuf_plain) {
  constexpr std::array<uint32_t, 2> kSlots = {0, 1024};
  constexpr std::array<uint32_t, 2> kSecond = {2048, 3072};
  const uint32_t words = rows.cols / 8;
  const uint32_t blocks = rows.cols / 32;
  for (const Tile& tile : Tiles(n, 128)) {
    const uint32_t slot = kSlots[tile.t % 2];
    const uint32_t rq = tile.m / 4;
    LoadRows(rows, tile.i0, tile.m, slot,
             std::format("{} ld t{}", note, tile.t));
    if (second != nullptr) {
      LoadRows(*second, tile.i0, 4 * rq, kSecond[tile.t % 2],
               std::format("{} ld 2nd t{}", note, tile.t));
      if (abuf_plain.has_value()) {
        program_->Vec({.op = VecOp::kQuant,
                       .src = slot,
                       .dst = *abuf_plain + ((tile.i0 / 4) * blocks),
                       .rows_q = rq,
                       .n8 = words},
                      std::format("{} Q(plain) t{}", note, tile.t));
      }
      program_->Vec({.op = VecOp::kAdd,
                     .src = slot,
                     .src2 = kSecond[tile.t % 2],
                     .dst = slot,
                     .rows_q = rq,
                     .n8 = words},
                    std::format("{} add t{}", note, tile.t));
    }
    program_->Vec({.op = VecOp::kQuant,
                   .src = slot,
                   .dst = abuf + ((tile.i0 / 4) * blocks),
                   .rows_q = rq,
                   .n8 = words},
                  std::format("{} quant t{}", note, tile.t));
  }
}

void MotionHead::ProjStore(const Lin& lin, const std::vector<Piece>& pieces,
                           uint32_t astride, uint32_t n, uint64_t out,
                           const std::string& key, std::string_view note) {
  constexpr std::array<uint32_t, 2> kSlots = {0, 1024};
  constexpr std::array<uint32_t, 2> kTmp = {7424, 7680};
  const uint32_t blocks = lin.Groups() / 4;
  lin.Load(*program_, 0, note);
  for (const Tile& tile : Tiles(n, 64)) {
    const uint32_t c = kSlots[tile.t % 2];
    const uint32_t q0 = tile.i0 / 4;
    lin.Gemm(*program_, q0 * astride, astride, tile.m, c, 0,
             std::format("{} t{}", note, tile.t), pieces);
    program_->Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = kTmp[tile.t % 2],
                   .rows_q = tile.m / 4,
                   .n8 = lin.Groups(),
                   .p_base = keys_.at(key),
                   .q = true},
                  std::format("{} bias+Q t{}", note, tile.t));
    program_->Store(Words(out + (uint64_t{q0} * blocks * kWordA), Mem::kAbuf,
                          kTmp[tile.t % 2], (tile.m / 4) * blocks),
                    std::format("{} st t{}", note, tile.t));
  }
}

void MotionHead::ProjStream(const Lin& lin, const Rows& rows_a,
                            const Rows* rows_b, uint32_t n, uint64_t out,
                            const std::string& key, std::string_view note,
                            const Lin* pre) {
  const uint32_t blocks = lin.Groups() / 4;
  constexpr uint32_t kA1 = 0;
  constexpr uint32_t kA2 = 16 * kKbD;
  constexpr uint32_t kAf = 32 * kKbD;
  const std::array<uint32_t, 2> tmp = {48 * kKbD, (48 * kKbD) + (16 * blocks)};
  lin.Load(*program_, 0, note);
  if (pre != nullptr) {
    pre->Load(*program_, 1024, std::format("{} pre", note));
  }
  for (const Tile& tile : Tiles(n, 64)) {
    const uint32_t rq = tile.m / 4;
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    LoadRows(rows_a, tile.i0, tile.m, slot,
             std::format("{} ld a t{}", note, tile.t));
    program_->Vec({.op = VecOp::kQuant,
                   .src = slot,
                   .dst = kA1,
                   .rows_q = rq,
                   .n8 = kNwD},
                  std::format("{} Q(a) t{}", note, tile.t));
    std::vector<Piece> pieces;
    uint32_t astride = kKbD;
    if (pre != nullptr) {
      pre->Gemm(*program_, kA1, kKbD, tile.m, 2048, 1024,
                std::format("{} pre t{}", note, tile.t));
      program_->Vec({.op = VecOp::kQuant,
                     .src = 2048,
                     .dst = kAf,
                     .rows_q = rq,
                     .n8 = pre->Groups()},
                    std::format("{} Q(pre) t{}", note, tile.t));
      astride = pre->Groups() / 4;
      pieces = {{.a_offset = kAf, .kb_offset = 0, .kb = astride}};
    } else if (rows_b != nullptr) {
      LoadRows(*rows_b, tile.i0, tile.m, 2048,
               std::format("{} ld b t{}", note, tile.t));
      program_->Vec({.op = VecOp::kQuant,
                     .src = 2048,
                     .dst = kA2,
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} Q(b) t{}", note, tile.t));
      pieces = {{.a_offset = kA1, .kb_offset = 0, .kb = kKbD},
                {.a_offset = kA2, .kb_offset = kKbD, .kb = kKbD}};
    } else {
      pieces = {{.a_offset = kA1, .kb_offset = 0, .kb = kKbD}};
    }
    lin.Gemm(*program_, 0, astride, tile.m, 3072, 0,
             std::format("{} t{}", note, tile.t), pieces);
    program_->Vec({.op = VecOp::kAddP,
                   .src = 3072,
                   .dst = tmp[tile.t % 2],
                   .rows_q = rq,
                   .n8 = lin.Groups(),
                   .p_base = keys_.at(key),
                   .q = true},
                  std::format("{} bias+Q t{}", note, tile.t));
    program_->Store(Words(out + (uint64_t{tile.i0 / 4} * blocks * kWordA),
                          Mem::kAbuf, tmp[tile.t % 2], rq * blocks),
                    std::format("{} st t{}", note, tile.t));
  }
}

void MotionHead::AttentionCore(const MotionLayer& layer, std::string_view note,
                               uint32_t keys, uint32_t valid, bool chunk_local,
                               bool masked) {
  EmitAttention(*program_,
                {.q = qb_,
                 .k = kb_,
                 .v = vb_,
                 .o = ob_,
                 .queries = kMotionQueries,
                 .embed = layer.embed,
                 .hd = layer.hd,
                 .keys = keys,
                 .valid = valid,
                 .chunk_local = chunk_local,
                 .masked = masked,
                 .mask = mask,
                 .mask_cs = mask_cs},
                note);
}

void MotionHead::OutProj(const MotionLayer& layer, std::string_view note,
                         bool decoupled) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  const uint32_t kbq = layer.embed / 32;
  const uint32_t nwq = layer.embed / 8;
  OnChip a(kAbufWords);
  const uint32_t ao = a.Take(kMotionQueries / 4 * kbq);
  const std::array<uint32_t, 2> ot = nwq == 64
                                         ? std::array<uint32_t, 2>{0, 2048}
                                         : std::array<uint32_t, 2>{0, 1024};
  for (const Tile& tile : Tiles(kMotionQueries, 128)) {
    const uint32_t slot = ot[tile.t % 2];
    const uint32_t q0 = tile.i0 / 4;
    program_->Load(Words(ob_ + (uint64_t{q0} * nwq * kWordF), Mem::kFbuf, slot,
                         (tile.m / 4) * nwq),
                   std::format("{} ld O t{}", note, tile.t));
    program_->Vec({.op = VecOp::kQuant,
                   .src = slot,
                   .dst = ao + (q0 * kbq),
                   .rows_q = tile.m / 4,
                   .n8 = nwq},
                  std::format("{} Q(O) t{}", note, tile.t));
  }
  const Lin& op = layer.attn[3];
  op.Load(*program_, 0, std::format("{} out_proj", note));
  constexpr std::array<uint32_t, 2> kCt = {0, 512};
  constexpr std::array<uint32_t, 2> kXt = {1024, 1536};
  const uint32_t pbo = keys_.at(std::format("{} o", note));
  if (decoupled) {
    const std::array<uint32_t, 2> af = {a.Take(kMotionQueries / 4 * kKbD),
                                        a.Take(kMotionQueries / 4 * kKbD)};
    for (uint32_t half = 0; half < 2; ++half) {
      const Rows& res = half == 0 ? xr : e;
      for (const Tile& tile : Tiles(kMotionQueries, 64)) {
        const uint32_t c = kCt[tile.t % 2];
        const uint32_t xt = kXt[tile.t % 2];
        const uint32_t q0 = tile.i0 / 4;
        program_->Gemm({.a_base = ao + (q0 * kbq),
                        .a_stride = kbq,
                        .kb = kbq,
                        .m = tile.m,
                        .w_base = half * 32 * op.Kb(),
                        .w_stride = op.Kb(),
                        .n8 = 32,
                        .c_base = c},
                       std::format("{} out_proj h{} t{}", note, half, tile.t));
        LoadRows(res, tile.i0, tile.m, xt,
                 std::format("{} ld identity h{} t{}", note, half, tile.t));
        program_->Vec({.op = VecOp::kAddP,
                       .src = c,
                       .dst = c,
                       .rows_q = tile.m / 4,
                       .n8 = kNwD,
                       .p_base = pbo + (half * (kNwD / 4))},
                      std::format("{} bias h{} t{}", note, half, tile.t));
        program_->Vec({.op = VecOp::kAdd,
                       .src = c,
                       .src2 = xt,
                       .dst = c,
                       .rows_q = tile.m / 4,
                       .n8 = kNwD},
                      std::format("{} + identity h{} t{}", note, half, tile.t));
        program_->Vec({.op = VecOp::kQuant,
                       .src = c,
                       .dst = af[half] + (q0 * kKbD),
                       .rows_q = tile.m / 4,
                       .n8 = kNwD},
                      std::format("{} Q h{} t{}", note, half, tile.t));
      }
    }
    p_.fc_after.Load(*program_, 0, std::format("{} fc_after", note));
    for (const Tile& tile : Tiles(kMotionQueries, 128)) {
      const uint32_t c = tile.t % 2 == 0 ? 0 : 1024;
      p_.fc_after.Gemm(*program_, (tile.i0 / 4) * kKbD, kKbD, tile.m, c, 0,
                       std::format("{} fc_after t{}", note, tile.t),
                       {{.a_offset = af[0], .kb_offset = 0, .kb = kKbD},
                        {.a_offset = af[1], .kb_offset = kKbD, .kb = kKbD}});
      StoreRows(xo, tile.i0, tile.m, c,
                std::format("{} st x t{}", note, tile.t));
    }
  } else {
    for (const Tile& tile : Tiles(kMotionQueries, 64)) {
      const uint32_t c = kCt[tile.t % 2];
      const uint32_t xt = kXt[tile.t % 2];
      const uint32_t q0 = tile.i0 / 4;
      op.Gemm(*program_, ao + (q0 * kbq), kbq, tile.m, c, 0,
              std::format("{} out_proj t{}", note, tile.t));
      LoadRows(xr, tile.i0, tile.m, xt,
               std::format("{} ld identity t{}", note, tile.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = c,
                     .dst = c,
                     .rows_q = tile.m / 4,
                     .n8 = kNwD,
                     .p_base = pbo},
                    std::format("{} bias t{}", note, tile.t));
      program_->Vec({.op = VecOp::kAdd,
                     .src = c,
                     .src2 = xt,
                     .dst = c,
                     .rows_q = tile.m / 4,
                     .n8 = kNwD},
                    std::format("{} + identity t{}", note, tile.t));
      StoreRows(xo, tile.i0, tile.m, c,
                std::format("{} st x t{}", note, tile.t));
    }
  }
  cur_x = (cur_x + 1) % 3;
}

void MotionHead::TempGnn(const MotionLayer& layer, std::string_view note) {
  ProjStream(layer.attn[0], x[cur_x], &e, kMotionQueries, qb_,
             std::format("{} q", note), std::format("{} q", note));
  ProjStream(layer.attn[1], tx, &te, kTemporalKeys, kb_,
             std::format("{} k", note), std::format("{} k", note));
  ProjStream(layer.attn[2], tx, nullptr, kTemporalKeys, vb_,
             std::format("{} v", note), std::format("{} v", note),
             &p_.fc_before);
  AttentionCore(layer, note, kTemporalKeys, kTemporalKeys, true, true);
  OutProj(layer, note, true);
}

void MotionHead::Gnn(const MotionLayer& layer, std::string_view note) {
  ProjStream(layer.attn[0], x[cur_x], &e, kMotionQueries, qb_,
             std::format("{} q", note), std::format("{} q", note));
  ProjStream(layer.attn[1], xsel, &esel, kSelectedPadded, kb_,
             std::format("{} k", note), std::format("{} k", note));
  ProjStream(layer.attn[2], xsel, nullptr, kSelectedPadded, vb_,
             std::format("{} v", note), std::format("{} v", note),
             &p_.fc_before);
  AttentionCore(layer, note, kSelectedPadded, kSelected, false, false);
  OutProj(layer, note, true);
}

void MotionHead::CrossGnn(const MotionLayer& layer, std::string_view note) {
  constexpr uint32_t kAq = 0;
  constexpr uint32_t kAk = kMotionQueries / 4 * kKbD;
  constexpr uint32_t kAv = kAk + (kMapSelectedPadded / 4 * kKbD);
  QuantRows(x[cur_x], kMotionQueries, kAq, std::format("{} Q(x+E)", note), &e);
  QuantRows(mxsel, kMapSelectedPadded, kAk, std::format("{} Q(mx+mE)", note),
            &mesel, kAv);
  ProjStore(layer.attn[0], {{.a_offset = kAq, .kb_offset = 0, .kb = kKbD}},
            kKbD, kMotionQueries, qb_, std::format("{} q", note),
            std::format("{} q", note));
  ProjStore(layer.attn[1], {{.a_offset = kAk, .kb_offset = 0, .kb = kKbD}},
            kKbD, kMapSelectedPadded, kb_, std::format("{} k", note),
            std::format("{} k", note));
  ProjStore(layer.attn[2], {{.a_offset = kAv, .kb_offset = 0, .kb = kKbD}},
            kKbD, kMapSelectedPadded, vb_, std::format("{} v", note),
            std::format("{} v", note));
  AttentionCore(layer, note, kMapSelectedPadded, kMapSelected, false, false);
  OutProj(layer, note, false);
}

void MotionHead::Norm(std::string_view note) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  for (const Tile& tile : Tiles(kMotionQueries, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    LoadRows(xr, tile.i0, tile.m, slot, std::format("{} ld t{}", note, tile.t));
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = slot,
                   .dst = 2048,
                   .rows_q = tile.m / 4,
                   .n8 = kNwD,
                   .p_base = keys_.at(std::string(note)),
                   .imm = LnEpsBits()},
                  std::format("{} LN t{}", note, tile.t));
    StoreRows(xo, tile.i0, tile.m, 2048,
              std::format("{} st t{}", note, tile.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void MotionHead::Ffn(const MotionLayer& layer, std::string_view note) {
  const Rows& xr = x[cur_x];
  const Rows& xo = x[(cur_x + 1) % 3];
  OnChip a(kAbufWords);
  const uint32_t aff = a.Take(kMotionQueries / 4 * kKbD);
  const uint32_t ah = a.Take(kMotionQueries / 4 * 16);
  for (const Tile& tile : Tiles(kMotionQueries, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    LoadRows(xr, tile.i0, tile.m, slot,
             std::format("{} ld x t{}", note, tile.t));
    program_->Vec({.op = VecOp::kLayerNorm,
                   .src = slot,
                   .dst = aff + ((tile.i0 / 4) * kKbD),
                   .rows_q = tile.m / 4,
                   .n8 = kNwD,
                   .p_base = keys_.at(std::format("{} ln", note)),
                   .imm = LnEpsBits(),
                   .q = true,
                   .scratch = 2048},
                  std::format("{} LN+Q t{}", note, tile.t));
  }
  layer.fc1.Load(*program_, 0, std::format("{} fc1", note));
  layer.fc2.Load(*program_, 1024, std::format("{} fc2", note));
  for (const Tile& tile : Tiles(kMotionQueries, 128)) {
    const uint32_t q0 = tile.i0 / 4;
    const uint32_t rq = tile.m / 4;
    layer.fc1.Gemm(*program_, aff + (q0 * kKbD), kKbD, tile.m, 0, 0,
                   std::format("{} fc1 t{}", note, tile.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = 0,
                   .dst = 0,
                   .rows_q = rq,
                   .n8 = 64,
                   .p_base = keys_.at(std::format("{} fc1", note))},
                  std::format("{} fc1 bias t{}", note, tile.t));
    program_->Vec({.op = VecOp::kRelu,
                   .src = 0,
                   .dst = ah + (q0 * 16),
                   .rows_q = rq,
                   .n8 = 64,
                   .q = true},
                  std::format("{} fc1 relu+Q t{}", note, tile.t));
  }
  layer.idf.Load(*program_, 0, std::format("{} idf", note));
  for (const Tile& tile : Tiles(kMotionQueries, 128)) {
    const uint32_t q0 = tile.i0 / 4;
    const uint32_t rq = tile.m / 4;
    const uint32_t c = tile.t % 2 == 0 ? 2048 : 3072;
    layer.fc2.Gemm(*program_, ah + (q0 * 16), 16, tile.m, c, 1024,
                   std::format("{} fc2 t{}", note, tile.t));
    layer.idf.Gemm(*program_, aff + (q0 * kKbD), kKbD, tile.m, c, 0,
                   std::format("{} identity_fc t{}", note, tile.t), {}, true);
    program_->Vec({.op = VecOp::kAddP,
                   .src = c,
                   .dst = c,
                   .rows_q = rq,
                   .n8 = kNwD,
                   .p_base = keys_.at(std::format("{} fc2", note))},
                  std::format("{} bias t{}", note, tile.t));
    StoreRows(xo, tile.i0, tile.m, c, std::format("{} st x t{}", note, tile.t));
  }
  cur_x = (cur_x + 1) % 3;
}

void MotionHead::Mlp(uint32_t a_in, uint32_t n, const MotionLayer& layer,
                     const std::string& branch, bool relu_ln,
                     const Rows& rows_out, uint32_t row0,
                     std::string_view note) {
  constexpr uint32_t kTile = 224;
  constexpr uint32_t kCt = 0;
  constexpr uint32_t kScratch = 1792;
  const uint32_t a2 = a_in + (n / 4 * kKbD);
  const uint32_t a3 = a2 + (n / 4 * kKbD);
  Require(a3 + (n / 4 * kKbD) <= kAbufWords, "motion MLP exceeds ABUF");
  const std::string bkey = std::format("{} {}", note, branch);
  const auto& lins = layer.branch.find(branch)->second;
  uint32_t src = a_in;
  for (size_t i = 0; i < lins.size(); ++i) {
    const uint32_t dst = src != a2 ? a2 : a3;
    const std::string name = std::format("{}{}", bkey, i);
    lins[i].Load(*program_, 0, name);
    for (const Tile& tile : Tiles(n, kTile)) {
      const uint32_t q0 = tile.i0 / 4;
      const uint32_t rq = tile.m / 4;
      lins[i].Gemm(*program_, src + (q0 * kKbD), kKbD, tile.m, kCt, 0,
                   std::format("{} t{}", name, tile.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = kCt,
                     .dst = kCt,
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = keys_.at(name)},
                    std::format("{} bias t{}", name, tile.t));
      if (relu_ln) {
        program_->Vec({.op = VecOp::kRelu,
                       .src = kCt,
                       .dst = kCt,
                       .rows_q = rq,
                       .n8 = kNwD},
                      std::format("{} relu t{}", name, tile.t));
        program_->Vec(
            {.op = VecOp::kLayerNorm,
             .src = kCt,
             .dst = dst + (q0 * kKbD),
             .rows_q = rq,
             .n8 = kNwD,
             .p_base = keys_.at(std::format("{} {}_ln{}", note, branch, i)),
             .imm = LnEpsBits(),
             .q = true,
             .scratch = kScratch},
            std::format("{} LN+Q t{}", name, tile.t));
      } else {
        program_->Vec({.op = VecOp::kRelu,
                       .src = kCt,
                       .dst = dst + (q0 * kKbD),
                       .rows_q = rq,
                       .n8 = kNwD,
                       .q = true},
                      std::format("{} relu+Q t{}", name, tile.t));
      }
    }
    src = dst;
  }
  const Lin& out = layer.branch_out.find(branch)->second;
  out.Load(*program_, 0, std::format("{} out", bkey));
  for (const Tile& tile : Tiles(n, kTile)) {
    const uint32_t q0 = tile.i0 / 4;
    const uint32_t rq = tile.m / 4;
    out.Gemm(*program_, src + (q0 * kKbD), kKbD, tile.m, kCt, 0,
             std::format("{} out t{}", bkey, tile.t));
    program_->Vec({.op = VecOp::kAddP,
                   .src = kCt,
                   .dst = kCt,
                   .rows_q = rq,
                   .n8 = out.Groups(),
                   .p_base = keys_.at(std::format("{}_out", bkey))},
                  std::format("{} out bias t{}", bkey, tile.t));
    StoreRows(rows_out, row0 + tile.i0, tile.m, kCt,
              std::format("{} out st t{}", bkey, tile.t));
  }
}

void MotionHead::Refine(const MotionLayer& layer, std::string_view note) {
  const Rows& xr = x[cur_x];
  for (const Tile& tile : Tiles(kMotionQueries, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    LoadRows(xr, tile.i0, tile.m, slot,
             std::format("{} ld x t{}", note, tile.t));
    LoadRows(e, tile.i0, tile.m, 2048,
             std::format("{} ld E t{}", note, tile.t));
    program_->Vec({.op = VecOp::kAdd,
                   .src = slot,
                   .src2 = 2048,
                   .dst = slot,
                   .rows_q = tile.m / 4,
                   .n8 = kNwD},
                  std::format("{} x+E t{}", note, tile.t));
    StoreRows(xe, tile.i0, tile.m, slot,
              std::format("{} st xE t{}", note, tile.t));
  }
  for (const Tile& chunk : Tiles(kModeQueriesPadded, kMotionQueries)) {
    for (const Tile& tile : Tiles(chunk.m, 128)) {
      const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
      LoadRows(sine, chunk.i0 + tile.i0, tile.m, slot,
               std::format("{} ld sine ch{} t{}", note, chunk.t, tile.t));
      program_->Vec({.op = VecOp::kQuant,
                     .src = slot,
                     .dst = 4096 + ((tile.i0 / 4) * kKbD),
                     .rows_q = tile.m / 4,
                     .n8 = kNwD},
                    std::format("{} Q(sine) t{}", note, tile.t));
    }
    p_.mae0.Load(*program_, 0, std::format("{} mae0", note));
    p_.mae1.Load(*program_, 1024, std::format("{} mae1", note));
    for (const Tile& tile : Tiles(chunk.m, 128)) {
      const uint32_t q0 = tile.i0 / 4;
      const uint32_t rq = tile.m / 4;
      p_.mae0.Gemm(*program_, 4096 + (q0 * kKbD), kKbD, tile.m, 0, 0,
                   std::format("{} mae0 ch{} t{}", note, chunk.t, tile.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = 0,
                     .dst = 0,
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = keys_.at("mae b0")},
                    std::format("{} mae0 bias t{}", note, tile.t));
      program_->Vec(
          {.op = VecOp::kRelu, .src = 0, .dst = 0, .rows_q = rq, .n8 = kNwD},
          std::format("{} mae0 relu t{}", note, tile.t));
      program_->Vec({.op = VecOp::kLayerNorm,
                     .src = 0,
                     .dst = 6144 + (q0 * kKbD),
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = keys_.at("mae ln"),
                     .imm = LnEpsBits(),
                     .q = true,
                     .scratch = 1024},
                    std::format("{} mae0 LN+Q t{}", note, tile.t));
      p_.mae1.Gemm(*program_, 6144 + (q0 * kKbD), kKbD, tile.m, 0, 1024,
                   std::format("{} mae1 ch{} t{}", note, chunk.t, tile.t));
      program_->Vec({.op = VecOp::kAddP,
                     .src = 0,
                     .dst = 0,
                     .rows_q = rq,
                     .n8 = kNwD,
                     .p_base = keys_.at("mae b1")},
                    std::format("{} mae1 bias t{}", note, tile.t));
      program_->Load(
          Tokens(rep6_ + (uint64_t{chunk.i0 + tile.i0} * 4), Mem::kFbuf, 2048,
                 tile.m, kKbD, true),
          std::format("{} ld xE rep6 ch{} t{}", note, chunk.t, tile.t));
      program_->Vec({.op = VecOp::kAdd,
                     .src = 0,
                     .src2 = 2048,
                     .dst = 0,
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} + xE t{}", note, tile.t));
      program_->Vec({.op = VecOp::kQuant,
                     .src = 0,
                     .dst = q0 * kKbD,
                     .rows_q = rq,
                     .n8 = kNwD},
                    std::format("{} Q(motion_query) t{}", note, tile.t));
    }
    Mlp(0, chunk.m, layer, "mcls", true, mcls, chunk.i0, std::string(note));
    Mlp(0, chunk.m, layer, "mreg", false, mreg, chunk.i0, std::string(note));
  }
  LoadRows(pmq, 0, kPlanQueriesPadded, 0,
           std::format("{} ld plan_modeq", note));
  program_->Load(
      Tokens(rep_ego_, Mem::kFbuf, 1024, kPlanQueriesPadded, kKbD, true),
      std::format("{} ld xE[900] x18", note));
  program_->Vec({.op = VecOp::kAdd,
                 .src = 0,
                 .src2 = 1024,
                 .dst = 0,
                 .rows_q = kPlanQueriesPadded / 4,
                 .n8 = kNwD},
                std::format("{} plan_query", note));
  program_->Vec({.op = VecOp::kQuant,
                 .src = 0,
                 .dst = 0,
                 .rows_q = kPlanQueriesPadded / 4,
                 .n8 = kNwD},
                std::format("{} Q(plan_query)", note));
  Mlp(0, kPlanQueriesPadded, layer, "pcls", true, pcls, 0, std::string(note));
  Mlp(0, kPlanQueriesPadded, layer, "preg", false, preg, 0, std::string(note));
  program_->Vec({.op = VecOp::kQuant,
                 .src = 1024,
                 .dst = 0,
                 .rows_q = kPlanQueriesPadded / 4,
                 .n8 = kNwD},
                std::format("{} Q(xE[900])", note));
  Mlp(0, 32, layer, "pst", false, pst, 0, std::string(note));
}

void MotionHead::AnchorEncoderOn(const Rows& a_rows, const Rows& e_out,
                                 uint32_t n, std::string_view note) {
  constexpr uint32_t kTile = 224;
  constexpr uint32_t kAt = 0;
  constexpr uint32_t kCt = 512;
  constexpr uint32_t kScratch = 2304;
  constexpr uint32_t kA1 = 0;
  constexpr uint32_t kA2 = kTile / 4;
  det_.pb_enc.Load(*program_, note);
  for (const Tile& tile : Tiles(n, kTile)) {
    const uint32_t rq = tile.m / 4;
    LoadRows(a_rows, tile.i0, tile.m, kAt,
             std::format("{} ld anchors t{}", note, tile.t));
    for (const auto& branch : det_.p.enc) {
      program_->Vec(
          {.op = VecOp::kMulP,
           .src = kAt,
           .dst = kA1,
           .rows_q = rq,
           .n8 = 4,
           .p_base = det_.keys_enc.at("mask." + branch.name),
           .q = true},
          std::format("{} {} select+Q t{}", note, branch.name, tile.t));
      for (size_t i = 0; i < branch.chain.size(); ++i) {
        const Lin& lin = branch.chain[i].first;
        const std::string where = std::format("{} {}{}", note, branch.name, i);
        lin.Load(*program_, 0, where);
        lin.Gemm(*program_, i == 0 ? kA1 : kA2, i == 0 ? 1 : lin.Kb(), tile.m,
                 kCt, 0, std::format("{} t{}", where, tile.t));
        program_->Vec({.op = VecOp::kAddP,
                       .src = kCt,
                       .dst = kCt,
                       .rows_q = rq,
                       .n8 = lin.Groups(),
                       .p_base = det_.keys_enc.at(
                           std::format("enc.{}.b{}", branch.name, i))},
                      std::format("{} bias t{}", where, tile.t));
        program_->Vec({.op = VecOp::kRelu,
                       .src = kCt,
                       .dst = kCt,
                       .rows_q = rq,
                       .n8 = lin.Groups()},
                      std::format("{} relu t{}", where, tile.t));
        const uint32_t ln =
            det_.keys_enc.at(std::format("enc.{}.ln{}", branch.name, i));
        if (i + 1 < branch.chain.size()) {
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = kCt,
                         .dst = kA2,
                         .rows_q = rq,
                         .n8 = lin.Groups(),
                         .p_base = ln,
                         .imm = LnEpsBits(),
                         .q = true,
                         .scratch = kScratch},
                        std::format("{} LN+Q t{}", where, tile.t));
        } else {
          program_->Vec({.op = VecOp::kLayerNorm,
                         .src = kCt,
                         .dst = kScratch,
                         .rows_q = rq,
                         .n8 = lin.Groups(),
                         .p_base = ln,
                         .imm = LnEpsBits()},
                        std::format("{} LN t{}", where, tile.t));
          program_->Store(
              {.ddr_offset = e_out.Addr(tile.i0) + (uint64_t{branch.col0} * 4),
               .mem = Mem::kFbuf,
               .onchip = kScratch,
               .rows = tile.m,
               .row_words = 1,
               .ddr_stride = static_cast<uint32_t>(e_out.Tb()),
               .mode = DmaMode::kRaw,
               .index = false,
               .row_format = true,
               .tok = lin.Rows() / 32},
              std::format("{} {} st E t{}", note, branch.name, tile.t));
        }
      }
    }
  }
}

void MotionHead::Queue() {
  constexpr uint32_t kQ = kMotionQueries / 4;
  constexpr uint32_t kNw = kMotionQueries / 32;
  program_->Load(Words(det_.pjprog, Mem::kCbuf, vprog::kRomWords,
                       static_cast<uint32_t>(det_.pj.Words().size())),
                 std::format("{} iq ld VEC program (anchor proj)", tag));
  program_->Load(Words(det_.proj_pb, Mem::kPbuf, 0, 18),
                 std::format("{} iq ld PBUF (T)", tag));
  program_->Load(Words(iq_zero_, Mem::kFbuf, 2048, 1),
                 std::format("{} iq ld dt = 0", tag));
  for (const Tile& tile : Tiles(kTemporalKeys, 1024)) {
    LoadRows(tanc, tile.i0, tile.m, 0,
             std::format("{} iq ld anchors t{}", tag, tile.t));
    program_->Run({.pc = vprog::kRomWords,
                   .src = 0,
                   .src2 = 2048,
                   .dst = 1024,
                   .rows_q = tile.m / 4,
                   .n8 = 4,
                   .p_base = 0},
                  std::format("{} iq anchor proj t{}", tag, tile.t));
    StoreRows(tanc_old, tile.i0, tile.m, 1024,
              std::format("{} iq st anchors t{}", tag, tile.t));
  }
  for (const Tile& tile : Tiles(kTemporalKeys, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    LoadRows(tx, tile.i0, tile.m, slot,
             std::format("{} iq ld temp_x t{}", tag, tile.t));
    StoreRows(tx_old, tile.i0, tile.m, slot,
              std::format("{} iq st temp_x t{}", tag, tile.t));
  }
  program_->Load(Words(iqprog_, Mem::kCbuf, vprog::kRomWords,
                       static_cast<uint32_t>(iq_.Words().size())),
                 std::format("{} ld VEC programs (instance queue)", tag));
  LoadRows(ego_anc0, 0, 4, 0, std::format("{} iq ld ego anchor", tag));
  LoadRows(pst, 0, 4, 4, std::format("{} iq ld plan_status", tag));
  program_->Run({.pc = iq_.Pc("ego"), .src2 = 4, .dst = 1},
                std::format("{} iq ego anchor VY", tag));
  StoreRows(anc, 900, 4, 0, std::format("{} iq st ego anchor", tag));
  constexpr uint32_t kG = 0;
  constexpr uint32_t kCls = 4 * kQ;
  constexpr uint32_t kI = 8 * kQ;
  constexpr uint32_t kMa = 12 * kQ;
  constexpr uint32_t kMb = 13 * kQ;
  constexpr uint32_t kList = 14 * kQ;
  constexpr uint32_t kConst = kList + (kQ / 2);
  program_->Load(Words(iq_t_, Mem::kPbuf, 0, 1),
                 std::format("{} iq ld PBUF (t)", tag));
  program_->Load(Tokens(stlist, Mem::kFbuf, kG, kMotionQueries, 1, true),
                 std::format("{} iq gather state", tag));
  det_.LoadRows(det_.cls, 0, kMotionQueries, kCls,
                std::format("{} iq ld cls", tag));
  LoadRows(st, 0, kMotionQueries, kI, std::format("{} iq ld own state", tag));
  program_->Load(Words(iq_cs_, Mem::kFbuf, kConst, 1),
                 std::format("{} iq ld const", tag));
  program_->Run({.pc = iq_.Pc("state"),
                 .src = kG,
                 .src2 = kCls,
                 .dst = kMa,
                 .rows_q = kQ,
                 .n8 = det_.ncls,
                 .p_base = 0,
                 .imm = kConst},
                std::format("{} iq state + mask", tag));
  StoreRows(st, 0, kMotionQueries, kG, std::format("{} iq st state", tag));
  for (const auto& [at, src] : {std::pair{0U, kMa}, {1U, kMb}}) {
    program_->Store({.ddr_offset = mask + (uint64_t{at} * kWordF),
                     .mem = Mem::kFbuf,
                     .onchip = src,
                     .rows = kQ,
                     .row_words = 1,
                     .ddr_stride = 18 * kWordF},
                    std::format("{} iq st mask {}", tag, at == 0 ? "A" : "B"));
  }
  for (const auto& [word, list, name] :
       {std::tuple{iq_cx_, tlist, "temp_x"},
        std::tuple{iq_ca_, talist, "temp_anchor"}}) {
    program_->Load(Words(word, Mem::kFbuf, kConst, 1),
                   std::format("{} iq ld const ({})", tag, name));
    program_->Run({.pc = iq_.Pc("list"),
                   .src = kG,
                   .src2 = kCls,
                   .dst = kList,
                   .rows_q = kQ,
                   .imm = kConst},
                  std::format("{} iq list {}", tag, name));
    program_->Store(Words(list, Mem::kFbuf, kList, kQ / 2),
                    std::format("{} iq st list {}", tag, name));
  }
  program_->Load(Words(det_.selidx, Mem::kFbuf, 0, kNw),
                 std::format("{} iq ld selidx", tag));
  program_->Load(Words(det_.ib_idx, Mem::kFbuf, kNw, kNw),
                 std::format("{} iq ld idx", tag));
  program_->Load(Words(iq_cl_, Mem::kFbuf, kConst, 1),
                 std::format("{} iq ld const (stlist)", tag));
  program_->Run({.pc = iq_.Pc("stlist"),
                 .src = 0,
                 .src2 = kNw,
                 .dst = 2 * kNw,
                 .rows_q = kNw,
                 .imm = kConst},
                std::format("{} iq stlist", tag));
  program_->Store(Words(stlist, Mem::kFbuf, 2 * kNw, kNw),
                  std::format("{} iq st stlist", tag));
  for (const Tile& tile : Tiles(kTemporalKeys, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    program_->Load(Tokens(tlist + (uint64_t{tile.i0} * 4), Mem::kFbuf, slot,
                          tile.m, kKbD, true),
                   std::format("{} gather temp_x t{}", tag, tile.t));
    StoreRows(tx, tile.i0, tile.m, slot,
              std::format("{} st temp_x t{}", tag, tile.t));
  }
  for (const Tile& tile : Tiles(kTemporalKeys, 1024)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    program_->Load(Tokens(talist + (uint64_t{tile.i0} * 4), Mem::kFbuf, slot,
                          tile.m, 1, true),
                   std::format("{} gather temp_anchor t{}", tag, tile.t));
    StoreRows(tanc, tile.i0, tile.m, slot,
              std::format("{} st temp_anchor t{}", tag, tile.t));
  }
}

void MotionHead::MotionAnchor() {
  const uint32_t nap = det_.C().nap;
  const uint32_t q = nap / 4;
  const uint32_t nw = nap / 32;
  program_->Load(Words(maprog_, Mem::kCbuf, vprog::kRomWords,
                       static_cast<uint32_t>(ma_.Words().size())),
                 std::format("{} ld VEC programs (motion anchor)", tag));
  const uint32_t cls_at = 0;
  const uint32_t work = 4 * q;
  const uint32_t list = (8 * q) + 2;
  const uint32_t cst = list + nw;
  det_.LoadRows(det_.cls, 0, nap, cls_at, std::format("{} ma ld cls", tag));
  program_->Load(Words(ma_const_, Mem::kFbuf, cst, 1),
                 std::format("{} ma ld const", tag));
  program_->Run({.pc = ma_.Pc("clsidx"),
                 .src = cls_at,
                 .src2 = work,
                 .dst = list,
                 .rows_q = q,
                 .n8 = det_.ncls,
                 .imm = cst},
                std::format("{} ma clsidx", tag));
  program_->Store(Words(ma_list_, Mem::kFbuf, list, nw),
                  std::format("{} ma st list", tag));
  const uint32_t anc_at = 0;
  const uint32_t tab = 4 * q;
  const uint32_t rx = 8 * q;
  const uint32_t bc = 0;
  det_.LoadRows(det_.anc[det_.cur_a], 0, nap, anc_at,
                std::format("{} ma ld anchor", tag));
  program_->Load(Tokens(ma_list_, Mem::kFbuf, tab, nap, 1, true),
                 std::format("{} ma gather table rows", tag));
  program_->Run(
      {.pc = ma_.Pc("rot"), .src = anc_at, .src2 = tab, .dst = rx, .rows_q = q},
      std::format("{} ma rot", tag));
  for (uint32_t m = 0; m < kModes; ++m) {
    program_->Run(
        {.pc = ma_.Pc("bc"), .src = rx, .dst = bc, .rows_q = q, .n8 = m},
        std::format("{} ma bcast m{}", tag, m));
    StoreRows(rxm_[m], 0, nap, bc, std::format("{} ma st rx{}", tag, m));
    StoreRows(rym_[m], 0, nap, bc + (4 * q),
              std::format("{} ma st ry{}", tag, m));
  }
  program_->Load(
      {.ddr_offset = ma_tb_,
       .mem = Mem::kTbuf,
       .onchip = 0,
       .rows = 1,
       .row_words = kTbufWords,
       .ddr_stride = static_cast<uint32_t>(uint64_t{kTbufWords} * kWordT)},
      std::format("{} ma ld TBUF (sin)", tag));
  program_->Load(Words(ma_pb_, Mem::kPbuf, 0, 32),
                 std::format("{} ma ld PBUF", tag));
  constexpr uint32_t kTq = 64;
  constexpr uint32_t kPy = 0;
  constexpr uint32_t kPx = 4 * kTq;
  constexpr uint32_t kPly = 8 * kTq;
  constexpr uint32_t kPlx = 9 * kTq;
  constexpr uint32_t kOut = 10 * kTq;
  for (const Tile& tile : Tiles(kModeQueriesPadded, 4 * kTq)) {
    const uint32_t count = tile.m / 4;
    program_->Load(Tokens(ma_glist_ + (uint64_t{tile.i0} * 4), Mem::kFbuf, kPx,
                          tile.m, 1, true),
                   std::format("{} ma gather posx t{}", tag, tile.t));
    program_->Load(Tokens(ma_glist_ + (uint64_t{kModeQueriesPadded} * 4) +
                              (uint64_t{tile.i0} * 4),
                          Mem::kFbuf, kPy, tile.m, 1, true),
                   std::format("{} ma gather posy t{}", tag, tile.t));
    program_->Run(
        {.pc = ma_.Pc("compact"), .src = kPy, .dst = kPly, .rows_q = count},
        std::format("{} ma compact y t{}", tag, tile.t));
    program_->Run(
        {.pc = ma_.Pc("compact"), .src = kPx, .dst = kPlx, .rows_q = count},
        std::format("{} ma compact x t{}", tag, tile.t));
    program_->Run({.pc = ma_.Pc("embed"),
                   .src = kPly,
                   .src2 = kPlx,
                   .dst = kOut,
                   .rows_q = count,
                   .p_base = 0},
                  std::format("{} ma embed t{}", tag, tile.t));
    StoreRows(sine, tile.i0, tile.m, kOut,
              std::format("{} ma st sine t{}", tag, tile.t));
  }
  program_->Load(
      {.ddr_offset = det_.tb,
       .mem = Mem::kTbuf,
       .onchip = 0,
       .rows = 1,
       .row_words = kTbufWords,
       .ddr_stride = static_cast<uint32_t>(uint64_t{kTbufWords} * kWordT)},
      std::format("{} ma ld TBUF (silu)", tag));
}

void MotionHead::PrepNpu() {
  program_->Load(Words(mpprog_, Mem::kCbuf, vprog::kRomWords,
                       static_cast<uint32_t>(mp_.Words().size())),
                 std::format("{} ld VEC programs (prep)", tag));
  program_->Load(Tokens(ego_o_.Addr(0, 0), Mem::kFbuf, 0, 44, kKbD),
                 std::format("{} ego ld", tag));
  program_->Run(
      {.pc = mp_.Pc("ego"),
       .src = 0,
       .dst = 352,
       .imm = exact::Real::Binary64(exact::Real::Ratio(1, 44).ToBinary64())
                  .ToBinary32()},
      std::format("{} ego mean", tag));
  StoreRows(x[0], 900, 4, 352, std::format("{} ego st rows 900..903", tag));
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
    LoadRows(h.cls, 0, nap, 0, std::format("{} {} ld cls", tag, note));
    program_->Load(Words(h.ib_idx, Mem::kFbuf, work + nw, nw),
                   std::format("{} {} ld idx", tag, note));
    program_->Load(Words(h.ib_lane, Mem::kFbuf, work + (6 * nw) + 2, 1),
                   std::format("{} {} ld lane", tag, note));
    program_->Load(Words(list, Mem::kFbuf, out, nlw),
                   std::format("{} {} ld list x", tag, note));
    program_->Load(Words(list + (uint64_t{nl} * 4), Mem::kFbuf, out + nw, nlw),
                   std::format("{} {} ld list E", tag, note));
    program_->Load(Words(mp_const_.at(key), Mem::kFbuf, cst, 1),
                   std::format("{} {} ld const", tag, note));
    program_->Run({.pc = mp_.Pc(name),
                   .src = 0,
                   .src2 = work,
                   .dst = out,
                   .rows_q = nap / 4,
                   .n8 = h.ncls,
                   .p_base = 0,
                   .imm = cst},
                  std::format("{} {} run", tag, note));
    program_->Store(Words(list, Mem::kFbuf, out, nlw),
                    std::format("{} {} st list x", tag, note));
    program_->Store(Words(list + (uint64_t{nl} * 4), Mem::kFbuf, out + nw, nlw),
                    std::format("{} {} st list E", tag, note));
  }
  for (const Tile& tile : Tiles(900, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    const uint32_t m4 = (tile.m + 3) / 4 * 4;
    det_.LoadRows(det_.anc[det_.cur_a], tile.i0, m4, slot,
                  std::format("{} ld det anchor t{}", tag, tile.t));
    StoreRows(anc, tile.i0, m4, slot,
              std::format("{} st anchor t{}", tag, tile.t));
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
  for (const Tile& tile : Tiles(900, 128)) {
    const uint32_t slot = tile.t % 2 == 0 ? 0 : 1024;
    const uint32_t m4 = (tile.m + 3) / 4 * 4;
    LoadRows(xd, tile.i0, m4, slot,
             std::format("{} ld det x t{}", tag, tile.t));
    StoreRows(x[0], tile.i0, m4, slot, std::format("{} st x t{}", tag, tile.t));
  }
  cur_x = 0;
  Queue();
  AnchorEncoderOn(anc, e, kMotionQueries, std::format("{} E", tag));
  AnchorEncoderOn(tanc, te, kTemporalKeys, std::format("{} temp_E", tag));
  const std::array<std::tuple<uint64_t, const Rows*, uint32_t>, 4> gathers = {
      {{sel, &xsel, kSelectedPadded},
       {sel + (uint64_t{kSelectedPadded} * 4), &esel, kSelectedPadded},
       {msel, &mxsel, kMapSelectedPadded},
       {msel + (uint64_t{kMapSelectedPadded} * 4), &mesel,
        kMapSelectedPadded}}};
  for (size_t i = 0; i < gathers.size(); ++i) {
    const auto& [list, rows, n] = gathers[i];
    program_->Load(Tokens(list, Mem::kFbuf, 0, n, kKbD, true),
                   std::format("{} gather sel {}", tag, i));
    StoreRows(*rows, 0, n, 0, std::format("{} st sel {}", tag, i));
  }
}

void MotionHead::EmitFrame(Program& program) {
  program_ = &program;
  Prep();
  for (size_t l = 0; l < p_.layers.size(); ++l) {
    const MotionLayer& layer = p_.layers[l];
    const std::string note = std::format("{} L{} {}", tag, l, Name(layer.op));
    pb_layers_[l].first.Load(program, note);
    keys_ = pb_layers_[l].second;
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
  const Rows& xr = x[cur_x];
  const std::string note = std::format("{} cache ego feature", tag);
  program.Load(Words(xr.Addr(900), Mem::kFbuf, 0, kD / 32),
               std::format("{} ld", note));
  program.Store(
      Words(tx.Addr((900 * kQueue) + kQueue - 1), Mem::kFbuf, 0, kD / 32),
      std::format("{} st", note));
  program.Signal(1, 1, std::format("{} SIGNAL 1 motion done", tag));
  program.WaitHost(1, std::format("{} WAIT_HOST 1", tag));
}

}  // namespace hotarubi::compiler::sparsedrive
