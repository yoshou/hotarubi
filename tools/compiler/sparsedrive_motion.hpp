#ifndef HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_MOTION_HPP_
#define HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_MOTION_HPP_

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "program.hpp"
#include "sparsedrive_head.hpp"
#include "sparsedrive_kernels.hpp"
#include "sparsedrive_layers.hpp"
#include "sparsedrive_parts.hpp"

namespace hotarubi::compiler::sparsedrive {

inline constexpr uint32_t kMotionQueries = 928;
inline constexpr uint32_t kQueue = 4;
inline constexpr uint32_t kTemporalKeys = kQueue * kMotionQueries;
inline constexpr uint32_t kSelected = 51;
inline constexpr uint32_t kSelectedPadded = 64;
inline constexpr uint32_t kMapSelected = 10;
inline constexpr uint32_t kMapSelectedPadded = 32;
inline constexpr uint32_t kModes = 6;
inline constexpr uint32_t kModeQueries = 900 * kModes;
inline constexpr uint32_t kModeQueriesPadded = 5408;
inline constexpr uint32_t kPlanQueries = 18;
inline constexpr uint32_t kPlanQueriesPadded = 32;

struct MotionLayer {
  Op op = Op::kNorm;
  uint32_t embed = 0;
  uint32_t hd = 0;
  std::vector<Lin> attn;
  LayerNorm ln;
  Lin fc1;
  Lin fc2;
  Lin idf;
  std::map<std::string, std::vector<Lin>, std::less<>> branch;
  std::map<std::string, std::vector<LayerNorm>, std::less<>> branch_ln;
  std::map<std::string, Lin, std::less<>> branch_out;
};

struct MotionParams {
  Lin fc_before;
  Lin fc_after;
  std::vector<MotionLayer> layers;
  Lin mae0;
  LayerNorm mae_ln;
  Lin mae1;
  std::vector<uint32_t> motion_anchor;
  std::vector<uint32_t> plan_mode_query;
  std::vector<uint32_t> ego_anchor;
  std::array<Conv, 2> ego_conv;
};

MotionParams ReadMotionParams(const HeadWeights& weights,
                              const Sources& sources);

class MotionHead {
 public:
  MotionHead(DdrImage& image, uint64_t base, MotionParams params,
             SparseHead& det, SparseHead& map, const FMap& f3);
  void EmitFrame(Program& program);

  std::string tag = "motion";
  std::array<Rows, 3> x;
  Rows e;
  Rows xe;
  Rows anc;
  Rows tx;
  Rows tanc;
  uint64_t tlist = 0;
  Rows te;
  Rows xsel;
  Rows esel;
  Rows mxsel;
  Rows mesel;
  uint64_t sel = 0;
  uint64_t msel = 0;
  uint32_t mask_cs = 144;
  uint64_t mask = 0;
  Rows tx_old;
  Rows tanc_old;
  uint64_t talist = 0;
  Rows st;
  uint64_t stlist = 0;
  Rows ego_anc0;
  Rows sine;
  Rows modeq;
  Rows pmq;
  Rows mcls;
  Rows mreg;
  Rows pcls;
  Rows preg;
  Rows pst;
  uint32_t cur_x = 0;

 private:
  void PlaceParams();
  [[nodiscard]] Place Pb(std::string_view key) const;
  void LoadRows(const Rows& rows, uint32_t j, uint32_t m, Place dst,
                std::string_view note);
  [[nodiscard]] Tile LoadRows(const Rows& rows, uint32_t j, uint32_t m,
                              std::string_view note);
  void StoreRows(const Rows& rows, uint32_t j, uint32_t m, Place src,
                 std::string_view note);
  void QuantRows(const Rows& rows, uint32_t n, Place abuf,
                 std::string_view note, const Rows* second = nullptr,
                 std::optional<Place> abuf_plain = std::nullopt);
  void ProjStore(const Lin& lin, const std::vector<Piece>& pieces,
                 uint32_t astride, uint32_t n, uint64_t out,
                 const std::string& key, std::string_view note);
  void ProjStream(const Lin& lin, const Rows& rows_a, const Rows* rows_b,
                  uint32_t n, uint64_t out, const std::string& key,
                  std::string_view note);
  void ProjThrough(const Lin& lin, const Lin& pre, const Rows& rows, uint32_t n,
                   uint64_t out, const std::string& key, std::string_view note);
  void AttentionCore(const MotionLayer& layer, std::string_view note,
                     uint32_t keys, uint32_t valid, bool chunk_local,
                     bool masked);
  void OutProj(const MotionLayer& layer, std::string_view note, bool decoupled);
  void TempGnn(const MotionLayer& layer, std::string_view note);
  void Gnn(const MotionLayer& layer, std::string_view note);
  void CrossGnn(const MotionLayer& layer, std::string_view note);
  void Norm(std::string_view note);
  void Ffn(const MotionLayer& layer, std::string_view note);
  void Refine(const MotionLayer& layer, std::string_view note);
  void Mlp(const Tile& a_in, uint32_t n, const MotionLayer& layer,
           const std::string& branch, bool relu_ln, const Rows& rows_out,
           uint32_t row0, std::string_view note);
  void Prep();
  void PrepNpu();
  void MotionAnchor();
  void Queue();

  DdrImage& image_;
  uint64_t base_;
  MotionParams p_;
  SparseHead& det_;
  SparseHead& map_;
  const FMap& f3_;
  Program* program_ = nullptr;
  CbufImage iq_;
  uint64_t iqprog_ = 0;
  uint64_t iq_cs_ = 0;
  uint64_t iq_cx_ = 0;
  uint64_t iq_ca_ = 0;
  uint64_t iq_cl_ = 0;
  uint64_t iq_t_ = 0;
  uint64_t iq_zero_ = 0;
  CbufImage mp_;
  uint64_t mpprog_ = 0;
  std::map<std::pair<int, uint32_t>, uint64_t> mp_const_;
  CbufImage ma_;
  uint64_t maprog_ = 0;
  Rows matab_;
  uint64_t ma_list_ = 0;
  uint64_t ma_const_ = 0;
  std::array<Rows, kModes> rxm_;
  std::array<Rows, kModes> rym_;
  uint64_t ma_glist_ = 0;
  uint64_t ma_tb_ = 0;
  uint64_t ma_pb_ = 0;
  uint64_t rep6_ = 0;
  uint64_t rep_ego_ = 0;
  uint64_t qb_ = 0;
  uint64_t kb_ = 0;
  uint64_t vb_ = 0;
  uint64_t ob_ = 0;
  FMap ego_t_;
  FMap ego_o_;
  std::vector<std::pair<Pbuf, Keys>> pb_layers_;
  Tile pb_;
  const Keys* keys_ = nullptr;
};

}  // namespace hotarubi::compiler::sparsedrive

#endif  // HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_MOTION_HPP_
