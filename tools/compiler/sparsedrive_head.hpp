#ifndef HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_HEAD_HPP_
#define HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_HEAD_HPP_

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "program.hpp"
#include "sparsedrive_kernels.hpp"
#include "sparsedrive_layers.hpp"

namespace hotarubi::compiler::sparsedrive {

inline constexpr uint32_t kLevels = 4;

struct HeadConfig {
  bool det = true;
  uint32_t na = 0;
  uint32_t nap = 0;
  uint32_t ntemp = 0;
  uint32_t ntp = 0;
  uint32_t acols = 0;
  uint32_t adim = 0;
  uint32_t dq = 0;
  uint32_t hd = 0;
  uint32_t npt = 0;
  uint32_t scam = 0;
};

enum class Op : uint8_t {
  kGnn,
  kTempGnn,
  kCrossGnn,
  kNorm,
  kDeformable,
  kFfn,
  kRefine,
};

struct EncoderBranch {
  std::string name;
  uint32_t lo = 0;
  uint32_t hi = 0;
  uint32_t col0 = 0;
  std::vector<std::pair<Lin, LayerNorm>> chain;
};

struct HeadLayer {
  Op op = Op::kNorm;
  std::vector<Lin> attn;
  LayerNorm ln;
  Lin lfc;
  std::vector<std::pair<Lin, LayerNorm>> cam;
  std::vector<Lin> wfc;
  Lin oproj;
  std::vector<uint32_t> fix_scale;
  Lin fc1;
  Lin fc2;
  Lin idf;
  std::vector<Lin> reg;
  std::vector<LayerNorm> reg_ln;
  Lin reg_out;
  std::vector<uint32_t> scale;
  std::vector<Lin> cls;
  std::vector<LayerNorm> cls_ln;
  Lin cls_out;
  std::vector<Lin> qt;
  std::vector<LayerNorm> qt_ln;
  Lin qt_out;
};

struct HeadParams {
  HeadConfig config;
  std::vector<uint32_t> anchor0;
  std::vector<uint32_t> x0;
  std::vector<EncoderBranch> enc;
  std::optional<Lin> fc_before;
  std::optional<Lin> fc_after;
  std::vector<HeadLayer> layers;
};

HeadParams ReadHeadParams(const HeadWeights& weights, bool det);

class SparseHead {
 public:
  SparseHead(DdrImage& image, uint64_t base, HeadParams params,
             const std::vector<std::vector<Level>>& levels, uint64_t w_off);

  void EmitFrame(Program& program, bool temp);
  void LoadRows(const Rows& rows, uint32_t j, uint32_t m, uint32_t dst,
                std::string_view note);
  void StoreRows(const Rows& rows, uint32_t j, uint32_t m, uint32_t src,
                 std::string_view note);

  [[nodiscard]] const HeadConfig& C() const { return p.config; }

  HeadParams p;
  std::string tag;
  uint32_t ncam = 0;
  uint32_t kbq = 0;
  uint32_t nwq = 0;
  uint32_t s = 0;
  std::array<Rows, 3> x;
  Rows e;
  Rows tx;
  std::array<Rows, 2> anc;
  Rows tanc;
  Rows x_init;
  Rows a_init;
  Rows lfc;
  Rows cls;
  std::optional<Rows> qt;
  Rows cam_in;
  Rows cam_vec;
  uint64_t qb = 0;
  uint64_t kb = 0;
  uint64_t vb = 0;
  uint64_t ob = 0;
  uint64_t cat = 0;
  uint32_t nq = 0;
  uint32_t wq = 0;
  uint32_t wg = 0;
  uint64_t w = 0;
  uint64_t lbc = 0;
  uint64_t lists_bytes = 0;
  uint64_t lists = 0;
  std::array<uint64_t, 2> order{};
  std::array<uint64_t, 2> order_use{};
  uint64_t keep = 0;
  uint64_t tb = 0;
  uint32_t cur_x = 0;
  uint32_t cur_a = 0;
  uint32_t ncls = 0;
  CbufImage kp;
  uint64_t vprog = 0;
  uint64_t kp_const = 0;
  uint64_t kp_lv = 0;
  std::map<size_t, uint64_t> kp_fs;
  uint64_t kp_zh = 0;
  uint32_t nw = 0;
  CbufImage ib;
  uint64_t ibprog = 0;
  uint64_t ib_idx = 0;
  uint64_t ib_lane = 0;
  uint64_t prevconf = 0;
  uint64_t selidx = 0;
  uint64_t keep_a = 0;
  Rows tanc_raw;
  uint64_t proj_pb = 0;
  uint64_t proj_dt = 0;
  uint64_t fs_flag = 0;
  CbufImage pj;
  uint64_t pjprog = 0;
  uint64_t null_list = 0;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> scale_patch;
  Pbuf pb_enc;
  Keys keys_enc;
  std::vector<std::tuple<Pbuf, Keys, std::string>> pb_layers;

 private:
  struct Quantize {
    VecArgs args;
    std::string note;
  };

  void PlaceParams();
  void QuantRows(const Rows& rows, uint32_t n, uint32_t abuf,
                 std::string_view note,
                 const std::function<void(uint32_t, uint32_t, uint32_t,
                                          uint32_t)>& extra = {});
  void QuantRowsAdd(const Rows& rows, const Rows& rows2, uint32_t n,
                    uint32_t abuf, std::string_view note,
                    std::optional<uint32_t> abuf_plain = std::nullopt,
                    std::optional<uint32_t> pb_vec = std::nullopt);
  void StoreColumns(const Rows& rows, uint32_t j, uint32_t m, uint32_t src,
                    uint32_t col0, uint32_t ncols, std::string_view note);
  void StoreColumnsStrided(const Rows& rows, uint32_t j, uint32_t m,
                           uint32_t src, uint32_t n8_src, uint32_t col0,
                           uint32_t ncols, std::string_view note);
  void AnchorEncoder(std::string_view note);
  void Attention(const HeadLayer& layer, const Keys& pb, std::string_view note);
  void OutProjection(const HeadLayer& layer, const Keys& pb,
                     std::string_view note);
  void Norm(const Keys& pb, std::string_view note);
  void CameraEncoder(const HeadLayer& layer, const Keys& pb,
                     std::string_view note);
  void Deformable(const HeadLayer& layer, size_t index, const Keys& pb,
                  std::string_view note);
  void DeformableWeights(const HeadLayer& layer, const Keys& pb,
                         std::string_view note);
  void WeightSoftmax(std::string_view note);
  void Gather(uint32_t q, uint32_t ag, uint32_t ring, uint32_t ring_quads,
              std::string_view note, std::optional<Quantize>& pending);
  void Keypoints(size_t index, const Keys& pb, std::string_view note,
                 const std::function<void(uint32_t)>& gather);
  void KeypointsMap(std::string_view note);
  void Ffn(const HeadLayer& layer, const Keys& pb, std::string_view note);
  void Refine(const HeadLayer& layer, const Keys& pb, std::string_view note);
  void Reorder(std::string_view note);
  void Keep(std::string_view note);
  void Init(std::string_view note);
  void KeepAnchor(std::string_view note);
  void AnchorProject(std::string_view note);
  void EnsureCbuf(int which);
  void Bank(bool update, std::string_view note);
  [[nodiscard]] uint64_t ListAddr(uint32_t q, uint32_t cam) const {
    return lists + (((uint64_t{q} * ncam) + cam) * lbc);
  }

  DdrImage& image_;
  uint64_t base_;
  Program* program_ = nullptr;
  bool temp_ = false;
  int cbuf_has_ = 0;
  std::map<std::tuple<bool, uint32_t, uint32_t>, uint64_t> ib_const_;
  std::map<std::pair<uint32_t, uint32_t>, uint64_t> order_id_;
};

}  // namespace hotarubi::compiler::sparsedrive

#endif  // HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_HEAD_HPP_
