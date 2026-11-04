#ifndef HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_KERNELS_HPP_
#define HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_KERNELS_HPP_

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vprog.hpp"

namespace hotarubi::compiler::sparsedrive {

using Word = std::array<uint32_t, 32>;

inline constexpr uint32_t kConstOne = 0;
inline constexpr uint32_t kConstNone = 1;
inline constexpr uint32_t kConstNhalf = 2;
inline constexpr uint32_t kConstTwo = 3;
inline constexpr uint32_t kConstMagic = 4;
inline constexpr uint32_t kConstNtwo = 5;
inline constexpr uint32_t kConstEps = 6;
inline constexpr uint32_t kConstSign = 7;
inline constexpr uint32_t kConstI2f0 = 8;
inline constexpr uint32_t kConstCf = 9;
inline constexpr uint32_t kConstNcf = 10;
inline constexpr uint32_t kConstHalf = 11;
inline constexpr uint32_t kConstRwx = 12;
inline constexpr uint32_t kConstRwy = 13;
inline constexpr uint32_t kConstZero = 14;
inline constexpr uint32_t kConstProjection = 16;

inline constexpr uint32_t kMapSamples = 20;
inline constexpr uint32_t kMapHeights = 5;
inline constexpr uint32_t kMapLearn = 3;
inline constexpr uint32_t kMapGroupsPerBlock = (kMapSamples + 7) / 8;
inline constexpr uint32_t kMapBlock = 64;
inline constexpr uint32_t kMapLfcCols = kMapHeights * kMapLearn * kMapBlock;
inline constexpr uint32_t kMapGroups =
    kMapHeights * kMapLearn * kMapGroupsPerBlock;

inline constexpr uint32_t kFsTs = 0;
inline constexpr uint32_t kFsTg = 1;
inline constexpr uint32_t kFsPts = 2;
inline constexpr uint32_t kFsPtg = 3;
inline constexpr uint32_t kFsFlag = 4;
inline constexpr uint32_t kFsDt = 5;
inline constexpr uint32_t kFsCol8 = 6;
inline constexpr uint32_t kFsKp = 8;
inline constexpr uint32_t kFsSc = 24;
inline constexpr uint32_t kFsDi = 32;
inline constexpr uint32_t kFsMi = 56;
inline constexpr uint32_t kFsW = 72;
inline constexpr uint32_t kFsNtwoPbuf = 4;

struct Level {
  uint64_t phys = 0;
  uint32_t h = 0;
  uint32_t w = 0;
  uint64_t tb = 0;
};

uint32_t FloatBits(int64_t value);
Word ConstWord();
Word FixScaleWord(const std::vector<uint32_t>& fix_scale);
Word LevelWord(const std::vector<Level>& levels);
Word ZhWord();
std::vector<Word> LaneConstantWords();
Word IbankConstWord(uint32_t base_x16, uint32_t tb_x16, uint32_t base_a16,
                    uint32_t tb_a16, uint32_t na, uint32_t ntemp);
std::vector<Word> LaneIndexWords(uint32_t n);
Word LaneIdWord();
Word FsCol8Word();
std::vector<Word> FsDetTemplate();
Word IqueueConstWord(uint32_t na, uint32_t tq);
Word IqueueListWord(uint32_t base_old16, uint32_t tb16, uint32_t base_cur16,
                    uint32_t tb_cur16, uint32_t na);
Word IqueueStlistWord(uint32_t base16, uint32_t tb16, uint32_t nt, uint32_t na);
Word IqueueTWord();
std::vector<uint64_t> SinTbuf();
std::vector<uint32_t> EmbedPbuf();
std::vector<int64_t> LfcPerm(uint32_t learn);
std::vector<int64_t> MapLfcPerm();
std::array<uint32_t, 3> MapGroup(uint32_t gi);

vprog::Assembler DetKeypointInit(uint32_t tq);
vprog::Assembler DetKeypointPoint(uint32_t tq, uint32_t learn);
vprog::Assembler DetKeypointLevel(uint32_t tq);
uint32_t DetKeypointWork(uint32_t tq);
vprog::Assembler MapKeypointInit(uint32_t tq);
vprog::Assembler MapKeypointPoint(uint32_t tq);
vprog::Assembler MapKeypointLevel(uint32_t tq);
uint32_t MapKeypointWork(uint32_t tq);

enum class BankMode : uint8_t {
  kUpdate,
  kCache,
  kSelect,
};

vprog::Assembler InstanceBankKernel(uint32_t nap, uint32_t na, uint32_t ntemp,
                                    BankMode mode, uint32_t k);
vprog::Assembler LongSoftmaxKernel();
vprog::Assembler EgoMeanKernel();
vprog::Assembler MotionClsIdx(uint32_t nap, uint32_t ncls);
vprog::Assembler MotionRot();
vprog::Assembler MotionBcast(uint32_t modes);
vprog::Assembler MotionCompact();
vprog::Assembler MotionEmbed();
vprog::Assembler AnchorProjectKernel(bool box, uint32_t words);
vprog::Assembler FrameStartKernel(uint32_t nkp, uint32_t ns);
vprog::Assembler MaskKernel(bool fill);
vprog::Assembler IqueueState(uint32_t ncls);
vprog::Assembler IqueueList();
vprog::Assembler IqueueStlist(uint32_t na);
vprog::Assembler IqueueEgo();

class CbufImage {
 public:
  uint32_t Add(const std::string& name, const vprog::Assembler& program);
  [[nodiscard]] uint32_t Pc(const std::string& name) const {
    return pcs_.at(name);
  }
  [[nodiscard]] const std::vector<Instruction>& Words() const { return words_; }

 private:
  std::vector<Instruction> words_;
  std::map<std::string, uint32_t> pcs_;
};

}  // namespace hotarubi::compiler::sparsedrive

#endif  // HOTARUBI_TOOLS_COMPILER_SPARSEDRIVE_KERNELS_HPP_
