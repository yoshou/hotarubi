#ifndef HOTARUBI_TOOLS_RUNTIME_SMOLVLA_PS_HPP_
#define HOTARUBI_TOOLS_RUNTIME_SMOLVLA_PS_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/formats.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"

namespace hotarubi::smolvla {

inline constexpr uint32_t kCameras = 2;
inline constexpr uint32_t kImageSize = 512;
inline constexpr uint32_t kChannels = 3;
inline constexpr uint32_t kPatch = 16;
inline constexpr uint32_t kGrid = kImageSize / kPatch;
inline constexpr uint32_t kPatchTokens = kGrid * kGrid;
inline constexpr uint32_t kPatchValues = kChannels * kPatch * kPatch;
inline constexpr uint32_t kTokensPerCamera = 64;
inline constexpr uint32_t kLanguageWidth = 960;
inline constexpr uint32_t kStateRows = 32;
inline constexpr uint32_t kStateWidth = 32;
inline constexpr uint32_t kChunk = 50;
inline constexpr uint32_t kActionWidth = 32;
inline constexpr uint32_t kActionDims = 6;

struct Image {
  uint32_t height = 0;
  uint32_t width = 0;
  std::vector<uint8_t> rgb;
};

struct Frame {
  std::array<Image, kCameras> cameras;
  std::vector<float> state;
  Matrix noise;
  bool generated_noise = false;
  std::string task;
};

struct Result {
  Matrix actions;
  Matrix x_t;
  Counters counters;
};

class LanguageTable {
 public:
  static std::expected<LanguageTable, std::string> Open(
      const std::filesystem::path& path);

  [[nodiscard]] std::expected<Matrix, std::string> Rows(
      std::string_view task) const;

 private:
  std::map<std::string, Matrix, std::less<>> rows_;
};

std::vector<float> Preprocess(const Image& image);
Matrix PatchRows(std::span<const float> pixels);
std::vector<std::byte> PatchInput(std::span<const Image> cameras);
Matrix LanguageInput(uint32_t rows, uint32_t first, const Matrix& language);
Matrix StateInput(std::span<const float> state);
Matrix NoiseInput(uint32_t rows, const Matrix& noise);
Matrix GenerateNoise(std::mt19937_64& generator);
Matrix Actions(const Matrix& x_t);

std::expected<Frame, std::string> ReadFrame(const SafeTensors& file,
                                            std::mt19937_64& generator);
std::expected<Result, std::string> RunFrame(Runtime& runtime,
                                            const Frame& frame,
                                            const Matrix& language,
                                            std::chrono::milliseconds timeout);
std::vector<std::pair<std::string, Tensor>> OutputTensors(const Frame& frame,
                                                          const Result& result);

}  // namespace hotarubi::smolvla

#endif  // HOTARUBI_TOOLS_RUNTIME_SMOLVLA_PS_HPP_
