#include "smolvla_ps.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "frame_files.hpp"
#include "hotarubi/driver.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/formats.hpp"
#include "hotarubi/require.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"

namespace hotarubi::smolvla {
namespace {

constexpr uint32_t kImageValues = kChannels * kImageSize * kImageSize;
constexpr uint32_t kShuffle = 4;
constexpr uint32_t kShuffleGrid = kGrid / kShuffle;
constexpr uint32_t kFloatBytes = 4;
constexpr uint32_t kMaxImageSide = 1U << 14;
constexpr int kUniformBits = 53;
constexpr int kDiscardedBits = 64 - kUniformBits;

static_assert(kTokensPerCamera == kShuffleGrid * kShuffleGrid);

struct Axis {
  std::vector<uint32_t> first;
  std::vector<uint32_t> second;
  std::vector<float> first_weight;
  std::vector<float> second_weight;
};

struct Layout {
  uint32_t language_rows = 0;
  uint32_t noise_rows = 0;
  uint32_t output_rows = 0;
  std::string output;
};

Axis Interpolation(uint32_t in, uint32_t out) {
  const auto scale =
      static_cast<float>(static_cast<double>(in) / static_cast<double>(out));
  Axis axis;
  axis.first.reserve(out);
  axis.second.reserve(out);
  axis.first_weight.reserve(out);
  axis.second_weight.reserve(out);
  for (uint32_t d = 0; d < out; ++d) {
    const float center = static_cast<float>(d) + 0.5F;
    const float source = std::max((scale * center) - 0.5F, 0.0F);
    const uint32_t first = std::min(static_cast<uint32_t>(source), in - 1);
    const float second_weight = source - static_cast<float>(first);
    axis.first.push_back(first);
    axis.second.push_back(std::min(first + 1, in - 1));
    axis.first_weight.push_back(1.0F - second_weight);
    axis.second_weight.push_back(second_weight);
  }
  return axis;
}

float Fma64(float a, float b, float c) {
  return static_cast<float>((static_cast<double>(a) * static_cast<double>(b)) +
                            static_cast<double>(c));
}

std::vector<float> Resize(std::span<const float> image, uint32_t height,
                          uint32_t width, uint32_t out_height,
                          uint32_t out_width) {
  const Axis rows = Interpolation(height, out_height);
  const Axis columns = Interpolation(width, out_width);
  std::vector<float> out(static_cast<size_t>(kChannels) * out_height *
                         out_width);
  const size_t plane_values = static_cast<size_t>(height) * width;
  for (uint32_t c = 0; c < kChannels; ++c) {
    const std::span<const float> plane =
        image.subspan(c * plane_values, plane_values);
    for (uint32_t y = 0; y < out_height; ++y) {
      const std::span<const float> top_row =
          plane.subspan(static_cast<size_t>(rows.first[y]) * width, width);
      const std::span<const float> bottom_row =
          plane.subspan(static_cast<size_t>(rows.second[y]) * width, width);
      const size_t base =
          ((static_cast<size_t>(c) * out_height) + y) * out_width;
      for (uint32_t x = 0; x < out_width; ++x) {
        const float top =
            Fma64(columns.first_weight[x], top_row[columns.first[x]],
                  columns.second_weight[x] * top_row[columns.second[x]]);
        const float bottom =
            Fma64(columns.first_weight[x], bottom_row[columns.first[x]],
                  columns.second_weight[x] * bottom_row[columns.second[x]]);
        out[base + x] =
            Fma64(rows.first_weight[y], top, rows.second_weight[y] * bottom);
      }
    }
  }
  return out;
}

std::vector<float> Floats(const Matrix& matrix) {
  std::vector<float> values(matrix.bits.size());
  std::ranges::transform(matrix.bits, values.begin(), [](uint32_t bits) {
    return std::bit_cast<float>(bits);
  });
  return values;
}

std::expected<Image, std::string> ReadImage(const SafeTensors& file,
                                            std::string_view name) {
  if (!file.Contains(name)) {
    return std::unexpected(std::format("no tensor {}", name));
  }
  const TensorInfo& info = file.Info(name);
  if (info.dtype != DType::kU8 || info.shape.size() != 3 ||
      info.shape[2] != kChannels || info.shape[0] == 0 || info.shape[1] == 0 ||
      info.shape[0] > kMaxImageSide || info.shape[1] > kMaxImageSide) {
    return std::unexpected(
        std::format("tensor {} is not a U8 image of shape [H, W, 3]", name));
  }
  return Image{.height = static_cast<uint32_t>(info.shape[0]),
               .width = static_cast<uint32_t>(info.shape[1]),
               .rgb = file.Bytes(name)};
}

std::expected<uint32_t, std::string> RegionRows(const Runtime& runtime,
                                                std::string_view name,
                                                uint32_t width,
                                                uint32_t min_rows) {
  const auto region = runtime.FindRegion(name);
  if (!region) {
    return std::unexpected(region.error());
  }
  const uint32_t row_bytes = width * kFloatBytes;
  const uint32_t rows = region->size / row_bytes;
  if (region->size % row_bytes != 0 || rows % 4 != 0 || rows < min_rows) {
    return std::unexpected(
        std::format("region {} of {} bytes does not hold {} rows of {} values",
                    name, region->size, min_rows, width));
  }
  return rows;
}

std::expected<Layout, std::string> ReadLayout(const Runtime& runtime,
                                              uint32_t language_rows) {
  const auto cameras = runtime.Parameter("cameras");
  const auto lang = runtime.Parameter("n_lang");
  const auto steps = runtime.Parameter("steps");
  for (const auto* parameter : {&cameras, &lang, &steps}) {
    if (!*parameter) {
      return std::unexpected(parameter->error());
    }
  }
  if (std::cmp_not_equal(*cameras, kCameras) ||
      std::cmp_not_equal(*lang, language_rows)) {
    return std::unexpected(std::format(
        "the program takes {} cameras and {} language rows, not {} and {}",
        *cameras, *lang, kCameras, language_rows));
  }
  Layout layout{.output = std::format("x_t.{}", *steps)};
  const auto xl = RegionRows(runtime, "xl.0", kLanguageWidth,
                             (kTokensPerCamera * kCameras) + language_rows);
  const auto noise = RegionRows(runtime, "x_t.0", kActionWidth, kChunk);
  const auto output = RegionRows(runtime, layout.output, kActionWidth, kChunk);
  for (const auto* rows : {&xl, &noise, &output}) {
    if (!*rows) {
      return std::unexpected(rows->error());
    }
  }
  layout.language_rows = *xl;
  layout.noise_rows = *noise;
  layout.output_rows = *output;
  return layout;
}

}  // namespace

std::expected<LanguageTable, std::string> LanguageTable::Open(
    const std::filesystem::path& path) {
  auto file = SafeTensors::Open(path);
  if (!file) {
    return std::unexpected(file.error());
  }
  LanguageTable table;
  for (const std::string& task : file->Names()) {
    const TensorInfo& info = file->Info(task);
    if (info.shape.size() != 2 || info.shape[0] == 0 ||
        info.shape[1] != kLanguageWidth) {
      return std::unexpected(
          std::format("{}: the rows of task {} are not [n, {}]", path.string(),
                      task, kLanguageWidth));
    }
    const auto values = ReadFloats(*file, task, info.shape);
    if (!values) {
      return std::unexpected(
          std::format("{}: {}", path.string(), values.error()));
    }
    Matrix rows{.rows = static_cast<uint32_t>(info.shape[0]),
                .cols = kLanguageWidth,
                .bits = std::vector<uint32_t>(values->size())};
    std::ranges::transform(*values, rows.bits.begin(), [](float value) {
      return std::bit_cast<uint32_t>(value);
    });
    table.rows_.emplace(task, std::move(rows));
  }
  return table;
}

std::expected<Matrix, std::string> LanguageTable::Rows(
    std::string_view task) const {
  const auto found = rows_.find(task);
  if (found == rows_.end()) {
    return std::unexpected(
        std::format("no language rows for task \"{}\"", task));
  }
  return found->second;
}

std::vector<float> Preprocess(const Image& image) {
  const uint32_t height = image.height;
  const uint32_t width = image.width;
  Require(
      height > 0 && width > 0 &&
          image.rgb.size() == static_cast<size_t>(height) * width * kChannels,
      "the image does not hold height x width RGB pixels");
  std::vector<float> pixels(static_cast<size_t>(kChannels) * height * width);
  for (uint32_t c = 0; c < kChannels; ++c) {
    for (size_t i = 0; i < static_cast<size_t>(height) * width; ++i) {
      pixels[(c * static_cast<size_t>(height) * width) + i] =
          static_cast<float>(image.rgb[(i * kChannels) + c]) / 255.0F;
    }
  }
  std::vector<float> out(kImageValues, 0.0F);
  if (height == kImageSize && width == kImageSize) {
    out = std::move(pixels);
  } else {
    const double ratio = std::max(static_cast<double>(width) / kImageSize,
                                  static_cast<double>(height) / kImageSize);
    const auto out_height =
        static_cast<uint32_t>(static_cast<double>(height) / ratio);
    const auto out_width =
        static_cast<uint32_t>(static_cast<double>(width) / ratio);
    Require(out_height > 0 && out_width > 0 && out_height <= kImageSize &&
                out_width <= kImageSize,
            "the image cannot be resized into 512 x 512");
    const std::vector<float> resized =
        Resize(pixels, height, width, out_height, out_width);
    for (uint32_t c = 0; c < kChannels; ++c) {
      for (uint32_t y = 0; y < out_height; ++y) {
        const size_t from =
            ((static_cast<size_t>(c) * out_height) + y) * out_width;
        const size_t row = (static_cast<size_t>(c) * kImageSize) +
                           (kImageSize - out_height) + y;
        const size_t to = (row * kImageSize) + (kImageSize - out_width);
        std::copy_n(resized.begin() + static_cast<std::ptrdiff_t>(from),
                    out_width, out.begin() + static_cast<std::ptrdiff_t>(to));
      }
    }
  }
  for (float& value : out) {
    value = (value * 2.0F) - 1.0F;
  }
  return out;
}

Matrix PatchRows(std::span<const float> pixels) {
  Require(pixels.size() == kImageValues, "the image is not 3 x 512 x 512");
  Matrix rows{.rows = kPatchTokens,
              .cols = kPatchValues,
              .bits = std::vector<uint32_t>(static_cast<size_t>(kPatchTokens) *
                                            kPatchValues)};
  for (uint32_t p = 0; p < kPatchTokens; ++p) {
    const uint32_t group = p / kTokensPerCamera;
    const uint32_t py = (kShuffle * ((p % kTokensPerCamera) / kShuffleGrid)) +
                        (group / kShuffle);
    const uint32_t px = (kShuffle * (p % kShuffleGrid)) + (group % kShuffle);
    for (uint32_t k = 0; k < kPatchValues; ++k) {
      const uint32_t c = k / (kPatch * kPatch);
      const uint32_t y = (py * kPatch) + ((k / kPatch) % kPatch);
      const uint32_t x = (px * kPatch) + (k % kPatch);
      rows.bits[(static_cast<size_t>(p) * kPatchValues) + k] =
          std::bit_cast<uint32_t>(
              pixels[(((static_cast<size_t>(c) * kImageSize) + y) *
                      kImageSize) +
                     x]);
    }
  }
  return rows;
}

std::vector<std::byte> PatchInput(std::span<const Image> cameras) {
  std::vector<std::byte> out;
  for (const Image& camera : cameras) {
    const std::vector<std::byte> packed =
        PackAbuf(EncodeMxint8(PatchRows(Preprocess(camera))));
    out.insert(out.end(), packed.begin(), packed.end());
  }
  return out;
}

Matrix LanguageInput(uint32_t rows, uint32_t first, const Matrix& language) {
  Require(language.cols == kLanguageWidth && first + language.rows <= rows,
          "the language rows do not fit in the LM input");
  Matrix out{.rows = rows,
             .cols = kLanguageWidth,
             .bits = std::vector<uint32_t>(
                 static_cast<size_t>(rows) * kLanguageWidth, 0)};
  std::ranges::copy(
      language.bits,
      out.bits.begin() + static_cast<std::ptrdiff_t>(
                             static_cast<size_t>(first) * kLanguageWidth));
  return out;
}

Matrix StateInput(std::span<const float> state) {
  Require(state.size() <= kStateWidth, "the state has more than 32 values");
  Matrix out{.rows = kStateRows,
             .cols = kStateWidth,
             .bits = std::vector<uint32_t>(
                 static_cast<size_t>(kStateRows) * kStateWidth, 0)};
  std::ranges::transform(state, out.bits.begin(), [](float value) {
    return std::bit_cast<uint32_t>(value);
  });
  return out;
}

Matrix NoiseInput(uint32_t rows, const Matrix& noise) {
  Require(noise.cols == kActionWidth && noise.rows <= rows,
          "the noise does not fit in the action input");
  Matrix out{.rows = rows,
             .cols = kActionWidth,
             .bits = std::vector<uint32_t>(
                 static_cast<size_t>(rows) * kActionWidth, 0)};
  std::ranges::copy(noise.bits, out.bits.begin());
  return out;
}

Matrix GenerateNoise(std::mt19937_64& generator) {
  Matrix noise{.rows = kChunk,
               .cols = kActionWidth,
               .bits = std::vector<uint32_t>(static_cast<size_t>(kChunk) *
                                             kActionWidth)};
  const exact::Real two_pi = exact::Pi().Scaled(1);
  for (size_t i = 0; i < noise.bits.size(); i += 2) {
    const auto first = static_cast<int64_t>(generator() >> kDiscardedBits);
    const auto second = static_cast<int64_t>(generator() >> kDiscardedBits);
    const exact::Real uniform =
        exact::Real::Integer((2 * first) + 1).Scaled(-(kUniformBits + 1));
    const exact::Real radius =
        exact::Sqrt(exact::Log(uniform).Scaled(1).Negated());
    const exact::Real angle =
        two_pi * exact::Real::Integer(second).Scaled(-kUniformBits);
    noise.bits[i] = (radius * exact::Cos(angle)).ToBinary32();
    noise.bits[i + 1] = (radius * exact::Sin(angle)).ToBinary32();
  }
  return noise;
}

Matrix Actions(const Matrix& x_t) {
  Require(x_t.rows >= kChunk && x_t.cols == kActionWidth,
          "the action output is not [rows, 32]");
  Matrix actions{
      .rows = kChunk,
      .cols = kActionDims,
      .bits = std::vector<uint32_t>(static_cast<size_t>(kChunk) * kActionDims)};
  for (uint32_t r = 0; r < kChunk; ++r) {
    for (uint32_t c = 0; c < kActionDims; ++c) {
      actions.bits[(static_cast<size_t>(r) * kActionDims) + c] = x_t.At(r, c);
    }
  }
  return actions;
}

std::expected<Frame, std::string> ReadFrame(const SafeTensors& file,
                                            std::mt19937_64& generator) {
  Frame frame;
  for (uint32_t c = 0; c < kCameras; ++c) {
    auto image = ReadImage(file, std::format("camera{}", c + 1));
    if (!image) {
      return std::unexpected(image.error());
    }
    frame.cameras.at(c) = std::move(*image);
  }
  if (!file.Contains("state") || file.Info("state").shape.size() != 1 ||
      file.Info("state").shape[0] > kStateWidth) {
    return std::unexpected("tensor state is not [n] with n <= 32");
  }
  auto state = ReadFloats(file, "state", file.Info("state").shape);
  if (!state) {
    return std::unexpected(state.error());
  }
  frame.state = std::move(*state);
  if (file.Contains("noise")) {
    const auto noise = ReadFloats(file, "noise", {kChunk, kActionWidth});
    if (!noise) {
      return std::unexpected(noise.error());
    }
    frame.noise = {.rows = kChunk,
                   .cols = kActionWidth,
                   .bits = std::vector<uint32_t>(noise->size())};
    std::ranges::transform(*noise, frame.noise.bits.begin(), [](float value) {
      return std::bit_cast<uint32_t>(value);
    });
  } else {
    frame.noise = GenerateNoise(generator);
    frame.generated_noise = true;
  }
  auto task = ReadText(file, "task");
  if (!task) {
    return std::unexpected(task.error());
  }
  frame.task = std::move(*task);
  return frame;
}

std::expected<Result, std::string> RunFrame(Runtime& runtime,
                                            const Frame& frame,
                                            const Matrix& language,
                                            std::chrono::milliseconds timeout,
                                            const TraceRequest& trace) {
  if (frame.state.size() > kStateWidth || frame.noise.rows != kChunk ||
      frame.noise.cols != kActionWidth) {
    return std::unexpected("the state or the noise has the wrong size");
  }
  Timing timing;
  auto clock = std::chrono::steady_clock::now();
  const auto lap = [&clock] {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - clock;
    clock = now;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
  };
  const auto layout = ReadLayout(runtime, language.rows);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  const std::vector<std::pair<std::string_view, std::vector<std::byte>>>
      inputs = {
          {"patch.in", PatchInput(frame.cameras)},
          {"xl.0",
           PackFbuf(LanguageInput(layout->language_rows,
                                  kTokensPerCamera * kCameras, language))},
          {"state.in", PackFbuf(StateInput(frame.state))},
          {"x_t.0", PackFbuf(NoiseInput(layout->noise_rows, frame.noise))}};
  for (const auto& [name, bytes] : inputs) {
    if (auto written = runtime.Write(name, bytes); !written) {
      return std::unexpected(written.error());
    }
  }
  timing.inputs = lap();
  auto counters = runtime.Run(timeout, {}, trace);
  timing.run = lap();
  if (!counters) {
    return std::unexpected(counters.error());
  }
  std::vector<std::byte> output(static_cast<size_t>(layout->output_rows) *
                                kActionWidth * kFloatBytes);
  if (auto read = runtime.Read(layout->output, output); !read) {
    return std::unexpected(read.error());
  }
  Matrix x_t = UnpackFbuf(output, layout->output_rows, kActionWidth);
  Matrix actions = Actions(x_t);
  timing.outputs = lap();
  return Result{.actions = std::move(actions),
                .x_t = std::move(x_t),
                .counters = *counters,
                .timing = timing};
}

std::vector<std::pair<std::string, Tensor>> OutputTensors(
    const Frame& frame, const Result& result) {
  std::vector<std::pair<std::string, Tensor>> tensors;
  tensors.emplace_back(
      "actions", FloatTensor({kChunk, kActionDims}, Floats(result.actions)));
  tensors.emplace_back("x_t", FloatTensor({result.x_t.rows, result.x_t.cols},
                                          Floats(result.x_t)));
  if (frame.generated_noise) {
    tensors.emplace_back(
        "noise", FloatTensor({kChunk, kActionWidth}, Floats(frame.noise)));
  }
  return tensors;
}

}  // namespace hotarubi::smolvla
