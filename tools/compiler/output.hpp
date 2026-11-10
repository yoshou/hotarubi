#ifndef HOTARUBI_TOOLS_COMPILER_OUTPUT_HPP_
#define HOTARUBI_TOOLS_COMPILER_OUTPUT_HPP_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "program.hpp"

namespace hotarubi::compiler {

struct Stream {
  std::string name;
  Program program;
  uint64_t offset = 0;
};

struct Compiled {
  std::vector<Stream> streams;
  DdrImage image;
  std::vector<std::pair<std::string, uint64_t>> parameters;
};

inline constexpr uint64_t kSharedBytes = uint64_t{320} << 20;
inline constexpr std::string_view kTraceRegion = "trace";
inline constexpr uint64_t kTraceRecordsPerInstruction = 2;

[[nodiscard]] uint64_t TraceRegionBytes(const std::vector<Stream>& streams);
uint64_t ReserveTraceRegion(DdrImage& image,
                            const std::vector<Stream>& streams);
[[nodiscard]] std::string ListingFileName(std::string_view stream);
[[nodiscard]] std::string MetaJson(const Compiled& compiled);
[[nodiscard]] std::expected<void, std::string> WriteCompiled(
    const std::filesystem::path& directory, const Compiled& compiled);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_OUTPUT_HPP_
