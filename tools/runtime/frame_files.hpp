#ifndef HOTARUBI_TOOLS_RUNTIME_FRAME_FILES_HPP_
#define HOTARUBI_TOOLS_RUNTIME_FRAME_FILES_HPP_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/safetensors.hpp"

namespace hotarubi {

std::expected<std::vector<float>, std::string> ReadFloats(
    const SafeTensors& file, std::string_view name,
    const std::vector<uint64_t>& shape);
std::expected<std::string, std::string> ReadText(const SafeTensors& file,
                                                 std::string_view key);
std::expected<std::vector<std::filesystem::path>, std::string> FramePaths(
    const std::filesystem::path& directory);

}  // namespace hotarubi

#endif  // HOTARUBI_TOOLS_RUNTIME_FRAME_FILES_HPP_
