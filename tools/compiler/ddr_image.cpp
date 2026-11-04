#include "ddr_image.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>

#include "isa.hpp"

namespace hotarubi::compiler {

DdrImage::DdrImage(uint64_t capacity) : bytes_(capacity) {}

uint64_t DdrImage::Alloc(std::string_view name, uint64_t size, uint64_t align) {
  const auto found = index_.find(name);
  if (reuse_ && found != index_.end()) {
    const DdrRegion& region = regions_[found->second].second;
    Require(size <= region.size,
            std::format("region {} reused with {} bytes but holds {}", name,
                        size, region.size));
    return region.offset;
  }
  const uint64_t offset = (end_ + align - 1) / align * align;
  Require(offset + size <= bytes_.size(),
          std::format("region {} of {} bytes at {:#x} exceeds the image of {} "
                      "bytes",
                      name, size, offset, bytes_.size()));
  if (found != index_.end()) {
    regions_[found->second].second = {.offset = offset, .size = size};
  } else {
    index_.emplace(std::string(name), regions_.size());
    regions_.emplace_back(std::string(name),
                          DdrRegion{.offset = offset, .size = size});
  }
  end_ = offset + size;
  return offset;
}

void DdrImage::Put(uint64_t offset, std::span<const std::byte> data) {
  Require(
      offset + data.size() <= bytes_.size(),
      std::format("{} bytes at {:#x} exceed the image", data.size(), offset));
  std::ranges::copy(data, bytes_.begin() + static_cast<std::ptrdiff_t>(offset));
}

void DdrImage::Define(std::string_view name, uint64_t offset, uint64_t size) {
  Require(offset + size <= end_ && !index_.contains(name),
          std::format("cannot define region {} at {:#x}", name, offset));
  index_.emplace(std::string(name), regions_.size());
  regions_.emplace_back(std::string(name),
                        DdrRegion{.offset = offset, .size = size});
}

const DdrRegion& DdrImage::Region(std::string_view name) const {
  const auto found = index_.find(name);
  Require(found != index_.end(), std::format("no region named {}", name));
  return regions_[found->second].second;
}

}  // namespace hotarubi::compiler
