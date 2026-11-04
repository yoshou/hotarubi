#ifndef HOTARUBI_TOOLS_COMPILER_DDR_IMAGE_HPP_
#define HOTARUBI_TOOLS_COMPILER_DDR_IMAGE_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hotarubi::compiler {

struct DdrRegion {
  uint64_t offset = 0;
  uint64_t size = 0;
};

class DdrImage {
 public:
  static constexpr uint64_t kPage = 4096;

  explicit DdrImage(uint64_t capacity);

  uint64_t Alloc(std::string_view name, uint64_t size, uint64_t align = kPage);
  void Put(uint64_t offset, std::span<const std::byte> data);
  void Define(std::string_view name, uint64_t offset, uint64_t size);
  void SetReuse(bool reuse) { reuse_ = reuse; }

  [[nodiscard]] uint64_t End() const { return end_; }
  [[nodiscard]] std::span<const std::byte> Bytes() const {
    return {bytes_.data(), end_};
  }
  [[nodiscard]] const DdrRegion& Region(std::string_view name) const;
  [[nodiscard]] const std::vector<std::pair<std::string, DdrRegion>>& Regions()
      const {
    return regions_;
  }

 private:
  std::vector<std::byte> bytes_;
  std::vector<std::pair<std::string, DdrRegion>> regions_;
  std::map<std::string, size_t, std::less<>> index_;
  uint64_t end_ = 0;
  bool reuse_ = false;
};

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_DDR_IMAGE_HPP_
