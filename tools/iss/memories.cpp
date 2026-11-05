#include "memories.hpp"

#include <algorithm>
#include <cstdint>

#include "rom.hpp"

namespace hotarubi::iss {

Memories::Memories()
    : abuf(kAbufDepth),
      wbuf(kWbufDepth),
      fbuf(kFbufDepth),
      pbuf(kPbufDepth),
      tbuf(kTbufDepth),
      cbuf(kCbufDepth) {
  std::ranges::copy(kRom, cbuf.begin());
}

}  // namespace hotarubi::iss
