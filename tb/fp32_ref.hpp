#ifndef HOTARUBI_TB_FP32_REF_HPP_
#define HOTARUBI_TB_FP32_REF_HPP_

#include <cstdint>

namespace hotarubi::tb {

inline constexpr uint32_t kFp32Zero = 0x00000000U;
inline constexpr uint32_t kFp32Half = 0x3F000000U;
inline constexpr uint32_t kFp32One = 0x3F800000U;
inline constexpr uint32_t kFp32OnePointFive = 0x3FC00000U;
inline constexpr uint32_t kFp32Two = 0x40000000U;
inline constexpr uint32_t kFp32Log2e = 0x3FB8AA3BU;
inline constexpr uint32_t kFp32SignBit = 0x80000000U;

uint32_t Fp32Round(bool negative, uint64_t magnitude, int exponent);
uint32_t Fp32Add(uint32_t a, uint32_t b);
uint32_t Fp32Sub(uint32_t a, uint32_t b);
uint32_t Fp32Mul(uint32_t a, uint32_t b);
uint32_t Fp32Neg(uint32_t a);
uint32_t Fp32FromInt(int64_t value, int exponent);
uint32_t Fp32Recip(uint32_t a);
uint32_t Fp32Rsqrt(uint32_t a);
bool Fp32Less(uint32_t a, uint32_t b);
uint32_t Fp32Max(uint32_t a, uint32_t b);
uint32_t Fp32FromDouble(double value);
double Fp32ToDouble(uint32_t bits);

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_FP32_REF_HPP_
