#ifndef HOTARUBI_TOOLS_COMPILER_EXACT_HPP_
#define HOTARUBI_TOOLS_COMPILER_EXACT_HPP_

#include <cstdint>
#include <vector>

namespace hotarubi::compiler::exact {

class Real {
 public:
  static constexpr int kBits = 384;

  Real() = default;

  static Real Integer(int64_t value);
  static Real Ratio(int64_t numerator, int64_t denominator);
  static Real Binary32(uint32_t bits);
  static Real Binary64(uint64_t bits);

  [[nodiscard]] bool IsZero() const { return magnitude_.empty(); }
  [[nodiscard]] bool Negative() const { return negative_; }
  [[nodiscard]] bool Exact() const { return exact_; }
  [[nodiscard]] int64_t TopBit() const;

  [[nodiscard]] Real Negated() const;
  [[nodiscard]] Real Inexact() const;
  [[nodiscard]] Real Scaled(int64_t power_of_two) const;
  [[nodiscard]] Real DividedBy(uint32_t divisor) const;
  [[nodiscard]] int64_t Floor() const;
  [[nodiscard]] int64_t Nearest() const;

  [[nodiscard]] uint32_t ToBinary32() const;
  [[nodiscard]] uint64_t ToBinary64() const;

  friend Real operator+(const Real& a, const Real& b);
  friend Real operator-(const Real& a, const Real& b);
  friend Real operator*(const Real& a, const Real& b);
  friend Real operator/(const Real& a, const Real& b);

 private:
  struct Rounded {
    int64_t exponent = 0;
    uint64_t significand = 0;
  };

  static Real Make(bool negative, std::vector<uint32_t> magnitude,
                   int64_t exponent, bool exact);
  [[nodiscard]] Rounded Round(int significand_bits, int64_t min_exponent,
                              int64_t max_exponent) const;

  bool negative_ = false;
  std::vector<uint32_t> magnitude_;
  int64_t exponent_ = 0;
  bool exact_ = true;
};

Real Pi();
Real Ln2();
Real Sqrt(const Real& x);
Real Exp(const Real& x);
Real Log(const Real& x);
Real Pow(const Real& x, const Real& y);
Real Sin(const Real& x);
Real Cos(const Real& x);
Real Tanh(const Real& x);

}  // namespace hotarubi::compiler::exact

#endif  // HOTARUBI_TOOLS_COMPILER_EXACT_HPP_
