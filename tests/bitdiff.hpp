// Bit-exactness helpers for diffing DOD against official Open_VINS.
//
// The unit of comparison is ULP distance, and the pass bar is 0. Anything
// looser cannot distinguish "harmless rounding" from "wrong formula", which is
// precisely the ambiguity that has kept the parity gap alive.
//
// TEST-ONLY header. Nothing here is linked into the shipped library.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <Eigen/Dense>

namespace bitdiff {

// Signed-magnitude -> biased so that adjacent doubles are adjacent integers.
inline int64_t ordered(double x) {
    int64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits < 0 ? INT64_MIN - bits : bits;
}

// ULP distance. NaN vs NaN is 0 (same class); NaN vs number is "infinite".
inline int64_t ulp_diff(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return 0;
    if (std::isnan(a) || std::isnan(b)) return INT64_MAX;
    int64_t d = ordered(a) - ordered(b);
    return d < 0 ? -d : d;
}

inline bool bitexact(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

struct Report {
    int checks = 0;
    int failures = 0;

    // Compares elementwise. `dod` and `ov` must already be the same shape --
    // a shape mismatch is itself a failure worth shouting about.
    template <typename A, typename B>
    bool expect(const char* name, const A& dod, const B& ov, int64_t max_ulp = 0) {
        ++checks;
        if (dod.rows() != ov.rows() || dod.cols() != ov.cols()) {
            std::printf("[FAIL] %s: shape %ldx%ld (dod) vs %ldx%ld (ov)\n", name,
                        (long)dod.rows(), (long)dod.cols(), (long)ov.rows(), (long)ov.cols());
            ++failures;
            return false;
        }
        int64_t worst = 0;
        Eigen::Index wr = -1, wc = -1;
        for (Eigen::Index c = 0; c < dod.cols(); ++c) {
            for (Eigen::Index r = 0; r < dod.rows(); ++r) {
                int64_t u = ulp_diff(dod(r, c), ov(r, c));
                if (u > worst) { worst = u; wr = r; wc = c; }
            }
        }
        if (worst > max_ulp) {
            double a = dod(wr, wc), b = ov(wr, wc);
            std::printf("[FAIL] %s: worst %lld ULP at (%ld,%ld)\n", name,
                        (long long)worst, (long)wr, (long)wc);
            std::printf("         dod = %.17g  (%a)\n", a, a);
            std::printf("         ov  = %.17g  (%a)\n", b, b);
            ++failures;
            return false;
        }
        return true;
    }

    bool expect_scalar(const char* name, double dod, double ov, int64_t max_ulp = 0) {
        ++checks;
        int64_t u = ulp_diff(dod, ov);
        if (u > max_ulp) {
            std::printf("[FAIL] %s: %lld ULP\n", name, (long long)u);
            std::printf("         dod = %.17g  (%a)\n", dod, dod);
            std::printf("         ov  = %.17g  (%a)\n", ov, ov);
            ++failures;
            return false;
        }
        return true;
    }

    // Returns a process exit code: 0 iff everything was bit-exact.
    int finish(const char* stage) const {
        std::printf("%s: %d checks, %d failures\n", stage, checks, failures);
        return failures == 0 ? 0 : 1;
    }
};

// Deterministic input generation. Never Eigen::Random -- inputs must be
// reproducible across builds and machines for a diff to mean anything.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed ? seed : 1) {}
    uint64_t next() {  // splitmix64
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform(double lo, double hi) {
        // 53-bit mantissa fill, exactly representable and platform-independent.
        double u = (double)(next() >> 11) * (1.0 / 9007199254740992.0);
        return lo + u * (hi - lo);
    }
    Eigen::VectorXd vec(int n, double lo, double hi) {
        Eigen::VectorXd v(n);
        for (int i = 0; i < n; ++i) v(i) = uniform(lo, hi);
        return v;
    }
};

}  // namespace bitdiff
