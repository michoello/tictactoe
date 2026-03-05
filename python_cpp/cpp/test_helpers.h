#pragma once

#include "invert.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

template <typename T> inline bool approxEqual(T a, T b, double tol = 1e-3) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(a - b) <= tol;
  } else {
    return a == b; // exact for integers
  }
}

inline bool assertEqualVectors(const std::vector<std::vector<double>> &got,
                               const std::vector<std::vector<double>> &expected,
                               int round = 3) {
  float tol = std::pow(10.0f, -round);

  auto print_matrices = [expected, got, round](){
    std::cerr << "Expected:\n";
    print_matrix(expected, round);
    std::cerr << "Got:\n";
    print_matrix(got, round);
  };

  if (got.size() != expected.size()) {
    std::cerr << "Assertion failed (different number of rows):" << got.size()
              << " vs " << expected.size() << "\n";
    print_matrices();
    return false;
  }

  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i].size() != expected[i].size()) {
      std::cerr << "Assertion failed (different number of columns in row " << i
                << ")\n";
      print_matrices();
      return false;
    }
    for (size_t j = 0; j < got[i].size(); ++j) {
      if (!approxEqual(got[i][j], expected[i][j], tol)) {
        std::cerr << "Assertion failed";
        std::cerr << "Mismatch at (" << i << "," << j << "): "
                  << "expected " << expected[i][j] << " but got " << std::fixed
                  << std::setprecision(round) << got[i][j] << std::defaultfloat
                  << " (tolerance " << tol << ")\n";
        print_matrices();
        return false;
      }
    }
  }
  return true;
}

inline bool assertEqualVectors(const Matrix &got,
                               const std::vector<std::vector<double>> &expected,
                               int round = 3) {
    return assertEqualVectors(value(got), expected, round);
}

inline bool assertEqualVectors(const Matrix &got, const Matrix &expected, int round = 3) {
    return assertEqualVectors(value(got), value(expected), round);
}
