#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// Shared state per running test
inline int __checks_failed = 0;
inline int __checks_total = 0;

// Store all registered tests
struct TestCase {
  std::string name;
  std::function<void()> func;
};

inline std::vector<TestCase> &get_tests() {
  static std::vector<TestCase> tests;
  return tests;
}

// Register a test case
#define TEST_CASE(name)                                                        \
  void name();                                                                 \
  struct name##_registrar {                                                    \
    name##_registrar() { get_tests().push_back({#name, name}); }               \
  };                                                                           \
  static name##_registrar name##_instance;                                     \
  void name()

// Assertion macro
#define CHECK(agent)                                                           \
  do {                                                                         \
    ++__checks_total;                                                          \
    if (!(agent)) {                                                            \
      ++__checks_failed;                                                       \
      std::cerr << "    Failed: " #agent << " at " << __FILE__ << ":"          \
                << __LINE__ << "\n";                                           \
    }                                                                          \
  } while (0)

// Run a single test
inline bool run_test(const TestCase &t) {
  __checks_failed = 0;
  __checks_total = 0;
  std::cout << "[     ... ] " << t.name << "\n";
  t.func();
  if (__checks_failed == 0) {
    std::cout << "[ ✅ PASS ] " << t.name << " (" << __checks_total
              << " checks)\n";
    return true;
  } else {
    std::cout << "[ ❌ FAIL ] " << t.name << " (" << __checks_failed << "/"
              << __checks_total << " failed)\n";
    return false;
  }
}

// Run all or one test based on CLI
inline int run_tests(int argc, char **argv) {
  std::string filter = argc > 1 ? argv[1] : ".*";
  int total_ran = 0;
  int total_failed = 0;
  for (auto &t : get_tests()) {
    if (std::regex_match(t.name, std::regex(filter))) {
      ++total_ran;
      if (!run_test(t)) {
        ++total_failed;
      }
    }
  }
  std::cout << "\n=== Summary: " << (total_ran - total_failed)
            << " passed, " << total_failed << " failed ===\n";
  return total_failed == 0 ? 0 : 1;
}

#define ASSERT_THROWS(expression, expected_substring)                          \
    do {                                                                       \
        bool caught = false;                                                   \
        try {                                                                  \
            (expression);                                                      \
        } catch (const std::exception& e) {                                    \
            caught = true;                                                     \
            std::string_view actual_msg(e.what());                             \
            std::string_view sub(expected_substring);                          \
            if (!sub.empty() && actual_msg.find(sub) == std::string_view::npos) {\
                std::cerr << "[FAIL] Message mismatch!\n"                      \
                          << "  Expected contains: " << sub << "\n"            \
                          << "  Actual message:    " << actual_msg << "\n"     \
                          << "  At: " << __FILE__ << ":" << __LINE__ << "\n";  \
                __checks_failed++;    \
            }                                                                  \
        } catch (...) {                                                        \
            caught = true; /* Caught non-standard exception */                \
        }                                                                    \
        __checks_total++;  \
        if (!caught) {                                                         \
            __checks_failed++;    \
            std::cerr << "[FAIL] No exception thrown by: " << #expression << "\n"\
                      << "  At: " << __FILE__ << ":" << __LINE__ << "\n";      \
        } else {                                                               \
            std::cout << "[PASS] " << #expression << "\n";                     \
        }                                                                      \
    } while (0)
