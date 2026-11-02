/**
 * @file vc_test.h
 * @brief 轻量测试框架, 用法与 NFOCC/cpp 的 tests/nf_test.h 一致。
 *
 * 不引入 gtest: 板端与交叉编译时少一个依赖, 而这里的测试都是"算个数比一下"
 * 的量级, 用不上 gtest 的一半功能。
 *
 * 用法:
 *   VC_TEST(case_name) {
 *       VC_CHECK(expr);
 *       VC_CHECK_NEAR(a, b, 1e-9);
 *   }
 *   int main() { return vc_test_run_all(); }
 */
#ifndef VC_TEST_H
#define VC_TEST_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vctest {

struct Case {
    const char *name;
    void (*fn)();
};

inline std::vector<Case> &registry() {
    static std::vector<Case> r;
    return r;
}
inline int &fail_count() { static int n = 0; return n; }
inline int &check_count() { static int n = 0; return n; }
inline const char *&current_case() { static const char *c = ""; return c; }

struct Registrar {
    Registrar(const char *name, void (*fn)()) {
        Case c;
        c.name = name;
        c.fn = fn;
        registry().push_back(c);
    }
};

inline void report_fail(const char *file, int line, const std::string &msg) {
    ++fail_count();
    std::printf("  [FAIL] %s:%d  (%s)\n        %s\n", file, line, current_case(),
                msg.c_str());
}

inline int run_all() {
    int failed_cases = 0;
    for (size_t i = 0; i < registry().size(); ++i) {
        current_case() = registry()[i].name;
        int before = fail_count();
        registry()[i].fn();
        if (fail_count() != before) {
            ++failed_cases;
            std::printf("[FAIL] %s\n", registry()[i].name);
        } else {
            std::printf("[ ok ] %s\n", registry()[i].name);
        }
    }
    std::printf("\n%d cases, %d checks, %d failed\n", (int)registry().size(),
                check_count(), failed_cases);
    return failed_cases == 0 ? 0 : 1;
}

} // namespace vctest

#define VC_TEST(name)                                                     \
    static void name();                                                   \
    static ::vctest::Registrar registrar_##name(#name, name);             \
    static void name()

#define VC_CHECK(expr)                                                    \
    do {                                                                  \
        ++::vctest::check_count();                                        \
        if (!(expr)) ::vctest::report_fail(__FILE__, __LINE__, #expr);    \
    } while (0)

#define VC_CHECK_MSG(expr, msg)                                           \
    do {                                                                  \
        ++::vctest::check_count();                                        \
        if (!(expr)) {                                                    \
            char _buf[256];                                               \
            std::snprintf(_buf, sizeof(_buf), "%s | %s", #expr, msg);     \
            ::vctest::report_fail(__FILE__, __LINE__, _buf);              \
        }                                                                 \
    } while (0)

#define VC_CHECK_NEAR(a, b, tol)                                          \
    do {                                                                  \
        ++::vctest::check_count();                                        \
        double _a = (double)(a), _b = (double)(b);                        \
        if (!(std::fabs(_a - _b) <= (tol))) {                             \
            char _buf[256];                                               \
            std::snprintf(_buf, sizeof(_buf), "%s=%g vs %s=%g (tol %g)",  \
                          #a, _a, #b, _b, (double)(tol));                 \
            ::vctest::report_fail(__FILE__, __LINE__, _buf);              \
        }                                                                 \
    } while (0)

#define VC_CHECK_EQ_INT(a, b)                                             \
    do {                                                                  \
        ++::vctest::check_count();                                        \
        long long _a = (long long)(a), _b = (long long)(b);               \
        if (_a != _b) {                                                   \
            char _buf[256];                                               \
            std::snprintf(_buf, sizeof(_buf), "%s=%lld vs %s=%lld", #a, _a, \
                          #b, _b);                                        \
            ::vctest::report_fail(__FILE__, __LINE__, _buf);              \
        }                                                                 \
    } while (0)

#define VC_CHECK_STR(a, b)                                                \
    do {                                                                  \
        ++::vctest::check_count();                                        \
        if (std::strcmp((a), (b)) != 0) {                                 \
            char _buf[256];                                               \
            std::snprintf(_buf, sizeof(_buf), "\"%s\" vs \"%s\"", (a), (b)); \
            ::vctest::report_fail(__FILE__, __LINE__, _buf);              \
        }                                                                 \
    } while (0)

inline int vc_test_run_all() { return ::vctest::run_all(); }

#endif /* VC_TEST_H */
