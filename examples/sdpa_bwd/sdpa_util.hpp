#pragma once
#include <cstdio>
#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include <string>

bool isclose(float a, float b, float atol, float rtol) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

template<typename T, typename V>
float
cosinesimilarity(T *refe, V *test, size_t m) {
    float ab = 0.0f;
    float a2 = 0.0f;
    float b2 = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        float t_f = (float)test[i];
        float r_f = (float)refe[i];
        ab += t_f * r_f;
        a2 += t_f * t_f;
        b2 += r_f * r_f;
    }
    float factor = ab / sqrtf(a2 * b2);
    // printf("f=%f\n", factor);
    return factor;
}

template<typename T, typename V>
float
cosinesimilarity(T *refe, V *test, size_t L, size_t M, size_t M_PAD, size_t N, size_t N_PAD) {
    float ab = 0.0f;
    float a2 = 0.0f;
    float b2 = 0.0f;
    for (size_t l = 0; l < L; ++l) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                size_t i = l * M * N + m * N + n;
                size_t j = l * M_PAD * N_PAD + m * N_PAD + n;
                float r_f = (float)refe[i];
                float t_f = (float)test[j];
                ab += t_f * r_f;
                a2 += t_f * t_f;
                b2 += r_f * r_f;
            }
        }
    }
    float factor = ab / sqrtf(a2 * b2);
    // printf("f=%f\n", factor);
    return factor;
}

template<typename T, typename V, bool is_bhsd>
float
cosinesimilarity(T *refe, V *test, size_t B, size_t H, size_t S, size_t S_PAD, size_t D) {
    float ab = 0.0f;
    float a2 = 0.0f;
    float b2 = 0.0f;
    if (is_bhsd) {
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int s = 0; s < S; ++s) {
                    for (int d = 0; d < D; ++d) {
                        int i = b * H * S * D + h * S * D + s * D + d;
                        int j = b * H * S_PAD * D + h * S_PAD * D + s * D + d;
                        float r_f = (float)refe[i];
                        float t_f = (float)test[j];
                        ab += t_f * r_f;
                        a2 += t_f * t_f;
                        b2 += r_f * r_f;
                    }
                }
            }
        }
    } else {
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                for (int h = 0; h < H; ++h) {
                    for (int d = 0; d < D; ++d) {
                        int i = b * S * H * D + s * H * D + h * D + d;
                        int j = b * S_PAD * H * D + s * H * D + h * D + d;
                        float r_f = (float)refe[i];
                        float t_f = (float)test[j];
                        ab += t_f * r_f;
                        a2 += t_f * t_f;
                        b2 += r_f * r_f;
                    }
                }
            }
        }
    }
    float factor = ab / sqrtf(a2 * b2);
    // printf("f=%f\n", factor);
    return factor;
}

template<typename T, typename V>
bool allclose(T *refe, V *test, int L, int M, int N, float atol, float rtol) {
    size_t err = 0;
    size_t count = L * M * N;
    bool flag = true;
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expect = (float)refe[l * M * N + m * N + n];
                float value = (float)test[l * M * N + m * N + n];
                // printf("(%d, %d, %d) expect: %f value: %f ratio %f\n", l, m, n, expect, value, value / expect);
                if (not isclose(expect, value, atol, rtol)) {
                    printf("(%d, %d, %d) expect: %f value: %f ratio %f\n", l, m, n, expect, value, value / expect);
                    err++;
                }
            }
        }
    }
    float ratio = static_cast<float>(count - err) / static_cast<float>(count);
    // printf("c=%f (%ld)\n", ratio, err);
    // printf("CHECK SUM SUCCESS\n");
    return ratio > 0.99f;
}

template<typename T, typename V>
bool allclose(T *refe, V *test, int L, int M, int M_PAD, int N, int N_PAD, float atol, float rtol) {
    size_t err = 0;
    size_t count = L * M * N;
    bool flag = true;
    for (int l = 0; l < L; ++l) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                int i = l * M * N + m * N + n;
                int j = l * M_PAD * N_PAD + m * N_PAD + n;
                float expect = (float)refe[i];
                float value = (float)test[j];
                // printf("(%d, %d, %d) expect: %f value: %f ratio %f\n", l, m, n, expect, value, value / expect);
                if (not isclose(expect, value, atol, rtol)) {
                    printf("(%d, %d, %d) expect: %f value: %f ratio %f\n", l, m, n, expect, value, value / expect);
                    err++;
                }
            }
        }
    }
    float ratio = static_cast<float>(count - err) / static_cast<float>(count);
    // printf("c=%f (%ld)\n", ratio, err);
    // printf("CHECK SUM SUCCESS\n");
    return ratio > 0.99f;
}

template<typename T, typename V, bool is_bhsd>
bool allclose(T *refe, V *test, int B, int H, int S, int S_PAD, int D, float atol, float rtol) {
    size_t err = 0;
    size_t count = B * S * H * D;
    bool flag = true;
    if (is_bhsd) {
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int s = 0; s < S; ++s) {
                    for (int d = 0; d < D; ++d) {
                        int i = b * H * S * D + h * S * D + s * D + d;
                        int j = b * H * S_PAD * D + h * S_PAD * D + s * D + d;
                        float expect = (float)refe[i];
                        float value = (float)test[j];
                        if (not isclose(expect, value, atol, rtol)) {
                            printf("(%d, %d, %d, %d) expect: %f value: %f ratio %f\n", b, h, s, d, expect, value, value / expect);
                            err++;
                        }
                    }
                }
            }
        }
    } else {
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                for (int h = 0; h < H; ++h) {
                    for (int d = 0; d < D; ++d) {
                    int i = b * S * H * D + s * H * D + h * D + d;
                    int j = b * S_PAD * H * D + s * H * D + h * D + d;
                    float expect = (float)refe[i];
                    float value = (float)test[j];
                    if (not isclose(expect, value, atol, rtol)) {
                        printf("(%d, %d, %d, %d) expect: %f value: %f ratio %f\n", b, s, h, d, expect, value, value / expect);
                        err++;
                    }
                    }
                }
            }
        }
    }
    float ratio = static_cast<float>(count - err) / static_cast<float>(count);
    // printf("c=%f (%ld)\n", ratio, err);
    // printf("CHECK SUM SUCCESS\n");
    return ratio > 0.99f;
}

static constexpr char strSUCCESS[] = "\x1B[32mPASS\x1B[0m";
static constexpr char strFAILURE[] = "\x1B[31mFAIL\x1B[0m";
template<typename T, typename V>
void verify(T *refe, V *test, int l, int m, int n, float atol, float rtol) {
    bool close = allclose(refe, test, l, m, n, atol, rtol);
    bool cosine = cosinesimilarity(refe, test, l * m * n) > 0.99f;
    printf("%s allclose %s cosinesim %s\n", (close and cosine) ? strSUCCESS : strFAILURE, close ? strSUCCESS : strFAILURE, cosine ? strSUCCESS : strFAILURE);
}

template<typename T, typename V>
void verify(T *refe, V *test, int l, int m, int m_pad, int n, int n_pad, float atol, float rtol) {
    bool close = allclose(refe, test, l, m, m_pad, n, n_pad, atol, rtol);
    bool cosine = cosinesimilarity(refe, test, l, m, m_pad, n, n_pad) > 0.99f;
    printf("%s allclose %s cosinesim %s\n", (close and cosine) ? strSUCCESS : strFAILURE, close ? strSUCCESS : strFAILURE, cosine ? strSUCCESS : strFAILURE);
}

template<typename T, typename V, bool is_bhsd>
void verify(T *refe, V *test, int b, int h, int s, int s_pad, int d, float atol, float rtol) {
    bool close = allclose<T, V, is_bhsd>(refe, test, b, h, s, s_pad, d, atol, rtol);
    bool cosine = cosinesimilarity<T, V, is_bhsd>(refe, test, b, h, s, s_pad, d) > 0.99f;
    printf("%s allclose %s cosinesim %s\n", (close and cosine) ? strSUCCESS : strFAILURE, close ? strSUCCESS : strFAILURE, cosine ? strSUCCESS : strFAILURE);
}

template<typename T>
void read_file(T *ptr, std::string filename, size_t rsize) {
    std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size_t fsize = file.tellg();
        assert(fsize == rsize);
        size_t len = fsize / sizeof(T);
        file.seekg(0, std::ios::beg);
        file.read((char *)ptr, len * sizeof(T));
        file.close();
    } else {
        std::cout << "fail to open " << filename << std::endl;
    }
}
