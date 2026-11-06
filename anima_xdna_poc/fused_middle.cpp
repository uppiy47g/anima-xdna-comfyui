#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" __declspec(dllexport) int avx512_supported() {
    int registers[4];
    __cpuid(registers, 1);
    if (!(registers[2] & (1 << 27)) || (_xgetbv(0) & 0xe6) != 0xe6) return 0;
    __cpuidex(registers, 7, 0);
    const unsigned required = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    return (static_cast<unsigned>(registers[1]) & required) == required;
}

extern "C" __declspec(dllexport) int fused_middle(
    const int32_t* accumulator, const float* row_scale, const float* weight_scale,
    const float* gelu_bf16_lut, int8_t* output, float* output_scale,
    float* scratch, int rows, int columns, int threads) {
    if (!accumulator || !row_scale || !weight_scale || !gelu_bf16_lut ||
        !output || !output_scale || !scratch || rows <= 0 ||
        columns <= 0 || columns % 16 || threads < 1 || threads > 64) return 1;
    for (int c = 0; c < columns; ++c) {
        if (!(weight_scale[c] > 0) || !std::isfinite(weight_scale[c])) return 2;
    }

    int invalid = 0;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
    for (int r = 0; r < rows; ++r) {
        if (!(row_scale[r] > 0) || !std::isfinite(row_scale[r])) {
            invalid |= 2;
            continue;
        }
        float* local = scratch + static_cast<size_t>(omp_get_thread_num()) * columns;
        const auto sx = _mm512_set1_ps(row_scale[r]);
        const auto abs_mask = _mm512_set1_epi32(0x7fffffff);
        const auto finite_max = _mm512_set1_ps(std::numeric_limits<float>::max());
        __m512 maximum = _mm512_setzero_ps();
        for (int c = 0; c < columns; c += 16) {
            auto a = _mm512_loadu_si512(accumulator + static_cast<size_t>(r) * columns + c);
            auto y = _mm512_mul_ps(_mm512_cvtepi32_ps(a), sx);
            y = _mm512_mul_ps(y, _mm512_loadu_ps(weight_scale + c));
            auto abs_y = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(y), abs_mask));
            if (_mm512_cmp_ps_mask(abs_y, finite_max, _CMP_LE_OQ) != 0xffff) invalid |= 4;

            // BF16 has only 65536 bit patterns. The LUT preserves the reference
            // BF16 erf-GELU output, including both BF16 rounding boundaries.
            auto bits = _mm512_castps_si512(y);
            auto tie = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
            bits = _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), tie));
            auto indices = _mm512_srli_epi32(bits, 16);
            auto activated = _mm512_i32gather_ps(indices, gelu_bf16_lut, 4);
            auto magnitude = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(activated), abs_mask));
            if (_mm512_cmp_ps_mask(magnitude, finite_max, _CMP_LE_OQ) != 0xffff) invalid |= 4;
            _mm512_storeu_ps(local + c, activated);
            maximum = _mm512_max_ps(maximum, magnitude);
        }
        const float scale = std::max(_mm512_reduce_max_ps(maximum), 1.0e-12f) / 127.0f;
        output_scale[r] = scale;
        const auto divisor = _mm512_set1_ps(scale);
        for (int c = 0; c < columns; c += 16) {
            auto normalized = _mm512_div_ps(_mm512_loadu_ps(local + c), divisor);
            auto rounded = _mm512_cvt_roundps_epi32(
                normalized, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            rounded = _mm512_max_epi32(rounded, _mm512_set1_epi32(-127));
            rounded = _mm512_min_epi32(rounded, _mm512_set1_epi32(127));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(output + static_cast<size_t>(r) * columns + c),
                _mm512_cvtepi32_epi8(rounded));
        }
    }
    return invalid;
}
