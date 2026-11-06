#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

inline __m512 load_bf16(const uint16_t* source) {
    const auto packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
}

// Integer round-to-nearest-even identical to c10::BFloat16 for finite inputs.
// vcvtneps2bf16 is deliberately avoided because its denormal handling differs.
inline __m512 round_bf16(__m512 value) {
    const auto bits = _mm512_castps_si512(value);
    const auto tie = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    const auto biased = _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), tie));
    return _mm512_castsi512_ps(
        _mm512_and_si512(biased, _mm512_set1_epi32(static_cast<int>(0xffff0000u))));
}

inline void store_bf16(uint16_t* destination, __m512 rounded) {
    const auto narrowed = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(rounded), 16));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), narrowed);
}

inline __m512 magnitude(__m512 value) {
    return _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(0x7fffffff)));
}

inline bool all_finite(__m512 value) {
    return _mm512_cmp_ps_mask(
        magnitude(value), _mm512_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ) == 0xffff;
}

}  // namespace

extern "C" __declspec(dllexport) int avx512_supported() {
    int registers[4];
    __cpuid(registers, 1);
    if (!(registers[2] & (1 << 27)) || (_xgetbv(0) & 0xe6) != 0xe6) return 0;
    __cpuidex(registers, 7, 0);
    const unsigned required = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    return (static_cast<unsigned>(registers[1]) & required) == required;
}

// normalized * (1 + scale) + shift with the three BF16 roundings of the PyTorch
// reference, followed by per-row symmetric INT8 quantization identical to
// round(x / (max(amax, 1e-12) / 127)).clamp(-127, 127).
// scratch holds (2 + threads) * columns floats.
extern "C" __declspec(dllexport) int modulate_quantize(
    const uint16_t* normalized, const uint16_t* shift, const uint16_t* scale,
    int8_t* output, float* output_scale, float* scratch,
    int rows, int columns, int threads) {
    if (!normalized || !shift || !scale || !output || !output_scale || !scratch ||
        rows <= 0 || columns <= 0 || columns % 16 || threads < 1 || threads > 64) return 1;
    float* factor = scratch;
    float* offset = scratch + columns;
    float* locals = scratch + 2 * static_cast<size_t>(columns);
    for (int c = 0; c < columns; c += 16) {
        _mm512_storeu_ps(factor + c, round_bf16(_mm512_add_ps(
            _mm512_set1_ps(1.0f), load_bf16(scale + c))));
        _mm512_storeu_ps(offset + c, load_bf16(shift + c));
    }

    int invalid = 0;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
    for (int r = 0; r < rows; ++r) {
        float* local = locals + static_cast<size_t>(omp_get_thread_num()) * columns;
        const uint16_t* row = normalized + static_cast<size_t>(r) * columns;
        __m512 maximum = _mm512_setzero_ps();
        for (int c = 0; c < columns; c += 16) {
            const auto scaled = round_bf16(
                _mm512_mul_ps(load_bf16(row + c), _mm512_loadu_ps(factor + c)));
            const auto modulated = round_bf16(
                _mm512_add_ps(scaled, _mm512_loadu_ps(offset + c)));
            if (!all_finite(modulated)) invalid |= 4;
            _mm512_storeu_ps(local + c, modulated);
            maximum = _mm512_max_ps(maximum, magnitude(modulated));
        }
        const float row_quant_scale = std::max(_mm512_reduce_max_ps(maximum), 1.0e-12f) / 127.0f;
        output_scale[r] = row_quant_scale;
        const auto divisor = _mm512_set1_ps(row_quant_scale);
        for (int c = 0; c < columns; c += 16) {
            auto normalized_value = _mm512_div_ps(_mm512_loadu_ps(local + c), divisor);
            auto rounded = _mm512_cvt_roundps_epi32(
                normalized_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            rounded = _mm512_max_epi32(rounded, _mm512_set1_epi32(-127));
            rounded = _mm512_min_epi32(rounded, _mm512_set1_epi32(127));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(output + static_cast<size_t>(r) * columns + c),
                _mm512_cvtepi32_epi8(rounded));
        }
    }
    return invalid;
}

// hidden + gate * bf16((float(acc) * row_scale) * weight_scale) with the BF16
// roundings of the PyTorch reference: ff, gate * ff, and hidden + (gate * ff).
extern "C" __declspec(dllexport) int dequant_gated_residual(
    const int32_t* accumulator, const float* row_scale, const float* weight_scale,
    const uint16_t* gate, const uint16_t* hidden, uint16_t* output,
    int rows, int columns, int threads) {
    if (!accumulator || !row_scale || !weight_scale || !gate || !hidden || !output ||
        rows <= 0 || columns <= 0 || columns % 16 || threads < 1 || threads > 64) return 1;
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
        const auto sx = _mm512_set1_ps(row_scale[r]);
        const size_t base = static_cast<size_t>(r) * columns;
        for (int c = 0; c < columns; c += 16) {
            auto value = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_loadu_si512(accumulator + base + c)), sx);
            value = _mm512_mul_ps(value, _mm512_loadu_ps(weight_scale + c));
            const auto feed_forward = round_bf16(value);
            const auto gated = round_bf16(_mm512_mul_ps(load_bf16(gate + c), feed_forward));
            const auto updated = round_bf16(_mm512_add_ps(load_bf16(hidden + base + c), gated));
            if (!all_finite(updated)) invalid |= 4;
            store_bf16(output + base + c, updated);
        }
    }
    return invalid;
}
