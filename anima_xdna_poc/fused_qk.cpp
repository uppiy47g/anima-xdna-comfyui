#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int heads = 16;
constexpr int depth = 128;
constexpr int hidden = heads * depth;

inline __m512 load_bf16(const uint16_t* p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))), 16));
}

inline __m256 load_bf16x8(const uint16_t* p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))), 16));
}

inline bool finite(__m512 x) {
    const auto magnitude = _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
    return _mm512_cmp_ps_mask(
        magnitude, _mm512_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ) == 0xffff;
}

inline __m512 bf16(__m512 x) {
    const auto bits = _mm512_castps_si512(x);
    const auto tie = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    return _mm512_castsi512_ps(_mm512_and_si512(
        _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), tie)),
        _mm512_set1_epi32(static_cast<int>(0xffff0000u))));
}

inline void store(uint16_t* p, __m512 x) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm512_cvtepi32_epi16(
        _mm512_srli_epi32(_mm512_castps_si512(x), 16)));
}

// Deployed PyTorch's FP32 sum uses four 8-lane accumulators, combines them
// left-to-right, then adds lanes serially. A tree reduction changes BF16 ties.
inline float inverse_rms(const uint16_t* p) {
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    for (int c = 0; c < depth; c += 32) {
        for (int a = 0; a < 4; ++a) {
            const auto v = load_bf16x8(p + c + a * 8);
            sums[a] = _mm256_add_ps(sums[a], _mm256_mul_ps(v, v));
        }
    }
    const auto combined = _mm256_add_ps(
        _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), sums[2]), sums[3]);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, combined);
    float total = 0;
    for (int i = 0; i < 8; ++i) total = total + lanes[i];
    const float variance = total / 128.0f;
    if (!std::isfinite(variance)) return std::numeric_limits<float>::quiet_NaN();
    return 1.0f / std::sqrt(variance + 1.0e-6f);
}

int normalize_row(const uint16_t* input, const uint16_t* weight,
                  const float* cosine, const float* sine, uint16_t* output) {
    const float inv = inverse_rms(input);
    if (!std::isfinite(inv)) return 4;
    const auto factor = _mm512_set1_ps(inv);
    for (int c = 0; c < depth / 2; c += 16) {
        const auto a = bf16(_mm512_mul_ps(
            bf16(_mm512_mul_ps(load_bf16(input + c), factor)), load_bf16(weight + c)));
        const auto b = bf16(_mm512_mul_ps(
            bf16(_mm512_mul_ps(load_bf16(input + c + 64), factor)), load_bf16(weight + c + 64)));
        if (!finite(a) || !finite(b)) return 4;
        auto first = a;
        auto second = b;
        if (cosine) {
            const auto negative_b = _mm512_castsi512_ps(_mm512_xor_si512(
                _mm512_castps_si512(b), _mm512_set1_epi32(static_cast<int>(0x80000000u))));
            first = _mm512_add_ps(
                _mm512_mul_ps(a, _mm512_loadu_ps(cosine + c)),
                _mm512_mul_ps(negative_b, _mm512_loadu_ps(sine + c)));
            second = _mm512_add_ps(
                _mm512_mul_ps(b, _mm512_loadu_ps(cosine + c + 64)),
                _mm512_mul_ps(a, _mm512_loadu_ps(sine + c + 64)));
        }
        if (!finite(first) || !finite(second)) return 4;
        first = bf16(first);
        second = bf16(second);
        if (!finite(first) || !finite(second)) return 4;
        store(output + c, first);
        store(output + c + 64, second);
    }
    return 0;
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

// Token-major Q/K/V -> contiguous head-major AIR inputs. Null rotary pointers
// select cross-attention; self-attention uses split-half (not interleaved) RoPE.
extern "C" __declspec(dllexport) int prepare_attention(
    const uint16_t* query, const uint16_t* key, const uint16_t* value,
    const uint16_t* query_weight, const uint16_t* key_weight,
    const float* cosine, const float* sine,
    uint16_t* out_query, uint16_t* out_key, uint16_t* out_value,
    int queries, int keys, int threads) {
    if (!query || !key || !value || !query_weight || !key_weight ||
        !out_query || !out_key || !out_value || queries != 1024 ||
        (keys != 512 && keys != 1024) || threads < 1 || threads > 64 ||
        (!!cosine != !!sine) || (cosine && keys != queries)) return 1;
    for (int c = 0; c < depth; c += 16) {
        if (!finite(load_bf16(query_weight + c)) || !finite(load_bf16(key_weight + c))) return 4;
    }
    int invalid = 0;
    const int query_count = heads * queries;
    const int key_count = heads * keys;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
    for (int i = 0; i < query_count + 2 * key_count; ++i) {
        const bool is_query = i < query_count;
        const bool is_value = i >= query_count + key_count;
        const int row = is_query ? i : i - query_count - (is_value ? key_count : 0);
        const int rows = is_query ? queries : keys;
        const int head = row / rows;
        const int token = row % rows;
        const uint16_t* source = (is_query ? query : is_value ? value : key)
            + static_cast<size_t>(token) * hidden + head * depth;
        uint16_t* destination = (is_query ? out_query : is_value ? out_value : out_key)
            + static_cast<size_t>(row) * depth;
        if (is_value) {
            for (int c = 0; c < depth; c += 16) {
                const auto v = load_bf16(source + c);
                if (!finite(v)) invalid |= 4;
                store(destination + c, v);
            }
        } else {
            invalid |= normalize_row(
                source, is_query ? query_weight : key_weight,
                cosine ? cosine + token * depth : nullptr,
                sine ? sine + token * depth : nullptr, destination);
        }
    }
    return invalid;
}
