// Reuse the validated BF16 rounding and deployed PyTorch RMSNorm reduction.
#include "fused_qk.cpp"
#include <algorithm>

namespace {

inline __m512 absolute(__m512 value) {
    return _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(0x7fffffff)));
}

inline const uint16_t* token_slice(
    const uint16_t* source, int row, int column, int rows, int columns, bool head_major) {
    if (!head_major) return source + static_cast<size_t>(row) * columns + column;
    return source + (static_cast<size_t>(column / depth) * rows + row) * depth + column % depth;
}

}  // namespace

extern "C" __declspec(dllexport) int quantize_attention_input(
    const uint16_t* source, int8_t* output, float* scales,
    int rows, int columns, int head_major, int threads) {
    if (!source || !output || !scales || (rows != 512 && rows != 1024) ||
        (columns != 1024 && columns != 2048) ||
        (head_major != 0 && head_major != 1) ||
        (head_major && (rows != 1024 || columns != hidden)) ||
        threads < 1 || threads > 64) return 1;
    int invalid = 0;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
    for (int r = 0; r < rows; ++r) {
        __m512 maximum = _mm512_setzero_ps();
        bool valid = true;
        for (int c = 0; c < columns; c += 16) {
            const auto v = load_bf16(token_slice(source, r, c, rows, columns, head_major != 0));
            if (!finite(v)) valid = false;
            maximum = _mm512_max_ps(maximum, absolute(v));
        }
        if (!valid) {
            invalid |= 4;
            continue;
        }
        const float scale = std::max(_mm512_reduce_max_ps(maximum), 1.0e-12f) / 127.0f;
        scales[r] = scale;
        const auto divisor = _mm512_set1_ps(scale);
        for (int c = 0; c < columns; c += 16) {
            const auto v = load_bf16(token_slice(source, r, c, rows, columns, head_major != 0));
            auto rounded = _mm512_cvt_roundps_epi32(
                _mm512_div_ps(v, divisor), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            rounded = _mm512_max_epi32(rounded, _mm512_set1_epi32(-127));
            rounded = _mm512_min_epi32(rounded, _mm512_set1_epi32(127));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(output + static_cast<size_t>(r) * columns + c),
                _mm512_cvtepi32_epi8(rounded));
        }
    }
    return invalid;
}

extern "C" __declspec(dllexport) int prepare_attention_i32(
    const int32_t* query, const int32_t* key, const int32_t* value,
    const float* query_scale, const float* key_scale, const float* value_scale,
    const float* query_wscale, const float* key_wscale, const float* value_wscale,
    const uint16_t* query_weight, const uint16_t* key_weight,
    const float* cosine, const float* sine,
    uint16_t* out_query, uint16_t* out_key, uint16_t* out_value,
    int queries, int keys, int threads) {
    if (!query || !key || !value || !query_scale || !key_scale || !value_scale ||
        !query_wscale || !key_wscale || !value_wscale || !query_weight || !key_weight ||
        !out_query || !out_key || !out_value || queries != 1024 ||
        (keys != 512 && keys != 1024) || threads < 1 || threads > 64 ||
        (!!cosine != !!sine) || (cosine && keys != queries)) return 1;
    for (int c = 0; c < hidden; ++c) {
        for (const float* scales : {query_wscale, key_wscale, value_wscale}) {
            if (!(scales[c] > 0) || !std::isfinite(scales[c])) return 2;
        }
    }
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
        const int32_t* source = (is_query ? query : is_value ? value : key)
            + static_cast<size_t>(token) * hidden + head * depth;
        const float* weight_scale = (is_query ? query_wscale : is_value ? value_wscale : key_wscale)
            + head * depth;
        const float row_scale = (is_query ? query_scale : is_value ? value_scale : key_scale)[token];
        if (!(row_scale > 0) || !std::isfinite(row_scale)) {
            invalid |= 2;
            continue;
        }
        uint16_t* destination = (is_query ? out_query : is_value ? out_value : out_key)
            + static_cast<size_t>(row) * depth;
        alignas(32) uint16_t dequantized[depth];
        bool valid = true;
        for (int c = 0; c < depth; c += 16) {
            auto v = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_loadu_si512(source + c)), _mm512_set1_ps(row_scale));
            v = _mm512_mul_ps(v, _mm512_loadu_ps(weight_scale + c));
            if (!finite(v)) valid = false;
            v = bf16(v);
            if (!finite(v)) valid = false;
            store((is_value ? destination : dequantized) + c, v);
        }
        if (!valid) {
            invalid |= 4;
            continue;
        }
        if (!is_value) {
            invalid |= normalize_row(
                dequantized, is_query ? query_weight : key_weight,
                cosine ? cosine + token * depth : nullptr,
                sine ? sine + token * depth : nullptr, destination);
        }
    }
    return invalid;
}
