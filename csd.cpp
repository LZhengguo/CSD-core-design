
#include <functional>
#include <fstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <random>
#include <cstdio>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <llg-matcher.hpp>

namespace fs = std::filesystem;

#define __INFO(__fmt, ...)  _env->logger().post(Logger::INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...)  _env->logger().post(Logger::WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __ERROR(__fmt, ...) _env->logger().post(Logger::ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...)                                                                         \
    _env->logger().post(Logger::KPIS, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })
#define __DEBUG(__fmt, ...)                                                                        \
    _env->logger().post(Logger::DEBUG, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })
#define __TRACE(__fmt, ...)                                                                        \
    _env->logger().post(Logger::TRACE, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })

namespace CSD {

using qc     = CSD::Config;
using Logits = std::span<float>;

class SelfSpecDecDialog : public Dialog {
    enum { VERSION = 1 };

  public:
    SelfSpecDecDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

    virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;
    virtual bool process(std::vector<uint8_t>& embedding_vectors, Dialog::T2ECallback t2eCallback, Dialog::Callback callback) override;
    virtual void reset() override;

    virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override {
        return false;
    }

    virtual bool save(const std::string& name) override;
    virtual bool restore(const std::string& name) override;

  private:
    Sampler& _t_sampler;
    LLGMatcher* _llgmatcher;
    int32_t _vocab;

    std::string _kv_prefix_name{"forecast-prefix"};

    // AR8
    size_t              _draft{1};
    std::vector<size_t> _branches{3};

    size_t _forecast_prefix{16};
    size_t _forecast_token_offset{32000};
    size_t _forecast_token_count{4};

    // Multistream parameters
    int32_t _n_streams;
    float   _p_threshold;

    InputType m_inputType{InputType::UNKNOWN};

    // ===== Decode阶段 Mask 复用：保存首层 draft 采样时的 mask，供下次 AR 采样复用 =====
    std::vector<bool> _saved_mask_for_next_ar;
    bool _has_saved_mask_for_next_ar{false};
    size_t _mask_save_count{0};   // 统计：保存 mask 的次数
    size_t _mask_reuse_count{0};  // 统计：复用 mask 的次数
    // ===== 消融实验：精确计时统计 =====
    float _total_compute_mask_time_us{0.0f};  // compute_mask 累计时间 (微秒)
    size_t _compute_mask_count{0};            // compute_mask 调用次数
    float _prefill_async_saved_time_us{0.0f}; // Prefill 阶段 async 省去的时间 (微秒)
    float _prefill_time_ms{0.0f};             // Prefill 阶段总时间 (毫秒)

    // ===== 创新点3 阶段1: Token Folding (首次Prefill前) =====
    bool _token_folding_phase1_enabled{true};  // 阶段1开关 (硬编码控制)
    size_t _phase1_folded_tokens{0};           // 阶段1折叠的token总数

    // ===== 创新点3 阶段2: Token Folding (Prefill后) =====
    bool _token_folding_phase2_enabled{true};  // 阶段2开关
    size_t _phase2_trigger_count{0};           // 阶段2触发次数 (n >= k)
    size_t _phase2_skip_count{0};              // 阶段2跳过次数 (n < k，进入正常流程)
    size_t _phase2_folded_tokens{0};           // 阶段2折叠的token总数

    // ===== 创新点3 阶段3: Token Folding (Decode中间阶段) =====
    bool _token_folding_phase3_enabled{true};  // 阶段3开关 (硬编码控制)
    size_t _phase3_trigger_count{0};           // 阶段3触发次数 (n >= 2k)
    size_t _phase3_skip_count{0};              // 阶段3跳过次数 (n < 2k，走正常验证)
    size_t _phase3_folded_tokens{0};           // 阶段3折叠的token总数

    // Phase3预采样AR token缓存（用于n < 2k时复用）
    int32_t _phase3_presampled_ar_token{-1};   // 预采样的AR token，-1表示无效
    bool _has_phase3_presampled_token{false};  // 是否有有效的预采样token

    bool processFollowOnGeneration(std::vector<int32_t>& tokens, std::vector<float>& logits, Dialog::Callback callback);
    // Multistream
    bool processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams, std::vector<float>& logits, Dialog::Callback callback);

    /*
        Helper function for combining masks for SSD mulstistream.

        @param  masks           The attention mask to be tiled
        @param  streamIndices   Indices of streams. The tiling count is equal to the size of this vector.
        @param  pastMap         A vector of stream indices for masking all past tokens after the prompt.
        @param  prefixOffset    Offset where KV prefix masking begins in each tile.
        @param  finalMask       A mask that combines all of the independent masks such that
                                they can be executed in the same inference.
    */
    void tileAttentionMask(const std::vector<int32_t>& mask, const std::vector<size_t> streamIndices, const std::vector<size_t>& pastMap, const size_t prefixOffset, std::vector<int32_t>& finalMask);

    std::vector<int32_t> gen_attention_map() const;
    auto                 get_len_flat_sample_tree() const;
    auto                 gen_forecast_tokens(int repeat) const;

    // Sampling and verification
    std::vector<int32_t> build_sample_tree(
            int32_t                     last_token,
            Logits                      logits,
            const std::vector<int32_t>& indices
    );
    std::tuple<std::vector<int32_t>, std::vector<int32_t>> verify_and_select_longest(
            std::span<int32_t> sample_tree,
            Logits             logits,
            bool               use_async_mask = false
    );
    // Sample topK draft tokens from the index parts of logits
    // save_mask_for_next_ar: 如果为true，保存计算的mask供下次AR采样复用
    std::vector<int32_t> sample_to_draft(Logits logits, size_t index, size_t count, bool save_mask_for_next_ar = false) {
        const auto    thislogit = logits.subspan(index * _vocab, _vocab);

        if (_llgmatcher && _matcherEnable) {
            if (_llgmatcher->is_error()) {
                __ERROR("{}\n", _llgmatcher->stop_reason());
                // throw std::runtime_error("Llgmatcher is trapped in error state!");
            }else if (!(_llgmatcher->is_stopped() && _llgmatcher->is_accepting())) {
                // ===== 计时统计 compute_mask =====
                Timer<> mask_timer;
                _llgmatcher->compute_mask();
                float mask_time_us = mask_timer.elapsed_usec();
                _total_compute_mask_time_us += mask_time_us;
                _compute_mask_count++;

                auto& mask = _llgmatcher->get_mask();
                assert(mask.size() == _vocab);

                // ===== Decode阶段 Mask 复用：保存 mask 供下次 AR 采样复用 =====
                if (save_mask_for_next_ar) {
                    _saved_mask_for_next_ar = mask;  // 复制 mask
                    _has_saved_mask_for_next_ar = true;
                    _mask_save_count++;
                }

                for (size_t i = 0; i < _vocab; ++i) {
                    if (!mask[i]) {
                        thislogit[i] = -INFINITY;
                    }
                }
            }
        }

        IndexedLogits logit(thislogit, _t_sampler.rng());
        logit.topK(count);

        if (_llgmatcher && _matcherEnable) {
            if (_llgmatcher->is_error()) {
                __ERROR("{}\n", _llgmatcher->stop_reason());
                // throw std::runtime_error("Llgmatcher is trapped in error state!");
                return logit.indices;
            }
            if (_llgmatcher->is_accepting() && _llgmatcher->is_stopped()) {
                return logit.indices;
            }

            if (_llgmatcher->consume_token(logit.indices[0])) {
                __ERROR("sample_to_draft: LLGMatcher failed to consume token\n");
            }
        }
        return logit.indices;
    }
    // Sample AR token from the index part of logits
    // use_saved_mask: 如果为true，使用保存的mask（来自上一轮build_sample_tree首层draft采样）
    int32_t sample_to_verify(Logits logits, size_t index, bool is_first=false, bool use_precomputed_mask=false, bool use_saved_mask=false) {
        const auto thislogit = logits.subspan(index * _vocab, _vocab);

        if (_llgmatcher && _matcherEnable && !is_first) {
            if (_llgmatcher->is_error()) {
                __ERROR("{}\n", _llgmatcher->stop_reason());
                // throw std::runtime_error("Llgmatcher is trapped in error state!");
            } else if (!(_llgmatcher->is_stopped() && _llgmatcher->is_accepting())) {
                // ===== Decode阶段 Mask 复用：优先使用保存的 mask =====
                if (use_saved_mask && _has_saved_mask_for_next_ar) {
                    // 使用上一轮 build_sample_tree 首层 draft 采样时保存的 mask
                    auto& mask = _saved_mask_for_next_ar;
                    assert(mask.size() == _vocab);
                    for (size_t i = 0; i < _vocab; ++i) {
                        if (!mask[i]) {
                            thislogit[i] = -INFINITY;
                        }
                    }
                    _has_saved_mask_for_next_ar = false;  // 使用后清除标记
                    _mask_reuse_count++;
                } else if (use_precomputed_mask) {
                    // 使用 prefill 阶段 async 预计算的 mask
                    auto& mask = _llgmatcher->get_mask();
                    assert(mask.size() == _vocab);
                    for (size_t i = 0; i < _vocab; ++i) {
                        if (!mask[i]) {
                            thislogit[i] = -INFINITY;
                        }
                    }
                } else {
                    // 正常计算 mask
                    // ===== 计时统计 compute_mask =====
                    Timer<> mask_timer;
                    _llgmatcher->compute_mask();
                    float mask_time_us = mask_timer.elapsed_usec();
                    _total_compute_mask_time_us += mask_time_us;
                    _compute_mask_count++;

                    auto& mask = _llgmatcher->get_mask();
                    assert(mask.size() == _vocab);
                    for (size_t i = 0; i < _vocab; ++i) {
                        if (!mask[i]) {
                            thislogit[i] = -INFINITY;
                        }
                    }
                }
            }
        }

        auto token = _t_sampler.process(thislogit);

        if (_llgmatcher && _matcherEnable) {

            if (_llgmatcher->is_error()) {
                __ERROR("{}\n", _llgmatcher->stop_reason());
                // throw std::runtime_error("Llgmatcher is trapped in error state!");
                return token;
            }

            if (_llgmatcher->is_accepting() && _llgmatcher->is_stopped()) {
                return token;
            }

            if (_llgmatcher->consume_token(token)) {
                __ERROR("sample_to_verify: LLGMatcher failed to consume token\n");
            }
        }
        return token;
    }
};

SelfSpecDecDialog::SelfSpecDecDialog(
        std::shared_ptr<Env> env,
        const std::string&   name,
        const json&          conf
)
    : Dialog(env, name, conf), 
      _t_sampler(*_sampler["primary"]),
      _llgmatcher(nullptr)
{

    auto ssd_version = qc::optional<int>(conf, "ssd-version", 0);
    if (ssd_version > SelfSpecDecDialog::VERSION) __WARN("newer ssd-version in config!");

    _vocab = _ctx->n_vocab();

    _branches = qc::optional(conf, "branches", _branches); //K of topK
    _draft    = _branches.size(); //Mask token(forecast token) num

    _forecast_prefix       = qc::optional(conf, "forecast-prefix", _forecast_prefix);
    _forecast_token_count  = qc::optional(conf, "forecast-token-count", _forecast_token_count);
    _forecast_token_offset = _vocab;

    _kv_prefix_name = qc::optional(conf, "forecast-prefix-name", _kv_prefix_name);

    _n_streams   = qc::optional<int32_t>(conf, "n-streams", 1);
    _p_threshold = qc::optional<float>(conf, "p-threshold", 0.0);

    if (!_engine.contains("primary")) {
        State::fatal("\"primary\" engine not present in config!");
        return;
    }

    // Init llgmatcher
    if (conf.contains("matcher") && conf["matcher"].contains("type")) {
        std::string matcher_type = conf["matcher"]["type"];
        if (matcher_type == "llg") {
            _llgmatcher = dynamic_cast<LLGMatcher*>(_matcher.get());
        }
    }

    // 创新点3 阶段2 Token Folding 开关 (硬编码控制)
    // 设为 true 启用阶段2，设为 false 禁用
    _token_folding_phase2_enabled = true;  // 开启Phase2

    // 创新点3 阶段1 Token Folding 开关 (硬编码控制)
    // 设为 true 启用阶段1（首次Prefill前），设为 false 禁用
    _token_folding_phase1_enabled = false;  // 关闭Phase1

    // 创新点3 阶段3 Token Folding 开关 (硬编码控制)
    // 设为 true 启用阶段3，设为 false 禁用
    _token_folding_phase3_enabled = true;  // 开启Phase3

    //Get Input Type from the engine
    m_inputType = _engine["primary"]->getInputType();
    // Load KV prefix
    Timer  timer;
    size_t n_restored_prefix = _engine["primary"]->restore(_kv_prefix_name, true);
    if (n_restored_prefix != _forecast_prefix) {
        // clang-format off
        throw std::runtime_error( fmt::format( "SSD : Loaded {} KV$ from {} but expected {} KV$",
                    n_restored_prefix, _kv_prefix_name, _forecast_prefix ) );
        // clang-format on
    }
    _n_past = _forecast_prefix;
    _kpis.restore.update(timer.elapsed_usec());
}

auto SelfSpecDecDialog::get_len_flat_sample_tree() const {
    size_t len_flat_sample_tree = 1;
    size_t last_tokens          = 1;
    for (int i = 0; i < _draft; ++i) {
        len_flat_sample_tree += last_tokens * _branches[i];
        last_tokens = last_tokens * _branches[i];
    }
    return len_flat_sample_tree;
}

auto SelfSpecDecDialog::gen_forecast_tokens(int repeat) const {
    std::vector<int32_t> forecast_tokens(_draft, 0);
    std::iota(forecast_tokens.begin(), forecast_tokens.end(), _forecast_token_offset);

    std::vector<int32_t> ret;
    for (auto i = 0; i < repeat; ++i)
        ret.insert(ret.end(), forecast_tokens.begin(), forecast_tokens.end());
    return ret;
}
// todo
std::vector<int32_t> SelfSpecDecDialog::gen_attention_map() const {
    auto                 len_flat_sample_tree = get_len_flat_sample_tree();
    std::vector<int32_t> attention_map(len_flat_sample_tree + len_flat_sample_tree * _draft, -1);

    auto build_verify_tree = [&attention_map,
                              this](auto self, int parent_begin, int parent_end, int level) {
        if (level == _draft) return;
        auto current = parent_end;
        for (auto parent = parent_begin; parent < parent_end; parent += 1) {
            for (auto child = current; child < current + _branches[level]; child += 1)
                attention_map[child] = parent;
            current += _branches[level];
        }
        self(self, parent_end, current, level + 1);
    };

    auto build_forecast_tree = [&attention_map, this](int parent_begin, int parent_end) {
        auto current = parent_end;
        for (auto parent = parent_begin; parent < parent_end; parent += 1) {
            for (auto child = current, current_parent = parent; child < current + _draft;
                 child += 1) {
                attention_map[child] = current_parent;
                current_parent       = child;
            }
            current += _draft;
        }
    };

    build_verify_tree(build_verify_tree, 0, 1, 0);
    build_forecast_tree(0, len_flat_sample_tree);
    return attention_map;
}
// Build draft_tree and return bfs flat tree (todo)
std::vector<int32_t> SelfSpecDecDialog::build_sample_tree(
        int32_t                     last_token,
        Logits                      logits,
        const std::vector<int32_t>& indices
) {
    std::vector<int32_t> tree = {last_token};

    int consume_depth = 0; /* If matcher(FSA) arrives accepting-state, this counter constrained by _consume_done flag will not increase. */
    if (_llgmatcher && _matcherEnable) {
        _llgmatcher->set_consume_done(false);
        if (_llgmatcher->is_stopped() && _llgmatcher->is_accepting()) {
            _llgmatcher->set_consume_done(true);
        }
    }

    for (auto draft = 0, repeat = 1; draft < _draft; ++draft) {
        // ===== Decode阶段 Mask 复用：首层 draft 采样时保存 mask =====
        // 首层 draft 采样时的 matcher 状态与下一轮首个 AR 采样时相同
        // 所以可以保存这个 mask 供下次 AR 采样复用，省去一次 compute_mask()
        // 仅当 async_enabled=true 时启用此优化
        bool async_enabled = (_llgmatcher && _matcherEnable && _llgmatcher->is_async_enabled());
        bool save_mask = (draft == 0) && async_enabled;
        auto samples = sample_to_draft(logits, indices[draft], _branches[draft], save_mask);

        if (_llgmatcher && _matcherEnable) {
            if (!_llgmatcher->get_consume_done()) consume_depth++;
            if (!_llgmatcher->get_consume_done() && _llgmatcher->is_stopped() && _llgmatcher->is_accepting()) {
                _llgmatcher->set_consume_done(true);
            }
        }

        for (auto i = 0; i < repeat; ++i) {
            tree.insert(tree.end(), samples.begin(), samples.end());
        }
        repeat *= _branches[draft];
    }

    if (_llgmatcher && _matcherEnable) {
        if (!_llgmatcher->is_error()) {
            if (consume_depth) {
                if (_llgmatcher->rollback(consume_depth)) {
                    __ERROR("build_sample_tree: matcher rollback failed\n");
                }
            }
        } else __ERROR("{}\n", _llgmatcher->stop_reason());
    }

    return tree;
}

std::tuple<std::vector<int32_t>, std::vector<int32_t>> SelfSpecDecDialog::verify_and_select_longest(
        std::span<int32_t> sample_tree,
        Logits             logits,
        bool               use_async_mask
) {
    // ===== Phase3预采样token复用 =====
    // 如果Phase3已经预采样并consume了AR token，直接使用它
    int32_t first_ar_token;
    if (_has_phase3_presampled_token) {
        first_ar_token = _phase3_presampled_ar_token;
        _has_phase3_presampled_token = false;  // 使用后清除
        _phase3_presampled_ar_token = -1;
        // 注意：matcher状态已经正确（Phase3已consume了这个token）
    } else {
        // ===== Decode阶段 Mask 复用：首个 AR 采样使用保存的 mask =====
        // 上一轮 build_sample_tree 首层 draft 采样时保存的 mask 可以复用
        // 因为两次采样时的 matcher 状态相同（都是 "已 consume 上一个 AR token"）
        // 仅当 async_enabled=true 时启用此优化
        bool async_enabled = (_llgmatcher && _matcherEnable && _llgmatcher->is_async_enabled());
        bool use_saved = async_enabled && _has_saved_mask_for_next_ar;
        first_ar_token = sample_to_verify(logits, 0, false, use_async_mask, use_saved);
    }
    std::vector<std::vector<int32_t>> accepted_all = {{first_ar_token}};
    std::vector<std::vector<int32_t>> node_ids_all = {{0}};

    std::vector<int32_t> draft_offset(_draft, 0);//the start index in sample tree of different level
    draft_offset[0] = 1;
    for (int32_t i = 1, draft_count = _branches[0]; i < _draft; ++i) {
        draft_offset[i] = draft_offset[i - 1] + draft_count;
        draft_count     = draft_count * _branches[i];
    }

    size_t longest = 0, longest_size = 1;
    int hit_length = 0;
    auto   verify_recursive = [&](auto                 self,
                                std::vector<int32_t> accepted,
                                std::vector<int32_t> node_ids,
                                int                  draft,
                                int                  offset_in_draft/* the offset in one level of sample tree */) -> void {
        auto target      = accepted.back();
        auto branch_base = draft_offset[draft] + offset_in_draft;
        for (auto branch = 0; branch < _branches[draft]; ++branch) {
            if (hit_length == _draft) break;
            auto ndx_node = branch_base + branch;
            if (!_ctx->is_eos(target) && target == sample_tree[ndx_node]) {
                hit_length++;
                auto sample_accepted = sample_to_verify(logits, ndx_node, false);
                accepted_all.push_back(accepted);
                accepted_all.back().push_back(sample_accepted);
                node_ids_all.push_back(node_ids);
                node_ids_all.back().push_back(ndx_node);
                if (node_ids_all.back().size() > longest_size) {
                    longest      = node_ids_all.size() - 1;
                    longest_size = node_ids_all.back().size();
                }
                if (draft + 1 < _draft)
                    self(self,
                         accepted_all.back(),
                         node_ids_all.back(),
                         draft + 1,
                         (offset_in_draft + branch) * _branches[draft + 1]);
            }
        }
    };
    verify_recursive(verify_recursive, accepted_all.back(), node_ids_all.back(), 0, 0);
    return {accepted_all[longest], node_ids_all[longest]};
}

void SelfSpecDecDialog::tileAttentionMask(const std::vector<int32_t>& mask, const std::vector<size_t> streamIndices, const std::vector<size_t>& pastMap, const size_t prefixOffset, std::vector<int32_t>& tiledMask) {

    const size_t sampleTreeLen = get_len_flat_sample_tree();
    const size_t pastMapLen    = pastMap.size();
    const int posVal = 1, negVal = 0;

    const size_t maskSize = mask.size();
    const size_t numTokens = maskSize * streamIndices.size();

    const size_t rowLength = _n_past + numTokens;
    tiledMask.resize(numTokens * rowLength);

    for (int maskIdx = 0; maskIdx < streamIndices.size(); maskIdx++) {
        // Number of rows to skip to reach the current tile.
        const size_t tileOffset = maskIdx * maskSize;
        int32_t* const tileStart = &tiledMask[tileOffset*rowLength + tileOffset + _n_past];
        for (int i = 0; i < maskSize; i++) {
            // Pointer to the start of row i of the current mask
            int32_t* rowPtr = &tiledMask[(tileOffset + i)*rowLength];
            // Skip kv-prefix attention for rows without speculative tokens.
            const int prefixFillVal = (i < prefixOffset) ? negVal : posVal;
            std::fill_n(rowPtr, _forecast_prefix, prefixFillVal);
            rowPtr += _forecast_prefix;
            // Always attend to prompt.
            std::fill_n(rowPtr, _n_prompt, posVal);
            rowPtr += _n_prompt;

            // Fill in the past valid tokens for this stream.
            for (const size_t& pastIdx : pastMap) {
                *rowPtr = (pastIdx == streamIndices[maskIdx]) ? posVal : negVal;
                rowPtr++;
            }

            // Clear the rest of the row. It will mostly consist of 0's.
            std::fill_n(rowPtr, rowLength - _n_prompt - _forecast_prefix - pastMapLen, negVal);
            // Move to the correct tile.
            rowPtr += tileOffset;
            // Translate the mask.
            const auto tokenId = mask[i];
            if (tokenId > -1) {
                std::copy_n(tileStart + (tokenId * rowLength), tokenId + 1, rowPtr);
            }
            // Always attend to self.
            rowPtr[i] = posVal;
        }
    }
}

// Takes a vector of tokens and produces a vector of embeddings via the provided T2E callback.
static inline void convertTokensToEmbeddings(std::vector<int32_t>& tokens,
                                             std::vector<uint8_t>& embeddings,
                                             size_t embeddingBufferSize,
                                             Dialog::T2ECallback t2eCallback) {
    for(auto &token : tokens){
        std::vector<uint8_t> embedding(embeddingBufferSize,0);
        t2eCallback(token, embedding.data(), embeddingBufferSize);
        embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());
    }
}

bool SelfSpecDecDialog::processFollowOnGeneration(std::vector<int32_t>& tokens, std::vector<float>& logits, Dialog::Callback callback){

    // Handles the printing of the subsequent generated tokens
    bool          keep_generating = true;
    const size_t  context         = _ctx->n_ctx();

    std::vector<int32_t> decode_buf(
            1, 0
    ); // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
    auto decode_token = [&](int32_t t) {
        if (!keep_generating) return;
        // Decode new token.
        // Return true to continue generation, and false otherwise
        decode_buf[0] = _last_tok = t;
        ++_n_generated;
        if (_ctx->is_eos(t)) {
            keep_generating = false;
            callback("", Sentence::END);
        } else {
            keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
        }
        return;
    };
    // set decode_buf from prompt processing
    decode_buf[0] = _last_tok;

    auto& engine = *_engine["primary"];

    auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool>& selected) {
        if (!engine.updateKV(past, selected))
            return Dialog::abort("context size exceeded", callback);
        return true;
    };


    // prepare the next inference
    std::vector<int32_t> indices(_draft, 0);
    std::iota(indices.begin(), indices.end(), 1);
    // step1.5: sample from logits and build draft tree([AR]+[draft]*)
    tokens = build_sample_tree(sample_to_verify(std::span{logits.data(),logits.size()}, 0, false), std::span{logits.data(),logits.size()}, indices);
    decode_token(tokens[0]); // judge if is eos and call_back

    // Prepare constant options for next inferences
    // step2: construct forecast(Mask) tokens and set attention mask
    const auto len_flat_sample_tree = get_len_flat_sample_tree();
    const auto forecast_tokens      = gen_forecast_tokens(len_flat_sample_tree);
    const auto attention_map        = gen_attention_map(); // set attention mask

    engine.set({{"kv-prefix-offset", len_flat_sample_tree}});

    std::vector<int32_t> accepted_counts(_draft + 1, 0);
    std::vector<bool>    selected(attention_map.size(), false);

    // ===== Timing statistics =====
    Timer<> total_timer;
    float total_npu_time_ms = 0.0f;
    float total_verify_time_ms = 0.0f;
    float total_iteration_time_ms = 0.0f;
    int iteration_count = 0;
    bool async_enabled = (_llgmatcher && _matcherEnable && _llgmatcher->is_async_enabled());

    while (!State::canceled() && keep_generating) {
        Timer<> iter_timer;

        // Append forecast tokens
        // step3.1: insert forecast(Mask) tokens at the end(tokens=[AR]+[Draft]+[Forecast])
        tokens.insert(tokens.end(), forecast_tokens.begin(), forecast_tokens.end());

        if (_n_past + tokens.size() > _ctx->size()) {
            __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
            callback("", Sentence::END);
            break;
        }

        Timer<> npu_timer;
        size_t n_tok_t = 0;
        // step3.2: infer tokens([AR]+[Draft]+[Forcast]) and get logits
        // Bifurcate based on embedding as input or token as input
        if (m_inputType == InputType::TOKENS)
            n_tok_t = engine.process(tokens, attention_map, logits, true /* all logits */);
        else if (m_inputType == InputType::EMBEDDINGS) {
            // Convert tokens to embedding for the processing in the engine.
            auto embedBufSize = engine.getEmbeddingBufferSize();
            std::vector<uint8_t> embedding;
            for(auto &token: tokens){
                std::vector<uint8_t> curTokenEmbedding(embedBufSize,0);
                m_t2eCallback(token, curTokenEmbedding.data(), embedBufSize);
                embedding.insert(embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
            }
            n_tok_t = engine.process(embedding, attention_map, logits, true /* all logits */);
        } else {
            return Dialog::abort("No valid Input Type is used", callback);
        }
        float npu_time_ms = npu_timer.elapsed_msec();
        total_npu_time_ms += npu_time_ms;

        if (n_tok_t != tokens.size()) return Dialog::abort("engine processing failed", callback);

        // ===== 创新点3 阶段3: Token Folding (Decode中间阶段) =====
        // 触发条件: 在NPU推理后、验证draft tree前
        // 设计: 采样AR token → consume → 检查ff_tokens → 根据n vs 2k决定流程
        bool phase3_enabled = (_llgmatcher && _matcherEnable && _token_folding_phase3_enabled &&
                               _llgmatcher->is_token_folding_enabled());
        size_t k = _draft;       // branches数组长度
        size_t threshold_2k = 2 * k;

        if (phase3_enabled) {
            // 1. 从logits[0]采样AR token (需要apply mask)
            // 注意: sample_to_verify会自动consume token
            int32_t ar_token = sample_to_verify(std::span{logits.data(), logits.size()}, 0, false);

            // 2. 检查ff_tokens (matcher已经consume了ar_token)
            auto ff_tokens_vec = _llgmatcher->compute_ff_tokens();
            size_t n = ff_tokens_vec.size();

            fprintf(stderr, "[TF-Phase3] After AR sampling, checking ff_tokens: n=%zu, 2k=%zu\n", n, threshold_2k);

            if (n >= threshold_2k) {
                // ===== n >= 2k: 跳过验证和draft采样 =====
                _phase3_trigger_count++;
                fprintf(stderr, "[TF-Phase3] n >= 2k, skipping verification and draft sampling\n");

                // 3.1 输出AR token
                decode_token(ar_token);
                if (!keep_generating) continue;

                // 3.2 输出并consume所有ff_tokens
                for (auto ff_tok : ff_tokens_vec) {
                    if (_ctx->is_eos(ff_tok)) {
                        keep_generating = false;
                        callback("", Sentence::END);
                        break;
                    }
                    _llgmatcher->consume_token(ff_tok);
                    decode_token(static_cast<int32_t>(ff_tok));
                    if (!keep_generating) break;
                    _phase3_folded_tokens++;
                }
                if (!keep_generating) continue;

                // 3.3 更新KV cache: 只保留AR token位置
                // tokens当前是 [AR_old] + [draft tree] + [forecast]
                // 我们只需要保留index 0 (AR token的位置)
                selected.resize(1);
                selected[0] = true;
                _n_past += 1;
                update_kv(_n_past, selected);

                // 3.4 构建新推理序列: [AR] + [ff_tokens] + [forecast]
                std::vector<int32_t> new_tokens;
                new_tokens.push_back(ar_token);
                for (auto ff_tok : ff_tokens_vec) {
                    new_tokens.push_back(static_cast<int32_t>(ff_tok));
                }
                // 添加forecast tokens
                for (size_t i = 0; i < _draft; ++i) {
                    new_tokens.push_back(_forecast_token_offset + i);
                }

                // 3.5 构建attention_map
                std::vector<int32_t> new_attn_map(new_tokens.size());
                std::iota(new_attn_map.begin(), new_attn_map.end(), -1);

                // 3.6 设置kv-prefix-offset
                size_t prefix_offset = 1 + ff_tokens_vec.size();
                engine.set({{"kv-prefix-offset", prefix_offset}});

                // 3.7 检查context限制
                if (_n_past + new_tokens.size() > _ctx->size()) {
                    __WARN("Context limit exceeded ({} + {} > {})", _n_past, new_tokens.size(), _ctx->size());
                    callback("", Sentence::END);
                    keep_generating = false;
                    continue;
                }

                // 3.8 推理
                if (m_inputType == InputType::TOKENS) {
                    n_tok_t = engine.process(new_tokens, new_attn_map, logits, true);
                } else if (m_inputType == InputType::EMBEDDINGS) {
                    auto embedBufSize = engine.getEmbeddingBufferSize();
                    std::vector<uint8_t> new_embedding;
                    for (auto& token : new_tokens) {
                        std::vector<uint8_t> curTokenEmbedding(embedBufSize, 0);
                        m_t2eCallback(token, curTokenEmbedding.data(), embedBufSize);
                        new_embedding.insert(new_embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
                    }
                    n_tok_t = engine.process(new_embedding, new_attn_map, logits, true);
                }
                if (n_tok_t != new_tokens.size()) {
                    return Dialog::abort("Phase3 inference failed", callback);
                }

                // 3.9 更新KV cache: 保留 ff_tokens (AR已经在前面加过了)
                std::vector<bool> new_selected(new_tokens.size(), false);
                for (size_t i = 0; i < 1 + ff_tokens_vec.size(); ++i) {
                    new_selected[i] = true;
                }
                _n_past += ff_tokens_vec.size();
                update_kv(_n_past, new_selected);

                // 3.10 从最后一个ff_token位置的logits采样新AR token
                // new_tokens = [AR, ff0, ff1, ..., ff(n-1), forecast0, ...]
                // 最后一个ff_token是ff(n-1)，对应logits索引 = ff_tokens_vec.size()
                size_t last_ff_logits_idx = ff_tokens_vec.size();
                int32_t new_ar_token = sample_to_verify(std::span{logits.data(), logits.size()}, last_ff_logits_idx, false);

                // 3.11 构建新的draft tree
                // logits中forecast tokens的位置从 (1 + ff_tokens_vec.size()) 开始
                size_t forecast_base = 1 + ff_tokens_vec.size();
                std::iota(indices.begin(), indices.end(), forecast_base);
                tokens = build_sample_tree(new_ar_token, std::span{logits.data(), logits.size()}, indices);

                // 恢复kv-prefix-offset为正常值
                engine.set({{"kv-prefix-offset", len_flat_sample_tree}});

                fprintf(stderr, "[TF-Phase3] Folded %zu tokens, built new draft tree, continuing main loop\n", ff_tokens_vec.size());

                // 继续主while循环的下一轮迭代
                float iter_time_ms = iter_timer.elapsed_msec();
                total_iteration_time_ms += iter_time_ms;
                iteration_count++;
                continue;

            } else {
                // ===== n < 2k: 退出阶段3，走正常验证流程 =====
                _phase3_skip_count++;
                fprintf(stderr, "[TF-Phase3] n < 2k, entering normal verify flow\n");

                // 保存预采样的AR token，供verify_and_select_longest复用
                // 不需要rollback，因为matcher状态已经正确（ar_token已被consume）
                _phase3_presampled_ar_token = ar_token;
                _has_phase3_presampled_token = true;
            }
        }

        // ===== 正常流程: 验证draft tree =====
        // step3.3: verify draft tokens and return the longest hit path
        Timer<> verify_timer;
        auto [accepted_tokens, accepted_ids] = verify_and_select_longest(std::span{tokens.data(),tokens.size()},
                                                                         std::span{logits.data(),logits.size()},
                                                                         false);
        float verify_time_ms = verify_timer.elapsed_msec();
        total_verify_time_ms += verify_time_ms;

        // Commit accepted tokens to kv-caches
        selected.resize(accepted_ids.back() + 1); // trim away rejected tokens
        std::fill(selected.begin(), selected.end(), false);
        for (auto id : accepted_ids)
            selected[id] = true;
        accepted_counts[accepted_tokens.size() - 1] += 1;

        for(uint32_t idx= 0;idx<accepted_tokens.size();idx++){
          engine.updateTokenCheckpoint(accepted_tokens[idx],_n_past+idx);
        }
        _n_past += accepted_tokens.size();
        update_kv(_n_past, selected);

        // Decode tokens
        std::for_each(accepted_tokens.begin(), accepted_tokens.end(), decode_token);

        // Prepare new tokens
        auto next_draft_offset = len_flat_sample_tree + accepted_ids.back() * _draft;
        std::iota(indices.begin(), indices.end(), next_draft_offset);
        // step3.4: build next draft tree and update tokens([AR]+[Draft])
        tokens = build_sample_tree(accepted_tokens.back(), std::span{logits.data(),logits.size()}, indices);

        float iter_time_ms = iter_timer.elapsed_msec();
        total_iteration_time_ms += iter_time_ms;
        iteration_count++;
    }

    State::busy(false);

    float total_time_ms = total_timer.elapsed_msec();
    auto total_iteration = std::accumulate(accepted_counts.begin(), accepted_counts.end(), 0);
    auto accept_rate =
            float(_n_generated - 1) / total_iteration; // -1: exclude first generated token
    __KPIS("SSD{{draft:{}, branch:{}, greedy:{}}}: accepted counts: {}, accept rate = {} tokens/iteration",
           _draft,
           _branches,
           _t_sampler.greedy(),
           accepted_counts,
           accept_rate);

    // ===== Output timing statistics =====
    fprintf(stderr, "===== CPU-NPU Overlap Timing (async_enabled=%d) =====\n", async_enabled ? 1 : 0);
    fprintf(stderr, "Prefill time: %.2f ms\n", _prefill_time_ms);
    fprintf(stderr, "Decode time: %.2f ms\n", total_time_ms);
    fprintf(stderr, "Total (Prefill+Decode) time: %.2f ms\n", _prefill_time_ms + total_time_ms);
    fprintf(stderr, "Total tokens generated: %zu\n", _n_generated);
    fprintf(stderr, "Decode Throughput: %.2f tokens/sec\n", _n_generated * 1000.0f / total_time_ms);
    fprintf(stderr, "End-to-End Throughput: %.2f tokens/sec\n", _n_generated * 1000.0f / (_prefill_time_ms + total_time_ms));
    fprintf(stderr, "Iterations: %d\n", iteration_count);
    fprintf(stderr, "Avg NPU inference time: %.3f ms/iter\n", total_npu_time_ms / iteration_count);
    fprintf(stderr, "Avg verify time: %.3f ms/iter\n", total_verify_time_ms / iteration_count);
    fprintf(stderr, "Avg iteration time: %.3f ms/iter\n", total_iteration_time_ms / iteration_count);

    // ===== 创新点1 性能收益统计 =====
    fprintf(stderr, "----- Innovation 1: CPU-NPU Parallel Performance -----\n");
    // Prefill 阶段
    fprintf(stderr, "[Prefill] Async mask saved: %.2f us (%.3f ms)\n",
            _prefill_async_saved_time_us, _prefill_async_saved_time_us / 1000.0f);
    // Decode 阶段
    float avg_compute_mask_time_us = (_compute_mask_count > 0) ?
            (_total_compute_mask_time_us / _compute_mask_count) : 0.0f;
    float decode_mask_reuse_saved_us = avg_compute_mask_time_us * _mask_reuse_count;
    fprintf(stderr, "[Decode] compute_mask: count=%zu, total=%.2fus, avg=%.2fus\n",
            _compute_mask_count, _total_compute_mask_time_us, avg_compute_mask_time_us);
    fprintf(stderr, "[Decode] Mask reuse: saved=%zu, reused=%zu, saved_time=%.2fus (%.3fms)\n",
            _mask_save_count, _mask_reuse_count, decode_mask_reuse_saved_us, decode_mask_reuse_saved_us / 1000.0f);
    // 总收益
    float total_saved_us = _prefill_async_saved_time_us + decode_mask_reuse_saved_us;
    float total_time_us = total_time_ms * 1000.0f;
    float improvement_pct = (total_saved_us / (total_time_us + total_saved_us)) * 100.0f;
    fprintf(stderr, "[Total] Saved time: %.2f us (%.3f ms), Improvement: %.2f%%\n",
            total_saved_us, total_saved_us / 1000.0f, improvement_pct);

    // ===== 创新点3 Token Folding 统计 =====
    fprintf(stderr, "----- Innovation 3: Token Folding Performance -----\n");
    fprintf(stderr, "[Phase1] Folded_tokens=%zu (before prefill)\n", _phase1_folded_tokens);
    fprintf(stderr, "[Phase2] Triggers=%zu, Skips=%zu, Folded_tokens=%zu\n",
            _phase2_trigger_count, _phase2_skip_count, _phase2_folded_tokens);
    fprintf(stderr, "[Phase3] Triggers=%zu, Skips=%zu, Folded_tokens=%zu\n",
            _phase3_trigger_count, _phase3_skip_count, _phase3_folded_tokens);
    size_t total_folded = _phase1_folded_tokens + _phase2_folded_tokens + _phase3_folded_tokens;
    fprintf(stderr, "[Total] Folded_tokens=%zu (Phase1: %zu + Phase2: %zu + Phase3: %zu)\n",
            total_folded, _phase1_folded_tokens, _phase2_folded_tokens, _phase3_folded_tokens);

    fprintf(stderr, "==============================================================\n");

    return true;
}

// Multistream AR generation
bool SelfSpecDecDialog::processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams, std::vector<float>& logits, Dialog::Callback callback) {

    auto& sampler = *_sampler["primary"];
    auto& engine  = *_engine["primary"];

    auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool>& selected) {
        if (!engine.updateKV(past, selected))
            return Dialog::abort("context size exceeded", callback);
        return true;
    };

    std::vector<size_t> streamIndices(streams.size());
    std::vector<size_t> past_map(streams.size());

    std::iota(streamIndices.begin(), streamIndices.end(), 0);
    // Since the first inference is done separately, it is
    // expected that each stream already has 1 valid AR token.
    std::iota(past_map.begin(), past_map.end(), 0);
    // Add generated token count from first inference.
    _n_generated += streams.size();

    bool keep_generating = true;
    const size_t context = _ctx->n_ctx();

    if (streams.size() == 0) {
        callback("\n", Sentence::END);
        return true;
    }

    // Prepare constant options for next inferences
    const auto len_flat_sample_tree = get_len_flat_sample_tree();
    const auto forecast_tokens      = gen_forecast_tokens(len_flat_sample_tree);
    const auto attention_map        = gen_attention_map();

    std::vector<std::vector<int32_t>> draftStreams(streams.size());

    std::vector<int32_t> accepted_counts(_draft + 1, 0);
    std::vector<int32_t> multi_attn_mask;

    for (int i = 0; i < streams.size(); i++) {
        // prepare the next inference
        std::vector<int32_t> indices(_draft, 0);
        std::iota(indices.begin(), indices.end(), 1);
        draftStreams[i] = build_sample_tree(sample_to_verify(std::span{logits.data(),logits.size()}, i*(1+_draft)), std::span{logits.data(),logits.size()}, indices);
        streams[i].push_back(draftStreams[i][0]);
    }

    engine.set({{"kv-prefix-offset", len_flat_sample_tree}});

    State::busy(true);
    while (true) {
        if (State::canceled()) break;

        // If this exceeds context length, truncate all streams and return
        if (_n_past + streamIndices.size() > _ctx->size()) {
            for (auto stream : streamIndices)
                callback(_tokenizer->decode(streams[stream]) + "\n", Sentence::CONTINUE);
            break;
        }

        // Accumulate input tokens from all streams
        std::vector<int32_t> multi_tokens;
        for (auto streamIdx : streamIndices) {
            multi_tokens.insert(multi_tokens.end(), draftStreams[streamIdx].begin(), draftStreams[streamIdx].end());
            multi_tokens.insert(multi_tokens.end(), forecast_tokens.begin(), forecast_tokens.end());
        }

        if (_n_past + multi_tokens.size() > _ctx->size()) {
            __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
            callback("", Sentence::END);
            break;
        }

        tileAttentionMask(attention_map, streamIndices, past_map, len_flat_sample_tree, multi_attn_mask);

        size_t n_tok_t = 0;

        if (m_inputType == InputType::TOKENS) {
            // Process input tokens for all streams in one batch
            n_tok_t = engine.process(multi_tokens, multi_attn_mask, logits, true);
        } else if (m_inputType == InputType::EMBEDDINGS) {
            // Accumulate input embeddings from all streams
            auto embedBufSize = engine.getEmbeddingBufferSize();
            std::vector<uint8_t> multi_embeddings;

            convertTokensToEmbeddings(multi_tokens, multi_embeddings, embedBufSize, m_t2eCallback);

            // Process input tokens for all streams in one batch
            n_tok_t = engine.process(multi_embeddings, multi_attn_mask, logits, true);
        }
        if (n_tok_t != multi_tokens.size()) return Dialog::abort("engine processing failed", callback);

        std::vector<bool> all_selected;

        // Process all logits independently
        std::span<float> logit_span   = std::span{logits.data(),logits.size()};
        std::span<int32_t> token_span = std::span{multi_tokens.data(), multi_tokens.size()};
        for (int i = 0; i < streamIndices.size(); i++) {
            const size_t streamIdx = streamIndices[i];
            std::vector<int32_t>& stream = streams[streamIdx];

            const size_t tileStride = draftStreams[streamIdx].size() + forecast_tokens.size();

            std::span<float> tiled_logits = logit_span.subspan(i * tileStride * _vocab, _vocab);

            // Accept tokens
            auto [accepted_tokens, accepted_ids] = verify_and_select_longest(token_span.subspan(i * tileStride, tileStride),
                                                                            tiled_logits);

            // Commit accepted tokens to kv-caches
            std::vector<bool> selected(tileStride, false);
            for (auto id : accepted_ids) {
                selected[id] = true;
                past_map.push_back(streamIdx);
            }
            all_selected.insert(all_selected.end(), selected.begin(), selected.end());
            accepted_counts[accepted_tokens.size() - 1] += 1;
            _n_past += accepted_tokens.size();

            // Decode tokens
            stream.insert(stream.end(), accepted_tokens.begin(), accepted_tokens.end());
            _n_generated += accepted_tokens.size();

            // Prepare new tokens
            std::vector<int32_t> indices(_draft, 0);
            auto next_draft_offset = len_flat_sample_tree + accepted_ids.back() * _draft;
            std::iota(indices.begin(), indices.end(), next_draft_offset);
            draftStreams[streamIdx] = build_sample_tree(accepted_tokens.back(), tiled_logits, indices);
        }

        update_kv(_n_past, all_selected);
        for (auto it = streamIndices.begin(); it != streamIndices.end();) {
            int32_t stream = *it;
            if (_ctx->is_eos(streams[stream].back())) {
                callback(_tokenizer->decode(streams[stream]) + "\n", Sentence::CONTINUE);
                it = streamIndices.erase(it);
            } else {
                ++it;
            }
        }

        if (streamIndices.size() == 0) break;
    }
    callback("\n", Sentence::END);

    State::busy(false);

    auto total_iteration = std::accumulate(accepted_counts.begin(), accepted_counts.end(), 0);
    auto accept_rate =
            float(_n_generated - 1) / total_iteration; // -1: exclude first generated token
    __KPIS("SSD{{draft:{}, branch:{}, greedy:{}}}: accepted counts: {}, accept rate = {} tokens/iteration",
           _draft,
           _branches,
           _t_sampler.greedy(),
           accepted_counts,
           accept_rate);

    return true;
}

// Handle prompt processing and generation will be done processFollowOnGeneration
// Pass t2e callback using setter and remove as an argument. call setter from the base query function of dialog

bool SelfSpecDecDialog::process(std::vector<uint8_t>& embedding,
                                T2ECallback             t2eCallback,
                                Dialog::Callback        callback ){

    // Check for prev failures and bail out early
    if (State::failed()) return false;

    if(m_inputType != InputType::EMBEDDINGS) {
        __ERROR("Input type for model is not embeddings.");
        return false;
    }

    Timer start;
    State::clear();

    std::vector<float> logits;
    auto&              engine = *_engine["primary"];

    auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool>& selected) {
        if (!engine.updateKV(past, selected))
            return Dialog::abort("context size exceeded", callback);
        return true;
    };

    // Store the t2e callback for reference during follow-on generation.
    m_t2eCallback = t2eCallback;

    auto embedBufSize = engine.getEmbeddingBufferSize();

    {
        std::vector<uint8_t> eosEmbedding(embedBufSize, 0.0);
        if (m_t2eCallback) {
            m_t2eCallback(_ctx->eos(), eosEmbedding.data(), embedBufSize);
        }
        if (!engine.cacheEosEmbedding(eosEmbedding)) {
            __DEBUG("Failed to set the eos token embedding.");
            return false;
        }
    }

    using FF = Engine::Feature::Flags;
    if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

    _env->logger().post(Logger::KPIS, kpis().dump(" "));
    start.reset();

    engine.set({{"kv-prefix-skip", _forecast_prefix}});

    std::vector<int32_t> tokens(1,0);

    // Process prompt
    // get number of tokens in the input
    size_t curTokensCount = embedding.size()/embedBufSize;

    if(curTokensCount * embedBufSize != embedding.size()){
        size_t expectedLength = (curTokensCount + (embedding.size()%embedBufSize != 0))*embedBufSize;
        __DEBUG("Input is wrong expected {} and found {}.", expectedLength, embedding.size());
        return Dialog::abort("Input is not an multiple for the embedding Length", callback);
    }

    _n_prompt += curTokensCount;

    std::vector<int32_t> attention_map(curTokensCount);
    std::iota(attention_map.begin(), attention_map.end(), -1);
    // kv-prefix is the prompt-tuning tokens' KVcache
    engine.set({{"kv-prefix-offset", curTokensCount}}); // Do not attend prefix

    // ===== 创新点3 阶段1: Token Folding (首次Prefill前) =====
    // 在prefill前检查确定性tokens，如果有则拼接到embedding一起prefill
    bool phase1_enabled = (_llgmatcher && _matcherEnable && _token_folding_phase1_enabled &&
                           _llgmatcher->is_token_folding_enabled());
    std::vector<uint32_t> phase1_ff_tokens;

    if (phase1_enabled) {
        // 调用compute_ff_tokens检查是否有确定性tokens
        phase1_ff_tokens = _llgmatcher->compute_ff_tokens();
        size_t n = phase1_ff_tokens.size();

        fprintf(stderr, "[TF-Phase1] Before prefill, checking ff_tokens: n=%zu\n", n);

        if (n > 0) {
            fprintf(stderr, "[TF-Phase1] Found %zu deterministic tokens, appending to embedding\n", n);

            // 将ff_tokens转换为embedding并拼接
            for (auto ff_tok : phase1_ff_tokens) {
                std::vector<uint8_t> ff_embedding(embedBufSize, 0);
                m_t2eCallback(static_cast<int32_t>(ff_tok), ff_embedding.data(), embedBufSize);
                embedding.insert(embedding.end(), ff_embedding.begin(), ff_embedding.end());

                // 同步matcher的consume
                _llgmatcher->consume_token(ff_tok);
                _phase1_folded_tokens++;
            }

            // 更新curTokensCount和attention_map
            curTokensCount += n;
            attention_map.resize(curTokensCount);
            std::iota(attention_map.begin(), attention_map.end(), -1);

            // 更新kv-prefix-offset
            engine.set({{"kv-prefix-offset", curTokensCount}});

            // 更新_n_prompt
            _n_prompt += n;

            fprintf(stderr, "[TF-Phase1] Embedding now has %zu tokens (original + %zu ff_tokens)\n",
                    curTokensCount, n);
        }
    }

    if (_n_past + curTokensCount > _ctx->size()) {
        __WARN("Context limit exceeded ({} + {} > {})", _n_past, curTokensCount, _ctx->size());
        callback("", Sentence::END);
        return true;
    }

    // ===== CPU-NPU Parallel: 在prefill前启动async mask计算 =====
    bool prefill_async_enabled = (_llgmatcher && _matcherEnable && _llgmatcher->is_async_enabled());
    fprintf(stderr, "[Prefill-Debug] llgmatcher=%p, matcherEnable=%d, async_enabled=%d, prefill_async_enabled=%d\n",
            (void*)_llgmatcher, _matcherEnable ? 1 : 0,
            (_llgmatcher ? _llgmatcher->is_async_enabled() : false) ? 1 : 0,
            prefill_async_enabled ? 1 : 0);
    Timer<> prefill_mask_timer;
    if (prefill_async_enabled) {
        _llgmatcher->start_prefill_async_mask();
        fprintf(stderr, "[Prefill-Async-Emb] Started async mask computation before prefill\n");
    }

    Timer<> prefill_npu_timer;
    // step1.1: first inference without draft and mask tokens.
    if (!engine.process(embedding, attention_map, logits, false))
        return Dialog::abort("engine prompt processing failed", callback); // Change this message also to some generic message.
    float prefill_npu_time_ms = prefill_npu_timer.elapsed_msec();
    _prefill_time_ms = prefill_npu_time_ms; // 记录Prefill时间用于端到端吞吐量计算

    // ===== CPU-NPU Parallel: 等待prefill mask结果 =====
    bool use_prefill_mask = false;
    if (prefill_async_enabled) {
        use_prefill_mask = _llgmatcher->wait_prefill_mask_result();
        float async_mask_time_us = _llgmatcher->get_async_mask_time_us();
        float prefill_npu_time_us = prefill_npu_time_ms * 1000.0f;
        // 省去的时间 = min(mask_time, npu_time)，如果mask被NPU时间cover住
        // 如果 mask_time < npu_time，mask 被完全 cover，省去 mask_time
        // 如果 mask_time > npu_time，有部分等待，省去 npu_time
        _prefill_async_saved_time_us = std::min(async_mask_time_us, prefill_npu_time_us);
        fprintf(stderr, "[Prefill-Async] mask_time=%.2fus, npu_time=%.2fus, saved=%.2fus (%.2f%%)\n",
                async_mask_time_us, prefill_npu_time_us, _prefill_async_saved_time_us,
                _prefill_async_saved_time_us / prefill_npu_time_us * 100.0f);
    }

    _n_past += curTokensCount;
    update_kv(_n_past, {});

    bool status = true;
    if (_n_streams <= 1) {
        // ===== 创新点3 阶段1: 输出phase1的确定性tokens =====
        // 如果阶段1收集了确定性tokens，在采样AR token前先输出它们
        if (phase1_enabled && !phase1_ff_tokens.empty()) {
            fprintf(stderr, "[TF-Phase1] Outputting %zu folded tokens after prefill\n", phase1_ff_tokens.size());

            bool is_first_output = true;
            for (auto ff_tok : phase1_ff_tokens) {
                if (_ctx->is_eos(ff_tok)) {
                    callback("", Sentence::END);
                    return true;
                }
                std::vector<int32_t> ff_vec = {static_cast<int32_t>(ff_tok)};
                if (is_first_output) {
                    if (!callback(_tokenizer->decode(ff_vec), Sentence::BEGIN)) return true;
                    is_first_output = false;
                    _n_generated++;

                    // 阶段1输出第一个token后，标记TTFT
                    if (!m_t2eCallback) {
                        callback("", Sentence::END);
                        return true;
                    }
                    _kpis.prompt.update(start.elapsed_usec());
                    start.reset();
                    State::busy(true);
                } else {
                    if (!callback(_tokenizer->decode(ff_vec), Sentence::CONTINUE)) return true;
                    _n_generated++;
                }
            }

            // 从最后一个ff_token的logits位置采样AR token
            // embedding序列是: [原始prompt tokens] + [ff_tokens]
            // logits对应索引: 原始prompt最后一个token对应 (原始curTokensCount - 1)
            // 但是现在curTokensCount已经更新了，所以需要用 (curTokensCount - 1) = 最后一个ff_token
            // 但prefill只返回最后一个token的logits，所以logits index是0
            size_t ar_logits_idx = 0;  // prefill后logits只有一个位置
            tokens[0] = sample_to_verify(std::span{logits.data(),logits.size()}, ar_logits_idx, false, use_prefill_mask);

            fprintf(stderr, "[TF-Phase1] Sampled AR token from logits[%zu] after ff_tokens\n", ar_logits_idx);
            fprintf(stderr, "[TF-Phase1] Phase1 complete: folded_tokens=%zu\n", _phase1_folded_tokens);
        } else {
            // step1.2: sample AR token，使用预计算的mask
            tokens[0] = sample_to_verify(std::span{logits.data(),logits.size()}, 0, false, use_prefill_mask);
        }

        // === 创新点3 阶段2: Token Folding (Prefill后) ===
        bool phase2_enabled = (_llgmatcher && _matcherEnable && _token_folding_phase2_enabled &&
                               _llgmatcher->is_token_folding_enabled());
        size_t k = _draft;  // branches数组长度 = draft tree深度

        if (phase2_enabled) {
            // 如果阶段1已经输出了tokens，则is_first_output应该为false
            bool is_first_output = !(phase1_enabled && !phase1_ff_tokens.empty());

            while (true) {
                // 1. 检查EOS
                _last_tok = tokens[0];
                if (_ctx->is_eos(_last_tok)) {
                    callback("", Sentence::END);
                    return true;
                }

                // 2. 输出当前AR token
                if (is_first_output) {
                    if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) return true;
                    is_first_output = false;
                    _n_generated++;

                    // 3. 检查t2eCallback并标记TTFT
                    if (!m_t2eCallback) {
                        callback("", Sentence::END);
                        return true;
                    }
                    _kpis.prompt.update(start.elapsed_usec());
                    start.reset();
                    State::busy(true);
                } else {
                    if (!callback(_tokenizer->decode(tokens), Sentence::CONTINUE)) return true;
                    _n_generated++;
                }

                // 4. 检查ff_tokens
                auto ff_tokens = _llgmatcher->compute_ff_tokens();
                size_t n = ff_tokens.size();

                fprintf(stderr, "[TF-Phase2] AR token output, checking ff_tokens: n=%zu, k=%zu\n", n, k);

                if (n >= k) {
                    // === n >= k: 跳过draft采样阶段 ===
                    _phase2_trigger_count++;
                    fprintf(stderr, "[TF-Phase2] n >= k, skipping draft phase, folding %zu tokens\n", n);

                    // 4.1 输出并consume所有ff_tokens
                    for (auto ff_tok : ff_tokens) {
                        if (_ctx->is_eos(ff_tok)) {
                            callback("", Sentence::END);
                            return true;
                        }
                        _llgmatcher->consume_token(ff_tok);
                        std::vector<int32_t> ff_vec = {static_cast<int32_t>(ff_tok)};
                        if (!callback(_tokenizer->decode(ff_vec), Sentence::CONTINUE)) return true;
                        _n_generated++;
                        _phase2_folded_tokens++;
                    }

                    // 4.2 构建推理序列: [AR] + [ff_tokens] + [forecast]
                    std::vector<int32_t> infer_tokens;
                    infer_tokens.push_back(tokens[0]);  // AR token
                    for (auto ff_tok : ff_tokens) {
                        infer_tokens.push_back(static_cast<int32_t>(ff_tok));
                    }
                    for (size_t i = 0; i < _draft; ++i) {
                        infer_tokens.push_back(_forecast_token_offset + i);
                    }

                    // 4.3 构建attention_map
                    std::vector<int32_t> attn_map(infer_tokens.size());
                    std::iota(attn_map.begin(), attn_map.end(), -1);

                    // 4.4 设置kv-prefix-offset (AR + ff_tokens不attend prefix)
                    size_t prefix_offset = 1 + ff_tokens.size();
                    engine.set({{"kv-prefix-offset", prefix_offset}});

                    // 4.5 检查context限制
                    if (_n_past + infer_tokens.size() > _ctx->size()) {
                        __WARN("Context limit exceeded ({} + {} > {})", _n_past, infer_tokens.size(), _ctx->size());
                        callback("", Sentence::END);
                        return true;
                    }

                    // 4.6 转换为embedding并推理
                    embedding.clear();
                    convertTokensToEmbeddings(infer_tokens, embedding, embedBufSize, m_t2eCallback);
                    if (!engine.process(embedding, attn_map, logits, true))
                        return Dialog::abort("Token folding phase2 inference failed", callback);

                    // 4.7 更新KV cache (AR + ff_tokens都需要保留)
                    std::vector<bool> selected(infer_tokens.size(), false);
                    for (size_t i = 0; i < 1 + ff_tokens.size(); ++i) {
                        selected[i] = true;
                    }
                    _n_past += 1 + ff_tokens.size();
                    update_kv(_n_past, selected);

                    // 4.8 从最后一个ff_token位置的logits采样新AR token
                    // infer_tokens = [AR, ff0, ff1, ..., ff(n-1), forecast0, ...]
                    // logits索引：   [0,   1,   2,  ...,  n,       n+1, ...]
                    // 最后一个ff_token是ff(n-1)，对应logits索引 n = ff_tokens.size()
                    size_t ar_logits_idx = ff_tokens.size();
                    tokens[0] = sample_to_verify(std::span{logits.data(), logits.size()}, ar_logits_idx, false);

                    fprintf(stderr, "[TF-Phase2] Sampled new AR token from logits[%zu], continuing loop\n", ar_logits_idx);
                    // 继续阶段2循环
                } else {
                    // === n < k: 退出阶段2，执行正常流程 ===
                    _phase2_skip_count++;
                    fprintf(stderr, "[TF-Phase2] n < k, exiting phase2 loop, entering normal draft flow\n");
                    break;
                }
            }

            // 退出阶段2后，tokens[0]是当前AR token（已输出）
            // 需要构建 [AR] + [forecast] 并推理，然后进入processFollowOnGeneration

            // 构建 [AR] + [forecast]
            tokens.resize(1);  // 只保留AR token
            for (size_t i = 0; i < _draft; ++i)
                tokens.push_back(_forecast_token_offset + i);

            attention_map.resize(tokens.size());
            std::iota(attention_map.begin(), attention_map.end(), -1);
            engine.set({{"kv-prefix-offset", 1}});

            if (_n_past + tokens.size() > _ctx->size()) {
                __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
                callback("", Sentence::END);
                return true;
            }

            embedding.clear();
            convertTokensToEmbeddings(tokens, embedding, embedBufSize, m_t2eCallback);
            if (!engine.process(embedding, attention_map, logits, true))
                return Dialog::abort("initial inference for SSD pipeline failed", callback);

            _n_past += 1;
            update_kv(_n_past, {});

            fprintf(stderr, "[TF-Phase2] Phase2 complete: triggers=%zu, skips=%zu, folded_tokens=%zu\n",
                    _phase2_trigger_count, _phase2_skip_count, _phase2_folded_tokens);

            status = processFollowOnGeneration(tokens, logits, callback);
        } else {
            // === 原有逻辑 (phase2未启用) ===
            _last_tok = tokens[0];
            if (_ctx->is_eos(_last_tok)) {
                callback("", Sentence::END);
                return true;
            }

            // 输出采样的AR token
            if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) return true;
            _n_generated++;

            if (!m_t2eCallback) {
                callback("", Sentence::END);
                return true;
            }

            // Mark TTFT
            _kpis.prompt.update(start.elapsed_usec());
            start.reset();
            State::busy(true);

            // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
            // process separately because logits are required for these tokens
            // step1.3: construct tokens([AR] + [Forecast token]*_draft).
            for (size_t i = 0; i < _draft; ++i)
                tokens.push_back(_forecast_token_offset + i);

            attention_map.resize(tokens.size());
            std::iota(attention_map.begin(), attention_map.end(), -1);
            engine.set({{"kv-prefix-offset", 1}}); // Prevent the last token from attending

            if (_n_past + tokens.size() > _ctx->size()) {
                __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
                callback("", Sentence::END);
                return true;
            }

            // Convert tokens to embeddings
            // reset embedding vector to make space for the next runs
            embedding.clear();
            convertTokensToEmbeddings(tokens, embedding, embedBufSize, m_t2eCallback);
            // step1.4: infer tokens([AR]+[Forecast]*) and get the whole logits
            if (!engine.process(embedding, attention_map, logits, true))
                return Dialog::abort("initial inference for SSD pipeline failed", callback);

            _n_past += 1;
            update_kv(_n_past, {});

            // Use existing as much as possible
            status = processFollowOnGeneration(tokens, logits, callback);
        }
    } else {
        std::vector<std::vector<int32_t>> streams;
        getTopK(logits, streams, _n_streams, _p_threshold, callback);
        _n_generated += streams.size();

        if (!m_t2eCallback) {
            for (auto& stream : streams) {
                if (!callback(_tokenizer->decode(stream) + "\n", Sentence::BEGIN)) return true;
            }
            callback("", Sentence::END);
            return true;
        }

        // Mark TTFT
        _kpis.prompt.update(start.elapsed_usec());
        start.reset();
        State::busy(true);

        if (streams.size() == 0) {
            callback("\n", Sentence::END);
            return true;
        }

        // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
        // process separately because logits are required for these tokens
        attention_map.resize(1 + _draft);
        std::iota(attention_map.begin(), attention_map.end(), -1);

        std::vector<size_t> stream_indices(streams.size());
        std::iota(stream_indices.begin(), stream_indices.end(), 0);

        std::vector<int32_t> multi_attn_mask;
        std::vector<size_t> past_map;
        const size_t kvPrefixOffset = 1;

        tileAttentionMask(attention_map, stream_indices, past_map, kvPrefixOffset, multi_attn_mask);

        // Accumulate input tokens from all streams
        std::vector<int32_t> multi_tokens;

        multi_tokens.reserve(streams.size() * (1 + _draft));
        for (int i = 0; i < streams.size(); i++) {
            multi_tokens.insert(multi_tokens.end(), streams[i].begin(), streams[i].end());
            for (int i = 0; i < _draft; ++i) {
                multi_tokens.push_back(_forecast_token_offset + i);
            }
        }

        // Convert tokens to embeddings
        // reset embedding vector to make space for the next runs
        embedding.clear();
        convertTokensToEmbeddings(multi_tokens, embedding, embedBufSize, m_t2eCallback);

        if (_n_past + multi_tokens.size() > _ctx->size()) {
            __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
            callback("", Sentence::END);
            return true;
        }

        if (!engine.process(embedding, multi_attn_mask, logits, true))
            return Dialog::abort("initial inference for SSD pipeline failed", callback);

        std::vector<bool> selected(multi_tokens.size(), false);
        for (int i = 0; i < multi_tokens.size(); i+=(_draft+1)) {
            selected[i] = true;
        }

        _n_past += streams.size();
        update_kv(_n_past, selected);

        status = processFollowOnGeneration(streams, logits, callback);
    }

    _kpis.generate.update(start.elapsed_usec());
    _env->logger().post(Logger::KPIS, kpis().dump(" "));
    start.reset();

    return status;
}

bool SelfSpecDecDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {

    // Check for prev failures and bail out early
    if (State::failed()) return false;

    Timer start;

    if(m_inputType != InputType::TOKENS) {
        __ERROR("Input type for model is not tokens.");
        return false;
    }

    State::clear();

    std::vector<float> logits;
    auto&              engine = *_engine["primary"];

    auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool>& selected) {
        if (!engine.updateKV(past, selected))
            return Dialog::abort("context size exceeded", callback);
        return true;
    };

    using FF = Engine::Feature::Flags;
    if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

    _env->logger().post(Logger::KPIS, kpis().dump(" "));
    start.reset();

    engine.set({{"kv-prefix-skip", _forecast_prefix}});

    std::vector<int32_t> attention_map(tokens.size());
    std::iota(attention_map.begin(), attention_map.end(), -1);

    // Process prompt
    _n_prompt += tokens.size();
    engine.set({{"kv-prefix-offset", tokens.size()}}); // Do not attend prefix

    if (_n_past + tokens.size() > _ctx->size()) {
        __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
        callback("", Sentence::END);
        return true;
    }

    // ===== CPU-NPU Parallel: 在prefill前启动async mask计算 =====
    bool prefill_async_enabled = (_llgmatcher && _matcherEnable && _llgmatcher->is_async_enabled());
    if (prefill_async_enabled) {
        _llgmatcher->start_prefill_async_mask();
    }

    if (!engine.process(tokens, attention_map, logits, false))
        return Dialog::abort("engine prompt processing failed", callback);

    // ===== CPU-NPU Parallel: 等待prefill mask结果 =====
    bool use_prefill_mask = false;
    if (prefill_async_enabled) {
        use_prefill_mask = _llgmatcher->wait_prefill_mask_result();
    }

    for(uint32_t idx= 0;idx<tokens.size();idx++){
        engine.updateTokenCheckpoint(tokens[idx],_n_past+idx);
    }

    _n_past += tokens.size();
    update_kv(_n_past, {});

    bool status = true;
    if (_n_streams <= 1) {
        // 使用预计算的mask采样第一个AR token
        tokens[0] = sample_to_verify(std::span{logits.data(),logits.size()}, 0, false, use_prefill_mask);
        tokens.resize(1);

        // Decode the first token.
        _last_tok = tokens[0];
        if (_ctx->is_eos(_last_tok)) {
            callback("", Sentence::END);
            return true;
        }

        if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) return true;
        _n_generated++;

        // Mark TTFT
        _kpis.prompt.update(start.elapsed_usec());
        start.reset();
        State::busy(true);

        // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
        // process separately because logits are required for these tokens
        for (int i = 0; i < _draft; ++i)
            tokens.push_back(_forecast_token_offset + i);

        attention_map.resize(tokens.size());
        std::iota(attention_map.begin(), attention_map.end(), -1);
        engine.set({{"kv-prefix-offset", 1}}); // Prevent the last token from attending

        if (_n_past + tokens.size() > _ctx->size()) {
            __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
            callback("", Sentence::END);
            return true;
        }

        if (!engine.process(tokens, attention_map, logits, true))
            return Dialog::abort("initial inference for SSD pipeline failed", callback);

        _n_past += 1;
        update_kv(_n_past, {});
        engine.updateTokenCheckpoint(tokens[0],_n_past);
        status = processFollowOnGeneration(tokens, logits, callback);
    } else {
        std::vector<std::vector<int32_t>> streams;
        getTopK(logits, streams, _n_streams, _p_threshold, callback);
        _n_generated += streams.size();

        // Mark TTFT
        _kpis.prompt.update(start.elapsed_usec());
        start.reset();
        State::busy(true);

        if (streams.size() == 0) {
            callback("\n", Sentence::END);
            return true;
        }

        // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
        // process separately because logits are required for these tokens
        attention_map.resize(1 + _draft);
        std::iota(attention_map.begin(), attention_map.end(), -1);

        std::vector<size_t> stream_indices(streams.size());
        std::iota(stream_indices.begin(), stream_indices.end(), 0);

        std::vector<int32_t> multi_attn_mask;
        std::vector<size_t> past_map;
        const size_t kvPrefixOffset = 1;

        tileAttentionMask(attention_map, stream_indices, past_map, kvPrefixOffset, multi_attn_mask);

        // Accumulate input tokens from all streams
        std::vector<int32_t> multi_tokens;

        multi_tokens.reserve(streams.size() * (1 + _draft));
        for (int i = 0; i < streams.size(); i++) {
            multi_tokens.insert(multi_tokens.end(), streams[i].begin(), streams[i].end());
            for (int i = 0; i < _draft; ++i) {
                multi_tokens.push_back(_forecast_token_offset + i);
            }
        }

        if (_n_past + multi_tokens.size() > _ctx->size()) {
            __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
            callback("", Sentence::END);
            return true;
        }

        if (!engine.process(multi_tokens, multi_attn_mask, logits, true))
            return Dialog::abort("initial inference for SSD pipeline failed", callback);

        std::vector<bool> selected(multi_tokens.size(), false);
        for (int i = 0; i < multi_tokens.size(); i+=(_draft+1)) {
            selected[i] = true;
        }

        _n_past += streams.size();
        update_kv(_n_past, selected);

        status = processFollowOnGeneration(streams, logits, callback);
    }

    _kpis.generate.update(start.elapsed_usec());
    _env->logger().post(Logger::KPIS, kpis().dump(" "));
    start.reset();

    return status;
}

void SelfSpecDecDialog::reset() {
  Dialog::reset();
  _n_past = _forecast_prefix;
  // ===== 重置统计变量 =====
  _has_saved_mask_for_next_ar = false;
  _mask_save_count = 0;
  _mask_reuse_count = 0;
  _total_compute_mask_time_us = 0.0f;
  _compute_mask_count = 0;
  _prefill_async_saved_time_us = 0.0f;
  // ===== 阶段2 Token Folding 统计变量重置 =====
  _phase2_trigger_count = 0;
  _phase2_skip_count = 0;
  _phase2_folded_tokens = 0;
  // ===== 阶段1 Token Folding 统计变量重置 =====
  _phase1_folded_tokens = 0;
  // ===== 阶段3 Token Folding 统计变量重置 =====
  _phase3_trigger_count = 0;
  _phase3_skip_count = 0;
  _phase3_folded_tokens = 0;
  size_t n_restored_prefix = _engine["primary"]->restore(_kv_prefix_name, true);
  if (n_restored_prefix != _forecast_prefix) {
    // clang-format off
    throw std::runtime_error( fmt::format( "SSD : Loaded {} KV$ from {} but expected {} KV$",
                                           n_restored_prefix, _kv_prefix_name, _forecast_prefix ) );
    // clang-format on
  }
}

bool SelfSpecDecDialog::save(const std::string& name) {
    if (_n_streams > 1) {
        throw std::runtime_error("Save is unsupported for multistream dialogs.");
    }
    return Dialog::save(name);
}

bool SelfSpecDecDialog::restore(const std::string& name) {
    if (_n_streams > 1) {
        throw std::runtime_error("Restore is unsupported for multistream dialogs.");
    }
    return Dialog::restore(name);
}

// Registrator instance
static OnLoad regy([]() {
    Dialog::__register(
            "ssd-q1",
            [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                return (Dialog*)new SelfSpecDecDialog(env, name, conf);
            }
    );
});

void needSsdDialog() {}

} // namespace CSD
