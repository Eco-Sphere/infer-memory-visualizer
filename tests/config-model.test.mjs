import assert from "node:assert/strict";
import test from "node:test";

import { extractModelStructure } from "../app/config-model.ts";

test("MiniMax 风格嵌套 text_config 提取", () => {
  const cfg = {
    model_type: "minimax_m3_vl",
    architectures: ["MiniMaxM3VLForCausalLM"],
    text_config: {
      model_type: "minimax_m3",
      hidden_size: 6144,
      num_hidden_layers: 64,
      num_attention_heads: 48,
      num_key_value_heads: 8,
      head_dim: 128,
      vocab_size: 151936,
      intermediate_size: 28672,
      moe_intermediate_size: 1024,
      num_experts: 256,
      num_experts_per_tok: 12,
      n_shared_experts: 1,
    },
    vision_config: { hidden_size: 1152 },
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.hiddenSize, 6144);
  assert.equal(s.numLayers, 64);
  assert.equal(s.attentionHeads, 48);
  assert.equal(s.kvHeads, 8);
  assert.equal(s.headDim, 128);
  assert.equal(s.vocabSize, 151936);
  assert.equal(s.expertCount, 256);
  assert.equal(s.topK, 12);
  assert.equal(s.sharedExperts, 1);
  assert.equal(s.intermediateSize, 28672);
  assert.equal(s.moeIntermediateSize, 1024);
  assert.equal(s.hasVision, true);
  assert.deepEqual(s.architectures, ["MiniMaxM3VLForCausalLM"]);
});

test("DeepSeek 风格平铺字段提取（num_experts 别名）", () => {
  const cfg = {
    model_type: "deepseek_v4",
    hidden_size: 7168,
    num_hidden_layers: 61,
    num_attention_heads: 128,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 129280,
    intermediate_size: 18432,
    moe_intermediate_size: 2048,
    n_routed_experts: 256,
    num_experts_per_tok: 8,
    n_shared_experts: 1,
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.hiddenSize, 7168);
  assert.equal(s.numLayers, 61);
  assert.equal(s.expertCount, 256);
  assert.equal(s.topK, 8);
  assert.equal(s.hasVision, false);
});

test("dense 模型无 expert 字段时 expertCount=0", () => {
  const cfg = {
    model_type: "qwen3",
    hidden_size: 4096,
    num_hidden_layers: 36,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    intermediate_size: 10240,
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.expertCount, 0);
  assert.equal(s.topK, 0);
});

test("Step-3.7-Flash 读取 text_config 内的专家、TopK 和 GQA 别名", () => {
  // 来自 stepfun-ai/Step-3.7-Flash/config.json 的相关字段。
  const s = extractModelStructure({
    model_type: "step3p7",
    vision_config: { model_type: "perception_encoder" },
    text_config: {
      hidden_size: 4096,
      num_hidden_layers: 45,
      num_attention_heads: 64,
      num_attention_groups: 8,
      head_dim: 128,
      moe_num_experts: 288,
      moe_top_k: 8,
      moe_intermediate_size: 1280,
    },
  });
  assert.equal(s.expertCount, 288);
  assert.equal(s.topK, 8);
  assert.equal(s.kvHeads, 8);
  assert.equal(s.expertCount / (4 * 8), 9);
  assert.equal(s.hiddenSize, 4096);
  assert.equal(s.moeIntermediateSize, 1280);
});

test("语言配置的别名优先于顶层字段", () => {
  const s = extractModelStructure({
    num_experts: 16,
    num_experts_per_tok: 2,
    num_key_value_heads: 4,
    text_config: { moe_num_experts: 288, moe_top_k: 8, num_attention_groups: 8 },
  });
  assert.equal(s.expertCount, 288);
  assert.equal(s.topK, 8);
  assert.equal(s.kvHeads, 8);
});

test("顶层 Step 别名仍然可读，标准字段优先于同级别名", () => {
  const s = extractModelStructure({ moe_num_experts: 288, moe_top_k: 8, num_attention_groups: 8 });
  assert.equal(s.expertCount, 288);
  assert.equal(s.topK, 8);
  assert.equal(s.kvHeads, 8);
  const standard = extractModelStructure({
    text_config: { num_experts: 128, moe_num_experts: 288,
      num_experts_per_tok: 4, moe_top_k: 8, num_key_value_heads: 2, num_attention_groups: 8 },
  });
  assert.equal(standard.expertCount, 128);
  assert.equal(standard.topK, 4);
  assert.equal(standard.kvHeads, 2);
});

test("InternVL llm_config 提取 MoE 参数，避免读取视觉部分或顶层的维度", () => {
  const s = extractModelStructure({
    model_type: "internvl_chat",
    hidden_size: 3200,
    vision_config: { hidden_size: 3200, num_hidden_layers: 45 },
    llm_config: {
      hidden_size: 4096, num_hidden_layers: 94, num_attention_heads: 64,
      num_key_value_heads: 4, head_dim: 128, vocab_size: 151936,
      num_experts: 128, num_experts_per_tok: 8, moe_intermediate_size: 1536,
    },
  });
  assert.equal(s.hiddenSize, 4096);
  assert.equal(s.numLayers, 94);
  assert.equal(s.attentionHeads, 64);
  assert.equal(s.kvHeads, 4);
  assert.equal(s.headDim, 128);
  assert.equal(s.vocabSize, 151936);
  assert.equal(s.expertCount, 128);
  assert.equal(s.topK, 8);
  assert.equal(s.moeIntermediateSize, 1536);
  assert.equal(s.expertCount / 32, 4);
  assert.equal(s.hasVision, true);
});

test("InternVL3_5-38B 是 Dense，但语言维度必须有效", () => {
  const s = extractModelStructure({
    model_type: "internvl_chat",
    vision_config: { hidden_size: 3200, num_hidden_layers: 45 },
    llm_config: {
      hidden_size: 5120, num_hidden_layers: 64, num_attention_heads: 64,
      num_key_value_heads: 8, head_dim: 128, vocab_size: 151936,
    },
  });
  assert.equal(s.hiddenSize, 5120);
  assert.equal(s.numLayers, 64);
  assert.equal(s.expertCount, 0);
  assert.equal(s.topK, 0);
});

test("Step-3.5 的顶层 TopK 与 Kimi-K3 的嵌套共享专家别名", () => {
  const step = extractModelStructure({
    hidden_size: 4096, moe_num_experts: 288, moe_top_k: 8, num_attention_groups: 8,
  });
  assert.equal(step.expertCount, 288);
  assert.equal(step.topK, 8);
  const kimi = extractModelStructure({
    text_config: { num_experts: 896, num_experts_per_token: 16, num_shared_experts: 2 },
  });
  assert.equal(kimi.expertCount, 896);
  assert.equal(kimi.topK, 16);
  assert.equal(kimi.sharedExperts, 2);
});
