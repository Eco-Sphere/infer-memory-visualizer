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