import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyTensor,
  calculateWeightFromTensors,
} from "../app/weight-calc.ts";

test("classifyTensor 覆盖各模块命名", () => {
  assert.equal(classifyTensor("model.embed_tokens.weight"), "embedding");
  assert.equal(classifyTensor("lm_head.weight"), "lm_head");
  assert.equal(classifyTensor("model.layers.0.self_attn.q_proj.weight"), "attention");
  assert.equal(classifyTensor("model.layers.0.self_attn.o_proj.weight"), "attention");
  assert.equal(classifyTensor("model.layers.0.mlp.gate_proj.weight"), "dense_mlp");
  assert.equal(classifyTensor("model.layers.0.mlp.up_proj.weight_scale"), "dense_mlp");
  assert.equal(classifyTensor("model.layers.0.mlp.experts.0.w1.weight"), "routed_expert");
  assert.equal(classifyTensor("block_sparse_moe.gate.weight"), "router");
  assert.equal(classifyTensor("model.norm.weight"), "norm");
  assert.equal(classifyTensor("model.layers.0.input_layernorm.weight"), "norm");
  assert.equal(classifyTensor("model.visual.blocks.0.attn.proj.weight"), "vision");
  assert.equal(classifyTensor("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"), "attention");
  assert.equal(classifyTensor("model.layers.0.mlp.shared_experts.gate_proj.weight"), "shared_expert");
});

test("calculateWeightFromTensors 全量精确求和且单卡按模块切分", () => {
  const tensors = [
    { name: "model.embed_tokens.weight", dtype: "BF16", shape: [100, 64], bytes: 12800 },
    { name: "model.layers.0.self_attn.q_proj.weight", dtype: "I8", shape: [64, 64], bytes: 4096 },
    { name: "model.layers.0.self_attn.o_proj.weight", dtype: "I8", shape: [64, 64], bytes: 4096 },
    { name: "model.layers.0.mlp.gate_proj.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
    { name: "model.layers.0.mlp.experts.0.w1.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
    { name: "model.layers.0.input_layernorm.weight", dtype: "F32", shape: [64], bytes: 256 },
  ];
  const p = {
    tp: 4,
    ep: 8,
    oprojTp: 2,
    embeddingTp: 4,
    lmHeadTp: 4,
    sharedExpertTp: 1,
  };

  const r = calculateWeightFromTensors(tensors, p);

  // 全量 = 12800+4096+4096+8192+8192+256 = 37632
  assert.equal(r.fullWeight, 37632);

  // 单卡：embed/4 + q/tp4 + o(oprojTp)/2 + mlp/tp4 + expert/ep8 + norm/1
  assert.equal(r.breakdown.embedding.perDevice, 3200);
  assert.equal(r.breakdown.attention.perDevice, 1024 + 2048);
  assert.equal(r.breakdown.dense_mlp.perDevice, 2048);
  assert.equal(r.breakdown.routed_expert.perDevice, 1024);
  assert.equal(r.breakdown.norm.perDevice, 256);
  assert.equal(r.perDeviceWeight, 3200 + 1024 + 2048 + 2048 + 1024 + 256);
});

test("TP=EP=1 时单卡权重等于全量权重", () => {
  const tensors = [
    { name: "model.embed_tokens.weight", dtype: "BF16", shape: [100, 64], bytes: 12800 },
    { name: "model.layers.0.mlp.experts.0.w1.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
  ];
  const p = { tp: 1, ep: 1, oprojTp: 1, embeddingTp: 1, lmHeadTp: 1, sharedExpertTp: 1 };
  const r = calculateWeightFromTensors(tensors, p);
  assert.equal(r.perDeviceWeight, r.fullWeight);
  assert.equal(r.fullWeight, 12800 + 8192);
});

test("DeepSeek MLA A 投影保持复制，不按 TP 切分", () => {
  const tensors = [
    { name: "model.layers.0.self_attn.q_a_proj.weight", dtype: "BF16", shape: [16, 64], bytes: 2048 },
    { name: "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", dtype: "BF16", shape: [16, 64], bytes: 2048 },
    { name: "model.layers.0.self_attn.q_b_proj.weight", dtype: "BF16", shape: [64, 16], bytes: 2048 },
  ];
  const r = calculateWeightFromTensors(tensors, { tp: 4, ep: 4, oprojTp: 4, embeddingTp: 4, lmHeadTp: 4, sharedExpertTp: 1 });
  assert.equal(r.breakdown.attention.perDevice, 2048 + 2048 + 2048 / 4);
});
