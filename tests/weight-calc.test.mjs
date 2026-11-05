import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyTensor,
  calculateWeightFromTensors,
  weightGroupFormula,
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
    attentionTp: 4,
    oprojTp: 2,
    embeddingTp: 4,
    lmHeadTp: 4,
    sharedExpertTp: 1,
  };

  const r = calculateWeightFromTensors(tensors, p);

  // 全量 = 12800+4096+4096+8192+8192+256 = 37632
  assert.equal(r.fullWeight, 37632);

  // 单卡：embed/4 + q(attentionTp)/4 + o(oprojTp)/2 + mlp/tp4 + expert/ep8 + norm/1
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
  const p = { tp: 1, ep: 1, attentionTp: 1, oprojTp: 1, embeddingTp: 1, lmHeadTp: 1, sharedExpertTp: 1 };
  const r = calculateWeightFromTensors(tensors, p);
  assert.equal(r.perDeviceWeight, r.fullWeight);
  assert.equal(r.fullWeight, 12800 + 8192);
});

test("公式合并重复层与专家，保留不同 shape、dtype 与并行切分", () => {
  const tensor = (name, dtype, shape, width) => ({
    name, dtype, shape, bytes: shape.reduce((a, b) => a * b, 1) * width,
  });
  const tensors = [
    tensor("model.layers.0.experts.0.w1.weight", "I8", [3072, 6144], 1),
    tensor("model.layers.1.experts.127.w1.weight", "I8", [3072, 6144], 1),
    tensor("model.layers.0.experts.0.w1.weight_scale", "F32", [3072, 1], 4),
    tensor("model.layers.2.experts.0.w1.weight", "I8", [6144, 3072], 1),
    tensor("model.layers.3.experts.0.w1.weight", "BF16", [3072, 6144], 2),
    tensor("model.layers.0.self_attn.q_proj.weight", "I8", [6144, 6144], 1),
    tensor("model.layers.0.self_attn.o_proj.weight", "I8", [6144, 6144], 1),
    tensor("vision_tower.patch_embedding.weight", "F32", [1280, 3, 2, 14, 14], 4),
  ];
  const result = calculateWeightFromTensors(tensors, {
    tp: 4, ep: 32, attentionTp: 4, oprojTp: 2,
    embeddingTp: 4, lmHeadTp: 4, sharedExpertTp: 1,
  });
  const groups = result.breakdown.routed_expert.groups;
  assert.equal(groups.length, 4);
  assert.equal(groups[0].name, "model.layers.*.experts.*.w1.weight");
  assert.equal(groups[0].count, 2);
  assert.equal(weightGroupFormula(groups[0]), "2 × (3072 × 6144) × 1 B ÷ EP 32 ÷ 1024³");
  assert.equal(weightGroupFormula(groups[1]), "1 × (3072 × 1) × 4 B ÷ EP 32 ÷ 1024³");
  assert.deepEqual(result.breakdown.attention.groups.map(g => g.divisor), [4, 2]);
  assert.equal(weightGroupFormula(result.breakdown.vision.groups[0]),
    "1 × (1280 × 3 × 2 × 14 × 14) × 4 B ÷ 复制 1 ÷ 1024³");
  for (const b of Object.values(result.breakdown)) {
    assert.equal(b.groups.reduce((sum, g) => sum + g.count, 0), b.tensors);
    assert.equal(b.groups.reduce((sum, g) => sum + g.full, 0), b.full);
    assert.equal(b.groups.reduce((sum, g) => sum + g.perDevice, 0), b.perDevice);
    for (const g of b.groups) {
      assert.equal(g.count * g.shape.reduce((a, b) => a * b, 1) * g.bytesPerElement / g.divisor, g.perDevice);
    }
  }
});


test("Step 的打包 MoE 权重及量化辅助张量按 EP 切分，共享专家与 Router 分开", () => {
  const tensors = [
    { name: "model.layers.3.moe.gate_proj.weight", dtype: "I8", shape: [288, 1280, 4096], bytes: 288 * 1280 * 4096 },
    { name: "model.layers.3.moe.up_proj.weight_scale", dtype: "F32", shape: [288, 1280, 1], bytes: 288 * 1280 * 4 },
    { name: "model.layers.3.moe.down_proj.weight_offset", dtype: "F32", shape: [288, 4096, 1], bytes: 288 * 4096 * 4 },
    { name: "model.layers.3.share_expert.down_proj.weight", dtype: "BF16", shape: [4096, 1280], bytes: 4096 * 1280 * 2 },
    { name: "model.layers.3.moe.gate.weight", dtype: "F32", shape: [288, 4096], bytes: 288 * 4096 * 4 },
    { name: "model.layers.3.moe.router_bias", dtype: "F32", shape: [288], bytes: 288 * 4 },
    { name: "model.layers.0.mlp.down_proj.weight", dtype: "BF16", shape: [4096, 11264], bytes: 4096 * 11264 * 2 },
  ];
  const p = { tp: 4, ep: 32, attentionTp: 4, oprojTp: 4, embeddingTp: 4, lmHeadTp: 4, sharedExpertTp: 2 };
  const result = calculateWeightFromTensors(tensors, p);
  assert.equal(result.breakdown.routed_expert.tensors, 3);
  assert.equal(result.breakdown.shared_expert.tensors, 1);
  assert.equal(result.breakdown.router.tensors, 2);
  assert.equal(result.breakdown.dense_mlp.tensors, 1);
  assert.equal(result.breakdown.routed_expert.perDevice, tensors.slice(0, 3).reduce((s, t) => s + t.bytes, 0) / 32);
  assert.equal(result.breakdown.shared_expert.perDevice, tensors[3].bytes / 2);
  assert.equal(result.breakdown.router.perDevice, tensors[4].bytes + tensors[5].bytes);
  assert.equal(result.breakdown.dense_mlp.perDevice, tensors[6].bytes / 4);
  assert.ok(result.breakdown.routed_expert.groups.every(g => g.parallelism === "EP" && g.divisor === 32));
  const full = tensors.reduce((s, t) => s + t.bytes, 0);
  assert.equal(result.fullWeight, full);
  const unsharded = calculateWeightFromTensors(tensors, { tp: 1, ep: 1, attentionTp: 1, oprojTp: 1, embeddingTp: 1, lmHeadTp: 1, sharedExpertTp: 1 });
  assert.equal(unsharded.perDeviceWeight, full);
  assert.equal(classifyTensor("model.layers.3.moe.down_proj.weight"), "routed_expert");
  assert.equal(classifyTensor("model.layers.3.moe.shared_experts.up_proj.weight"), "shared_expert");
  assert.equal(classifyTensor("model.layers.3.moe.norm.weight"), "norm");
});

test("公共 latent MoE 投影/Norm 与 Router expert_bias 不按 EP 切分", () => {
  const tensors = [
    { name: "model.layers.0.mlp.routed_expert_down_proj.weight", dtype: "BF16", shape: [3584, 7168], bytes: 3584 * 7168 * 2 },
    { name: "model.layers.0.mlp.routed_expert_up_proj.weight_scale", dtype: "F32", shape: [7168, 1], bytes: 7168 * 4 },
    { name: "model.layers.0.mlp.routed_expert_norm.weight", dtype: "BF16", shape: [3584], bytes: 3584 * 2 },
    { name: "model.layers.0.mlp.gate.expert_bias", dtype: "F32", shape: [256], bytes: 256 * 4 },
    { name: "model.layers.0.mlp.experts.0.gate_proj.weight", dtype: "BF16", shape: [3072, 3584], bytes: 3072 * 3584 * 2 },
  ];
  const p = { tp: 4, ep: 32, attentionTp: 4, oprojTp: 4, embeddingTp: 4, lmHeadTp: 4, sharedExpertTp: 1 };
  const r = calculateWeightFromTensors(tensors, p);
  assert.deepEqual(tensors.map(t => classifyTensor(t.name)), ["dense_mlp", "dense_mlp", "norm", "router", "routed_expert"]);
  assert.equal(r.breakdown.dense_mlp.perDevice, (tensors[0].bytes + tensors[1].bytes) / 4);
  assert.equal(r.breakdown.norm.perDevice, tensors[2].bytes);
  assert.equal(r.breakdown.router.perDevice, tensors[3].bytes);
  assert.equal(r.breakdown.routed_expert.perDevice, tensors[4].bytes / 32);
});

test("Hy3 共享 MLP、Qwen 共享门控和 DeepSeek V4 短名称分类", () => {
  const cases = [
    ["language_model.model.layers.0.block_sparse_moe.routed_expert_down_proj.weight", "dense_mlp"],
    ["language_model.model.layers.0.block_sparse_moe.routed_expert_up_proj.weight", "dense_mlp"],
    ["language_model.model.layers.0.block_sparse_moe.routed_expert_norm.weight", "norm"],
    ["model.layers.0.mlp.shared_mlp.gate_proj.weight", "shared_expert"],
    ["model.layers.0.mlp.shared_mlp.up_proj.weight_scale", "shared_expert"],
    ["model.layers.0.mlp.shared_mlp.down_proj.weight_offset", "shared_expert"],
    ["model.layers.0.mlp.expert_bias", "router"],
    ["model.language_model.layers.0.mlp.shared_expert_gate.weight", "router"],
    ["mtp.layers.0.mlp.shared_expert_gate.weight_scale", "router"],
    ["embed.weight", "embedding"],
    ["mtp.0.emb.tok_emb.weight", "embedding"],
    ["head.weight", "lm_head"],
    ["mtp.0.head.weight", "lm_head"],
    ["model.layers.0.shared_head.head.weight", "lm_head"],
    ["model.layers.0.transformer.shared_head.output.weight", "lm_head"],
    ["model.layers.0.shared_head.norm.weight", "norm"],
    ["vision_model.head.weight", "vision"],
  ];
  for (const [name, expected] of cases) assert.equal(classifyTensor(name), expected, name);
  const tensors = cases.slice(3, 10).map(([name]) => ({ name, dtype: "BF16", shape: [64, 64], bytes: 8192 }));
  const r = calculateWeightFromTensors(tensors, {
    tp: 4, ep: 32, attentionTp: 4, oprojTp: 4, embeddingTp: 2, lmHeadTp: 4, sharedExpertTp: 8,
  });
  assert.equal(r.breakdown.shared_expert.perDevice, 3 * 8192 / 8);
  assert.equal(r.breakdown.router.perDevice, 3 * 8192);
  assert.equal(r.breakdown.embedding.perDevice, 8192 / 2);
});


test("Ling/Ring attention.dense 及其量化 scale 使用 O-Proj TP", () => {
  const tensors = [
    { name: "model.layers.0.attention.dense.weight", dtype: "BF16", shape: [64, 64], bytes: 8192 },
    { name: "model.layers.0.attention.dense.weight_scale", dtype: "F32", shape: [64, 1], bytes: 256 },
    { name: "model.layers.0.attention.query_key_value.weight", dtype: "BF16", shape: [192, 64], bytes: 24576 },
  ];
  const r = calculateWeightFromTensors(tensors, {
    tp: 4, ep: 32, attentionTp: 4, oprojTp: 2, embeddingTp: 4, lmHeadTp: 4, sharedExpertTp: 1,
  });
  assert.equal(r.breakdown.attention.perDevice, (8192 + 256) / 2 + 24576 / 4);
  assert.deepEqual(r.breakdown.attention.groups.map(g => g.parallelism), ["O-Proj TP", "O-Proj TP", "Attention TP"]);
  assert.equal(classifyTensor("model.layers.0.mlp.dense.weight"), "dense_mlp");
});

test("Step-5 replicates MQA indexer weights and KV/provider groups when TP exceeds four", () => {
  const names = ["k_proj.weight", "v_proj.weight", "sparse_indexer_q.weight", "sparse_indexer_w.weight",
    "sparse_indexer_k.weight", "sparse_indexer_z.weight", "ssmax_s", "q_proj.weight", "o_proj.weight"];
  const tensors = names.map(name => ({ name: `model.layers.3.self_attn.${name}`, dtype: "BF16", shape: [64], bytes: 128 }));
  const r = calculateWeightFromTensors(tensors, {
    step5: true, tp: 8, ep: 32, attentionTp: 8, oprojTp: 2, embeddingTp: 8, lmHeadTp: 8, sharedExpertTp: 8,
  });
  assert.deepEqual(r.breakdown.attention.groups.map(g => g.divisor), [4, 4, 4, 4, 1, 1, 1, 8, 2]);
  assert.equal(r.breakdown.attention.perDevice, 4 * 128 / 4 + 3 * 128 + 128 / 8 + 128 / 2);
  assert.equal(classifyTensor("model.layers.3.moe.gate_proj.weight_scale_inv"), "routed_expert");
  assert.equal(classifyTensor("model.layers.3.share_expert.gate_proj.weight"), "shared_expert");
});

test("MTP layers 筛选追加投机层，权重、模块明细和公式同步变化", () => {
  const tensors = [
    { name: "model.layers.92.self_attn.q_proj.weight", dtype: "BF16", shape: [64, 64], bytes: 8192 },
    { name: "model.layers.91.moe.gate_proj.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
    { name: "model.layers.94.self_attn.q_proj.weight", dtype: "BF16", shape: [64, 64], bytes: 8192 },
    { name: "vision_model.encoder.layers.100.attn.weight", dtype: "BF16", shape: [64], bytes: 128 },
    { name: "model.layers.93.self_attn.q_proj.weight", dtype: "BF16", shape: [64, 64], bytes: 8192 },
  ];
  const p = { tp: 4, ep: 32, attentionTp: 4, oprojTp: 4, embeddingTp: 4, lmHeadTp: 4, sharedExpertTp: 4 };
  const results = [0, 1, 2, 3].map(layers => calculateWeightFromTensors(tensors, { ...p, mtp: { layers, mainLayers: 92 } }));
  for (let i = 0; i < results.length; i++) {
    assert.equal(results[i].fullWeight, 8192 + 128 + i * 8192);
    assert.equal(results[i].perDeviceWeight, 8192 / 32 + 128 + i * 8192 / 4);
    assert.equal(results[i].breakdown.attention.tensors, i);
    assert.equal(results[i].breakdown.vision.tensors, 1);
    if (i) assert.equal(results[i].breakdown.attention.groups[0].count, i);
  }
  assert.equal(calculateWeightFromTensors(tensors, p).fullWeight, results[3].fullWeight);
});

test("独立 MTP 路径按实际层编号启用，共用投影只计一次", () => {
  for (const prefix of ["mtp", "mtp.layers", "model.mtp_layers", "model.nextn_predict_layers"]) {
    const tensor = name => ({ name, dtype: "BF16", shape: [64], bytes: 128 });
    const tensors = [tensor(`${prefix}.2.self_attn.q_proj.weight`), tensor("model.norm.weight"),
      tensor(`${prefix}.1.self_attn.q_proj.weight`), tensor("mtp.fc.weight")];
    const p = { tp: 1, ep: 1, attentionTp: 1, oprojTp: 1, embeddingTp: 1, lmHeadTp: 1, sharedExpertTp: 1 };
    const calc = layers => calculateWeightFromTensors(tensors, { ...p, mtp: { layers, mainLayers: 92 } });
    assert.equal(calc(0).fullWeight, 128, prefix);
    assert.equal(calc(1).fullWeight, 384, prefix);
    assert.equal(calc(2).fullWeight, 512, prefix);
    assert.equal(calc(5).fullWeight, 512, prefix);
  }
});
