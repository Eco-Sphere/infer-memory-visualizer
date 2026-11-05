import assert from "node:assert/strict";
import test from "node:test";

import {
  fetchModelWeightData,
  fetchSafetensorsShards,
  getCachedModelWeightData,
} from "../app/modelscope.ts";

function makeHeaderFile(tensors) {
  const body = JSON.stringify({ __metadata__: {}, ...tensors });
  const buf = Buffer.alloc(8 + Buffer.byteLength(body));
  buf.writeBigUInt64LE(BigInt(Buffer.byteLength(body)), 0);
  buf.write(body, 8, "utf8");
  return buf;
}

const CONFIG = {
  hidden_size: 64,
  num_hidden_layers: 2,
  num_attention_heads: 8,
  num_key_value_heads: 4,
  head_dim: 16,
  vocab_size: 128,
  num_experts: 4,
  num_experts_per_tok: 2,
};

const INDEX = { weight_map: { "a.weight": "a.safetensors", "b.weight": "b.safetensors" } };
const HEADER_A = makeHeaderFile({ "a.weight": { dtype: "BF16", shape: [64, 64], data_offsets: [0, 8192] } });
const HEADER_B = makeHeaderFile({ "b.weight": { dtype: "F32", shape: [64, 64], data_offsets: [0, 16384] } });

function mountFetch(onCall) {
  const original = globalThis.fetch;
  globalThis.fetch = (async (input) => {
    onCall?.();
    const u = String(input);
    if (u.includes("config.json")) {
      return new Response(JSON.stringify(CONFIG), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (u.includes("safetensors.index.json")) {
      return new Response(JSON.stringify(INDEX), { status: 200 });
    }
    if (u.includes("a.safetensors")) return new Response(HEADER_A, { status: 206 });
    if (u.includes("b.safetensors")) return new Response(HEADER_B, { status: 206 });
    return new Response("not found", { status: 404 });
  });
  return () => { globalThis.fetch = original; };
}

test("fetchModelWeightData 命中会话缓存，切回不重复请求", async () => {
  let calls = 0;
  const restore = mountFetch(() => calls++);
  try {
    const first = await fetchModelWeightData("TestOwner", "CacheModel");
    const callsAfterFirst = calls;
    assert.ok(callsAfterFirst > 0, "首次应发起网络请求");
    assert.equal(first.tensors.length, 2);
    assert.equal(first.struct.hiddenSize, 64);
    assert.equal(first.shardCount, 2);
    // 张量按分片文件名排序合并
    assert.equal(first.tensors[0].name, "a.weight");
    assert.equal(first.tensors[0].bytes, 8192);
    assert.equal(first.tensors[1].name, "b.weight");
    assert.equal(first.tensors[1].bytes, 16384);

    const second = await fetchModelWeightData("TestOwner", "CacheModel");
    const callsAfterSecond = calls;
    assert.equal(callsAfterSecond, callsAfterFirst, "第二次应命中缓存，不再请求");
    assert.equal(second, first, "应返回同一缓存引用");
    assert.ok(getCachedModelWeightData("TestOwner", "CacheModel"));
  } finally {
    restore();
  }
});

test("拉取失败不缓存，重试仍会发起请求", async () => {
  let calls = 0;
  const original = globalThis.fetch;
  globalThis.fetch = (async (input) => {
    calls++;
    const u = String(input);
    if (u.includes("config.json")) return new Response("boom", { status: 500 });
    return new Response("not found", { status: 404 });
  });
  try {
    await assert.rejects(() => fetchModelWeightData("TestOwner", "FailModel"));
    assert.equal(getCachedModelWeightData("TestOwner", "FailModel"), undefined);
    const before = calls;
    await assert.rejects(() => fetchModelWeightData("TestOwner", "FailModel"));
    assert.ok(calls > before, "失败后重试应重新发起请求");
  } finally {
    globalThis.fetch = original;
  }
});

test("探测并计入可选权重，且按阶段上报加载进度", async () => {
  const optionalHeader = makeHeaderFile({ "global_rotation": { dtype: "F32", shape: [3, 3], data_offsets: [0, 36] } });
  const restore = mountFetchExtra(optionalHeader);

  const events = [];
  try {
    const data = await fetchModelWeightData("TestOwner", "OptionalModel", "master", (p) => events.push(p));

    // 可选权重已并入分片列表与张量清单
    assert.equal(data.shardCount, 2);
    assert.deepEqual(data.optionalShards, ["optional/quarot.safetensors"]);
    assert.equal(data.tensors.length, 2);
    assert.ok(data.tensors.some((t) => t.name === "global_rotation"));

    // 进度：先 index 阶段，再 shards 阶段，且 shards 以 done=total 收尾
    assert.ok(events.some((p) => p.phase === "index"), "应上报 index 阶段");
    const shardEvents = events.filter((p) => p.phase === "shards");
    assert.ok(shardEvents.length >= 1, "应上报 shards 阶段");
    const last = shardEvents[shardEvents.length - 1];
    assert.equal(last.total, 2);
    assert.equal(last.done, 2);
  } finally {
    restore();
  }
});

function mountFetchExtra(optionalHeader) {
  const original = globalThis.fetch;
  globalThis.fetch = (async (input) => {
    const u = String(input);
    if (u.includes("config.json")) return new Response(JSON.stringify(CONFIG), { status: 200, headers: { "content-type": "application/json" } });
    if (u.includes("safetensors.index.json")) return new Response(JSON.stringify({ weight_map: { "a.weight": "a.safetensors" } }), { status: 200 });
    if (u.includes("optional/quarot.safetensors")) return new Response(optionalHeader, { status: 206 });
    if (u.includes("quarot.safetensors")) return new Response("not found", { status: 404 });
    if (u.includes("a.safetensors")) return new Response(HEADER_A, { status: 206 });
    return new Response("not found", { status: 404 });
  });
  return () => { globalThis.fetch = original; };
}

test("发现旧版 Eco-Tech 动态量化索引，不误判为缺少权重", async () => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async input => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith("/quant_model_weight_w8a8_dynamic.safetensors.index.json")) {
      return new Response(JSON.stringify({ weight_map: {
        "model.layers.3.mlp.experts.0.gate_proj.weight": "quant_model_weight_w8a8_dynamic-00001-of-00162.safetensors",
        "model.layers.3.mlp.experts.1.gate_proj.weight": "quant_model_weight_w8a8_dynamic-00002-of-00162.safetensors",
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale": "quant_model_weight_w8a8_dynamic-00001-of-00162.safetensors",
      } }), { status: 200 });
    }
    return new Response("not found", { status: 404 });
  };
  try {
    const shards = await fetchSafetensorsShards("Eco-Tech", "DeepSeek-R1-0528-w8a8-mtp-QuaRot");
    assert.deepEqual(shards, [
      "quant_model_weight_w8a8_dynamic-00001-of-00162.safetensors",
      "quant_model_weight_w8a8_dynamic-00002-of-00162.safetensors",
    ]);
    assert.ok(calls.every(url => url.endsWith(".index.json")), "只读取索引定位分片");
  } finally {
    globalThis.fetch = original;
  }
});


test("兼容 Qwen2.5-VL 的量化索引和单文件名称", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    if (url.includes("/Qwen2.5-VL-72B-Instruct-w8a8/") && url.endsWith("/quant_model_weights.index.json")) {
      return new Response(JSON.stringify({ weight_map: { "model.embed_tokens.weight": "quant_model_weights-00001-of-00025.safetensors" } }), { status: 200 });
    }
    if (url.includes("/Qwen2.5-VL-7B-Instruct-w8a8/") && url.endsWith("/quant_model_weight_w8a8.safetensors")) {
      assert.equal(init.headers.Range, "bytes=0-7");
      return new Response(new Uint8Array(8), { status: 206 });
    }
    return new Response("not found", { status: 404 });
  };
  try {
    assert.deepEqual(await fetchSafetensorsShards("Eco-Tech", "Qwen2.5-VL-72B-Instruct-w8a8"), ["quant_model_weights-00001-of-00025.safetensors"]);
    assert.deepEqual(await fetchSafetensorsShards("Eco-Tech", "Qwen2.5-VL-7B-Instruct-w8a8"), ["quant_model_weight_w8a8.safetensors"]);
  } finally {
    globalThis.fetch = original;
  }
});
