import assert from "node:assert/strict";
import test from "node:test";

import { calculateKvCache, getKvCacheModelInfo } from "../app/kv-cache-model.ts";

test("MiniMax M3 stores K and V separately and supports FP4 index cache", () => {
  const info = getKvCacheModelInfo("minimax-m3");
  assert.ok(info);
  assert.equal(info.layers, 60);
  assert.equal(info.sparseLayers, 57);
  const result = calculateKvCache({
    modelId: "minimax-m3",
    tokens: 1024,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp4_int4",
  });

  assert.ok(result);
  assert.equal(result.kvCopies, 2);
  assert.equal(result.kvCache, 125_829_120);
  assert.equal(result.indexCache, 3_735_552);
  assert.equal(result.total, 129_564_672);
});

test("KV and index precision can be selected independently", () => {
  const bf16 = calculateKvCache({
    modelId: "minimax-m3",
    tokens: 4096,
    sequences: 8,
    kvPrecision: "bf16_fp16",
    indexPrecision: "bf16_fp16",
  });
  const compressed = calculateKvCache({
    modelId: "minimax-m3",
    tokens: 4096,
    sequences: 8,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp4_int4",
  });

  assert.ok(bf16 && compressed);
  assert.equal(compressed.kvCache, bf16.kvCache / 2);
  assert.equal(compressed.indexCache, bf16.indexCache / 4);
});

test("each MTP layer adds one KV layer and one sparse Index layer", () => {
  const common = {
    modelId: "minimax-m3",
    tokens: 1024,
    sequences: 1,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp4_int4",
  };
  const baseline = calculateKvCache({ ...common, mtpLayers: 0 });
  const threeMtp = calculateKvCache({ ...common, mtpLayers: 3 });

  assert.ok(baseline && threeMtp);
  assert.equal(threeMtp.effectiveKvLayers, 63);
  assert.equal(threeMtp.effectiveIndexLayers, 60);
  assert.equal(
    threeMtp.kvCache - baseline.kvCache,
    3 * 1024 * 2 * 4 * 128,
  );
  assert.equal(
    threeMtp.indexCache - baseline.indexCache,
    3 * 1024 * 128 * 0.5,
  );
});

test("KV cache shards across TP while the index cache stays replicated", () => {
  const common = {
    modelId: "minimax-m3",
    tokens: 4096,
    sequences: 8,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp8_int8",
  };
  const tp1 = calculateKvCache({ ...common, tpSize: 1 });
  const tp4 = calculateKvCache({ ...common, tpSize: 4 });

  assert.ok(tp1 && tp4);
  assert.equal(tp4.kvCache, tp1.kvCache / 4);
  assert.equal(tp4.indexCache, tp1.indexCache);
});

test("Step-5 combines 69 sliding layers and 23 paged DSA layers with fixed index formats", () => {
  assert.deepEqual(getKvCacheModelInfo("step5-preview"), {
    layers: 92, sparseLayers: 23, kvHeads: 4, headDim: 192, indexHeadDim: 256, source: "local",
  });
  const input = { modelId: "step5-preview", tokens: 1024, sequences: 2, kvPrecision: "bf16_fp16", indexPrecision: "fp4_int4", tpSize: 4 };
  const base = calculateKvCache(input);
  assert.equal(base.kvCache, 90_439_680);
  assert.equal(base.indexCache, 1_507_328);
  assert.equal(base.indexBytesPerElement, 1);
  const tp1 = calculateKvCache({ ...input, tpSize: 1 });
  const tp8 = calculateKvCache({ ...input, tpSize: 8 });
  assert.equal(tp1.kvCache, base.kvCache * 4);
  assert.equal(tp8.kvCache, base.kvCache);
  assert.equal(tp8.indexCache, base.indexCache);
  const fp8 = calculateKvCache({ ...input, kvPrecision: "fp8_int8", indexPrecision: "bf16_fp16" });
  assert.equal(fp8.kvCache, base.kvCache / 2);
  assert.equal(fp8.indexCache, base.indexCache);
});

test("Step-5 MTP adds only sliding KV; DSA capacity rounds to 16-token pages", () => {
  const input = { modelId: "step5-preview", tokens: 1024, sequences: 2, kvPrecision: "bf16_fp16", indexPrecision: "fp8_int8", tpSize: 4 };
  const base = calculateKvCache(input);
  const mtp = calculateKvCache({ ...input, mtpLayers: 3 });
  assert.equal(mtp.effectiveKvLayers, 95);
  assert.equal(mtp.effectiveIndexLayers, 23);
  assert.equal(mtp.indexCache, base.indexCache);
  assert.equal(mtp.kvCache - base.kvCache, 2_359_296);
  assert.equal(mtp.mtpKvCache, 2_359_296);
  assert.equal(calculateKvCache({ ...input, mtpLayers: 4 }).total, mtp.total);
  const aligned = calculateKvCache({ ...input, tokens: 1040 });
  assert.equal(calculateKvCache({ ...input, tokens: 1025 }).total, aligned.total);
  const short = calculateKvCache({ ...input, tokens: 1, sequences: 1 });
  assert.equal(short.kvCache, 2 * 192 * 2 * (23 * 16 + 69 * 512));
  assert.equal(short.indexCache, 23 * 2 * 256);
  for (const empty of [{ tokens: 0 }, { sequences: 0 }]) {
    assert.equal(calculateKvCache({ ...input, ...empty }).total, 0);
  }
});


test("Step-5 默认容量下每个 MTP 层的缓存增量在三位 GiB 显示中可见", () => {
  const results = [0, 1, 2, 3].map(mtpLayers => calculateKvCache({
    modelId: "step5-preview", tokens: 131072, sequences: 8,
    kvPrecision: "fp8_int8", indexPrecision: "fp8_int8", tpSize: 4, mtpLayers,
  }));
  assert.deepEqual(results.map(r => (r.total / 1024 ** 3).toFixed(3)), ["9.445", "9.446", "9.448", "9.449"]);
  for (let i = 0; i < results.length; i++) {
    assert.equal(results[i].mtpKvCache, i * 1.5 * 1024 ** 2);
    assert.equal(results[i].total - results[0].total, results[i].mtpKvCache);
    assert.equal(results[i].indexCache, 8 * 23 * (131072 / 8) * 256);
  }
});
