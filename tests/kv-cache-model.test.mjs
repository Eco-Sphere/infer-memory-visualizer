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
  assert.equal(tp1.kvHeadsPerDevice, 4);
  assert.equal(tp4.kvHeadsPerDevice, 1);
  assert.equal(tp4.kvCache, tp1.kvCache / 4);
  assert.equal(tp4.indexCache, tp1.indexCache);
});
