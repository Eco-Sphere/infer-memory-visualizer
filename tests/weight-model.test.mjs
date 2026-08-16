import assert from "node:assert/strict";
import test from "node:test";

import {
  calculateDeepseekV4Weight,
  calculateMiniMaxM3Weight,
  fp8BlockMemory,
  mxfp4K32Memory,
  mxfp8MatrixMemory,
} from "../app/weight-model.ts";

const profile = {
  vocabSize: 200064,
  totalLayers: 60,
  denseLayers: 3,
  moeLayers: 57,
  expertIntermediateSize: 3072,
  denseIntermediateSize: 12288,
  attentionHeads: 64,
  kvHeads: 4,
  headDim: 128,
  indexerHeads: 4,
  indexerHeadDim: 128,
  sharedExperts: 1,
  vocabPaddingSize: 64,
};

test("MXFP8 matrix memory adds one byte per [1, 32] block", () => {
  assert.deepEqual(mxfp8MatrixMemory(3, 33), {
    payload: 99,
    scales: 6,
    total: 105,
  });
});

test("TP4/DP8 shards routed, attention, dense and shared scales correctly", () => {
  const common = {
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 4,
    epSize: 32,
  };
  const replicatedShared = calculateMiniMaxM3Weight({
    ...common,
    enableSharedExpertTp: false,
  });
  const shardedShared = calculateMiniMaxM3Weight({
    ...common,
    enableSharedExpertTp: true,
  });

  assert.equal(replicatedShared.routedExpertScales, 403_439_616);
  assert.equal(replicatedShared.attentionScales, 52_936_704);
  assert.equal(replicatedShared.denseMlpScales, 5_308_416);
  assert.equal(replicatedShared.sharedExpertScales, 100_859_904);
  assert.equal(shardedShared.sharedExpertScales, 25_214_976);
  assert.equal(replicatedShared.sharedExpertScales / shardedShared.sharedExpertScales, 4);
  assert.equal(replicatedShared.total, 19_975_583_488);
  assert.equal(shardedShared.total, 17_479_300_864);
  assert.ok(Math.abs(replicatedShared.total / 1024 ** 3 - 18.60371) < 0.00001);
  assert.ok(Math.abs(shardedShared.total / 1024 ** 3 - 16.27887) < 0.00001);
});

test("index K stays replicated while index Q follows TP", () => {
  const tp1 = calculateMiniMaxM3Weight({
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 1,
    epSize: 2,
    enableSharedExpertTp: false,
  });
  const tp2 = calculateMiniMaxM3Weight({
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 2,
    epSize: 4,
    enableSharedExpertTp: false,
  });

  const mainAttentionScaleTp1 = 60 * 3_342_336;
  const mainAttentionScaleTp2 = 60 * 1_671_168;
  const indexScaleTp1 = tp1.attentionScales - mainAttentionScaleTp1;
  const indexScaleTp2 = tp2.attentionScales - mainAttentionScaleTp2;
  assert.equal(indexScaleTp1, 57 * (98_304 + 24_576));
  assert.equal(indexScaleTp2, 57 * (49_152 + 24_576));
});

const GIB = 1024 ** 3;

const v4Pro = {
  architecture: "deepseek-v4",
  vocabSize: 129280,
  totalLayers: 61,
  denseLayers: 0,
  moeLayers: 61,
  expertIntermediateSize: 3072,
  denseIntermediateSize: 3072,
  attentionHeads: 128,
  kvHeads: 1,
  headDim: 512,
  indexerHeads: 64,
  indexerHeadDim: 128,
  sharedExperts: 1,
  vocabPaddingSize: 64,
  qLoraRank: 1536,
  oGroups: 16,
  oLoraRank: 1024,
  hashLayers: 3,
  csaLayers: 29,
  hcaLayers: 32,
  slidingWindowLayers: 0,
  hcMult: 4,
};

const v4Flash = {
  ...v4Pro,
  totalLayers: 43,
  moeLayers: 43,
  expertIntermediateSize: 2048,
  denseIntermediateSize: 2048,
  attentionHeads: 64,
  qLoraRank: 1024,
  oGroups: 8,
  csaLayers: 20,
  hcaLayers: 21,
  slidingWindowLayers: 2,
};

function v4Weight(profile, hiddenSize, expertCount, tpSize, epSize, mtpLayers = 1) {
  return calculateDeepseekV4Weight({
    profile,
    hiddenSize,
    expertCount,
    topK: 6,
    tpSize,
    epSize,
    mtpLayers,
  });
}

test("FP8 block memory uses 128x128 e8m0 scales", () => {
  assert.deepEqual(fp8BlockMemory(256, 256), {
    payload: 65_536,
    scales: 4,
    total: 65_540,
  });
});

test("MXFP4 K32 memory stores two values per byte plus in/32 scales", () => {
  assert.deepEqual(mxfp4K32Memory(32, 32), {
    payload: 512,
    scales: 32,
    total: 544,
  });
});

test("DeepSeek V4-Pro TP8/DP8 matches RFC 18.52 GiB", () => {
  const result = v4Weight(v4Pro, 7168, 384, 8, 64);
  assert.ok(Math.abs(result.total / GIB - 18.52) < 0.02);
  assert.ok(Math.abs(result.routedExpertPayload / GIB - 11.25879) < 0.00001);
  assert.ok(Math.abs(result.routedExpertScales / GIB - 0.70367) < 0.00001);
});

test("DeepSeek V4-Flash TP8/DP8 matches RFC 4.32 GiB", () => {
  const result = v4Weight(v4Flash, 4096, 256, 8, 64);
  assert.ok(Math.abs(result.total / GIB - 4.32) < 0.02);
  assert.ok(Math.abs(result.routedExpertPayload / GIB - 2.01562) < 0.00001);
});

test("DeepSeek V4-Pro TP1/DP64 keeps routed experts and copies attention", () => {
  const tp8 = v4Weight(v4Pro, 7168, 384, 8, 64);
  const tp1 = v4Weight(v4Pro, 7168, 384, 1, 64);
  assert.equal(tp1.routedExpertPayload, tp8.routedExpertPayload);
  assert.equal(tp1.routedExpertScales, tp8.routedExpertScales);
  assert.equal(tp1.fusedWqaWkv, tp8.fusedWqaWkv);
  assert.equal(tp1.indexer, tp8.indexer);
  assert.equal(tp1.mhc, tp8.mhc);
  assert.equal(tp1.router, tp8.router);
  assert.ok(tp1.wqB > tp8.wqB);
  assert.ok(tp1.sharedExperts > tp8.sharedExperts);
  assert.ok(Math.abs(tp1.total / GIB - 39.29) < 0.02);
});

test("DeepSeek V4 Indexer stays replicated across TP", () => {
  const tp1 = v4Weight(v4Pro, 7168, 384, 1, 8);
  const tp8 = v4Weight(v4Pro, 7168, 384, 8, 64);
  assert.equal(tp1.indexer, tp8.indexer);
  assert.equal(tp1.compressorCsa, tp8.compressorCsa);
  assert.equal(tp1.compressorHca, tp8.compressorHca);
});

test("DeepSeek V4 full-model weights match RFC TP1/EP1 totals", () => {
  const pro = v4Weight(v4Pro, 7168, 384, 1, 1);
  const flash = v4Weight(v4Flash, 4096, 256, 1, 1);
  assert.ok(Math.abs(pro.total / GIB - 805.28) < 0.02);
  assert.ok(Math.abs(flash.total / GIB - 148.62) < 0.02);
});
