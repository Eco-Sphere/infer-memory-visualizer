import assert from "node:assert/strict";
import test from "node:test";

import {
  MODEL_INDEX,
  QUANT_OWNER,
  KV_MODEL_ID,
  kvCacheModelIdOf,
} from "../app/model-index.ts";

test("MODEL_INDEX 厂商唯一且有序", () => {
  const brands = MODEL_INDEX.map((v) => v.brand);
  assert.equal(new Set(brands).size, brands.length, "厂商 brand 应唯一");
  assert.deepEqual([...brands].sort(), brands, "厂商应按字母序");
});

test("每个厂商至少有一个原始权重，且原始权重名唯一", () => {
  for (const v of MODEL_INDEX) {
    assert.ok(v.models.length > 0, `${v.brand} 应有原始权重`);
    const names = v.models.map((m) => m.name);
    assert.equal(new Set(names).size, names.length, `${v.brand} 下原始权重名应唯一`);
  }
});

test("量化权重全部通过选型过滤（w*a*，且无 xllm/mindie/310/eagle/pdmix/laos）", () => {
  const banned = ["xllm", "mindie", "310", "eagle", "pdmix", "laos"];
  for (const v of MODEL_INDEX) {
    for (const m of v.models) {
      for (const q of m.quantized) {
        assert.match(q, /w\d+a\d+/i, `${q} 应是 w*a* 量化`);
        const lower = q.toLowerCase();
        for (const b of banned) {
          assert.ok(!lower.includes(b), `${q} 不应包含 ${b}`);
        }
        assert.ok(q.startsWith(m.name), `${q} 应以其原始权重 ${m.name} 为前缀`);
      }
    }
  }
});

test("Ling 与 Ring 一级分开，但同属 inclusionAI", () => {
  const ling = MODEL_INDEX.find((v) => v.brand === "Ling");
  const ring = MODEL_INDEX.find((v) => v.brand === "Ring");
  assert.ok(ling);
  assert.ok(ring);
  assert.equal(ling.owner, "inclusionAI");
  assert.equal(ring.owner, "inclusionAI");
});

test("KIMI 与 Kimi 归并到同一 Kimi 厂商", () => {
  const kimi = MODEL_INDEX.find((v) => v.brand === "Kimi");
  assert.ok(kimi);
  const names = kimi.models.map((m) => m.name);
  assert.ok(names.some((n) => /^KIMI/i.test(n)), "应包含 KIMI 大写命名的原始权重");
  assert.ok(names.some((n) => /^Kimi/i.test(n)), "应包含 Kimi 首字母大写的原始权重");
  assert.equal(kimi.owner, "moonshotai");
});

test("KV_MODEL_ID 映射键均存在于某些厂商的原始权重中", () => {
  const allRaw = new Set(MODEL_INDEX.flatMap((v) => v.models.map((m) => m.name)));
  for (const raw of Object.keys(KV_MODEL_ID)) {
    assert.ok(allRaw.has(raw), `KV 映射键 ${raw} 应存在于原始权重清单`);
  }
  assert.equal(kvCacheModelIdOf("MiniMax-M3"), "minimax-m3");
  assert.equal(kvCacheModelIdOf("Step-3.5-Flash"), undefined);
});

test("量化权重统一 owner 为 Eco-Tech", () => {
  const allQuant = MODEL_INDEX.flatMap((v) => v.models.flatMap((m) => m.quantized));
  assert.ok(allQuant.length > 0);
  assert.equal(QUANT_OWNER, "Eco-Tech");
});