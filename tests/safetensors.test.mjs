import assert from "node:assert/strict";
import test from "node:test";

import { dtypeBytes, parseSafetensorsHeader } from "../app/safetensors.ts";

function buildHeaderBytes(header) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const buf = new ArrayBuffer(8 + json.byteLength);
  const view = new DataView(buf);
  view.setBigUint64(0, BigInt(json.byteLength), true);
  new Uint8Array(buf, 8, json.byteLength).set(json);
  return buf;
}

test("dtypeBytes 覆盖常见 dtype", () => {
  assert.equal(dtypeBytes("F32"), 4);
  assert.equal(dtypeBytes("BF16"), 2);
  assert.equal(dtypeBytes("I8"), 1);
  assert.equal(dtypeBytes("F8_E4M3"), 1);
  assert.equal(dtypeBytes("I4"), 0.5);
  assert.equal(dtypeBytes("UNKNOWN"), 0);
});

test("解析包含元数据与多个张量的 header", () => {
  const header = {
    __metadata__: { format: "pt" },
    "model.embed.weight": { dtype: "BF16", shape: [200, 64], data_offsets: [0, 25600] },
    "model.layer.q_proj.weight": { dtype: "I8", shape: [64, 64], data_offsets: [25600, 29696] },
    "model.layer.q_proj.weight_scale": { dtype: "F32", shape: [64], data_offsets: [29696, 29952] },
  };
  const parsed = parseSafetensorsHeader(buildHeaderBytes(header));
  assert.equal(parsed.metadata?.["format"], "pt");
  assert.equal(parsed.tensors.length, 3);

  const embed = parsed.tensors.find((t) => t.name === "model.embed.weight");
  assert.ok(embed);
  assert.equal(embed.numel, 200 * 64);
  assert.equal(embed.bytes, 200 * 64 * 2); // BF16

  const q = parsed.tensors.find((t) => t.name === "model.layer.q_proj.weight");
  assert.equal(q.numel, 64 * 64);
  assert.equal(q.bytes, 64 * 64); // I8

  const scale = parsed.tensors.find((t) => t.name === "model.layer.q_proj.weight_scale");
  assert.equal(scale.bytes, 64 * 4); // F32
});

test("header 声明长度超过提供字节时报错", () => {
  const json = new TextEncoder().encode(JSON.stringify({ a: { dtype: "F32", shape: [1], data_offsets: [0, 4] } }));
  const buf = new ArrayBuffer(8 + json.byteLength);
  new DataView(buf).setBigUint64(0, BigInt(json.byteLength + 100), true);
  assert.throws(() => parseSafetensorsHeader(buf), /声明/);
});

test("不足 8 字节报错", () => {
  assert.throws(() => parseSafetensorsHeader(new ArrayBuffer(4)), /不足 8 字节/);
});