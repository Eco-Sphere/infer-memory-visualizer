import type { KvCacheInput, KvCacheBreakdown, KvCacheModelInfo } from "./kv-cache-model.ts";

// 官方配置及分配实现：stepfun-ai/Step-5-Preview，inference/model.py 与 kernel.py。
// DSA 保留完整 K/V，每 8 token 存一份 FP8 summary。
export const STEP5_CACHE_INFO: KvCacheModelInfo = {
  layers: 92, sparseLayers: 23, kvHeads: 4, headDim: 192, indexHeadDim: 256,
  source: "local",
};

export function calculateStep5Cache(input: KvCacheInput, kvBytes: number): KvCacheBreakdown {
  const tokens = Math.max(0, Math.floor(input.tokens));
  const sequences = Math.max(0, Math.floor(input.sequences));
  const mtp = Math.min(3, Math.max(0, Math.floor(input.mtpLayers ?? 0)));
  const tp = Math.max(1, Math.floor(input.tpSize ?? 1));
  // TP 超过 4 时，每卡仍需一组 KV；Index summary 是 MQA，在每个 rank 复制。
  const localHeads = Math.max(1, Math.ceil(4 / tp));
  const fullCapacity = Math.ceil(tokens / 16) * 16;
  const slidingLayers = 69 + mtp;
  const activeSequences = tokens > 0 ? sequences : 0;
  const kvCache = activeSequences * 2 * localHeads * 192 * kvBytes
    * (23 * fullCapacity + slidingLayers * 512);
  const indexCache = activeSequences * 23 * fullCapacity / 8 * 256;
  const mtpKvCache = activeSequences * mtp * 512 * 2 * localHeads * 192 * kvBytes;
  return {
    kvCache, indexCache, mtpKvCache, total: kvCache + indexCache,
    kvBytesPerElement: kvBytes, indexBytesPerElement: 1, kvCopies: 2,
    effectiveKvLayers: 92 + mtp, effectiveIndexLayers: 23,
    kvFormula: !activeSequences ? "0 B（tokens 或序列数为 0）" : `${sequences} × 2 × max(1, ceil(4 ÷ TP ${tp})) × 192 × ${kvBytes} B × (23 × ${fullCapacity} + ${slidingLayers} × 512)；DSA 容量 ceil(${tokens} ÷ 16) × 16，滑窗层 69 + MTP ${mtp}`,
    indexFormula: !activeSequences ? "0 B（tokens 或序列数为 0）" : `${sequences} × 23 × (${fullCapacity} ÷ 8) × 256 × 1 B，FP8 压缩 summary，TP 复制`,
  };
}
