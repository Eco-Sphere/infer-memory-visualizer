import kvCacheCalculator from "kv-cache-calculator";
import kvModelData from "kv-cache-calculator/models";

export type CachePrecision = "bf16_fp16" | "fp8_int8" | "fp4_int4";

export const CACHE_PRECISIONS: Record<CachePrecision, { label: string; bytes: number }> = {
  bf16_fp16: { label: "BF16 / FP16", bytes: 2 },
  fp8_int8: { label: "FP8 / INT8", bytes: 1 },
  fp4_int4: { label: "FP4 / INT4", bytes: 0.5 },
};

export type KvCacheModelInfo = {
  layers: number;
  sparseLayers: number;
  kvHeads: number;
  headDim: number;
  indexHeadDim: number;
};

function findUpstreamModel(modelId: string) {
  return kvModelData.models.find((model) => model.id === modelId);
}

export function getKvCacheModelInfo(modelId: string): KvCacheModelInfo | undefined {
  const upstream = findUpstreamModel(modelId);
  if (!upstream) return undefined;
  const fields = upstream.fields;
  return {
    layers: Number(fields.num_hidden_layers),
    sparseLayers: Number(fields.sparse_attention_layers ?? fields.num_hidden_layers),
    kvHeads: Number(fields.num_key_value_heads),
    headDim: Number(fields.head_dim),
    indexHeadDim: Number(fields.index_head_dim),
  };
}

type KvCacheInput = {
  modelId: string;
  tokens: number;
  sequences: number;
  kvPrecision: CachePrecision;
  indexPrecision: CachePrecision;
  mtpLayers?: number;
  tpSize?: number;
};

export type KvCacheBreakdown = {
  kvCache: number;
  indexCache: number;
  total: number;
  kvBytesPerElement: number;
  indexBytesPerElement: number;
  kvCopies: number;
  effectiveKvLayers: number;
  effectiveIndexLayers: number;
};

export function calculateKvCache({
  modelId,
  tokens,
  sequences,
  kvPrecision,
  indexPrecision,
  mtpLayers = 0,
  tpSize = 1,
}: KvCacheInput): KvCacheBreakdown | null {
  const upstream = findUpstreamModel(modelId);
  const info = getKvCacheModelInfo(modelId);
  if (!upstream || !info) return null;
  const safeTokens = Math.max(0, Math.floor(tokens));
  const safeSequences = Math.max(0, Math.floor(sequences));
  const safeMtpLayers = Math.max(0, Math.floor(mtpLayers));
  const safeTp = Math.max(1, Math.floor(tpSize));
  const effectiveKvLayers = info.layers + safeMtpLayers;
  const effectiveIndexLayers = info.sparseLayers + safeMtpLayers;
  const kvBytesPerElement = (CACHE_PRECISIONS[kvPrecision] ?? CACHE_PRECISIONS.bf16_fp16).bytes;
  const indexBytesPerElement = (CACHE_PRECISIONS[indexPrecision] ?? CACHE_PRECISIONS.fp4_int4).bytes;
  // Standard GQA stores K and V separately and shards KV heads across TP ranks.
  const kvCopies = 2;

  if (safeTokens === 0 || safeSequences === 0) {
    return {
      kvCache: 0,
      indexCache: 0,
      total: 0,
      kvBytesPerElement,
      indexBytesPerElement,
      kvCopies,
      effectiveKvLayers,
      effectiveIndexLayers,
    };
  }

  // Delegate the per-token math to the upstream kv-cache-calculator engine.
  // MTP layers extend both the main KV layers and the sparse index layers.
  const fields: Record<string, unknown> = {
    ...upstream.fields,
    num_hidden_layers: info.layers + safeMtpLayers,
    sparse_attention_layers: info.sparseLayers + safeMtpLayers,
  };
  // Upstream pins the M3 indexer precision; the visualizer lets users pick it.
  delete fields.indexer_fixed_precision_id;
  const result = kvCacheCalculator.calculate(
    { ...upstream, fields },
    {
      tokens: safeTokens,
      sequences: safeSequences,
      tensorParallel: 1,
      precision: kvPrecision,
      indexerPrecision: indexPrecision,
    },
  );

  const kvCache = result.kvBytes / safeTp;
  // The MSA index cache is a single key head, replicated across TP ranks.
  const indexCache = result.indexerBytes;

  return {
    kvCache,
    indexCache,
    total: kvCache + indexCache,
    kvBytesPerElement,
    indexBytesPerElement,
    kvCopies,
    effectiveKvLayers,
    effectiveIndexLayers,
  };
}
