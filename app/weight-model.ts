import type { WeightProfile } from "./models";

export type Mxfp8MatrixMemory = {
  payload: number;
  scales: number;
  total: number;
};

export function mxfp8MatrixMemory(out: number, input: number): Mxfp8MatrixMemory {
  const payload = out * input;
  const scales = Math.ceil(out) * Math.ceil(input / 32);
  return { payload, scales, total: payload + scales };
}

export type MiniMaxM3WeightBreakdown = {
  routedExpertPayload: number;
  routedExpertScales: number;
  attentionPayload: number;
  attentionScales: number;
  attentionMetadata: number;
  attention: number;
  denseMlpPayload: number;
  denseMlpScales: number;
  denseMlp: number;
  sharedExpertPayload: number;
  sharedExpertScales: number;
  sharedExperts: number;
  router: number;
  norms: number;
  embedding: number;
  lmHead: number;
  misc: number;
  paddedVocab: number;
  total: number;
};

type MiniMaxM3WeightInput = {
  profile: WeightProfile;
  hiddenSize: number;
  expertCount: number;
  tpSize: number;
  epSize: number;
  enableSharedExpertTp: boolean;
};

const align = (value: number, boundary: number) =>
  Math.ceil(value / boundary) * boundary;

const sumLinearMemory = <T extends { payload: number; scales: number; total: number }>(
  matrices: T[],
) => matrices.reduce(
  (sum, matrix) => ({
    payload: sum.payload + matrix.payload,
    scales: sum.scales + matrix.scales,
    total: sum.total + matrix.total,
  }),
  { payload: 0, scales: 0, total: 0 },
);

const sumMatrices = (matrices: Mxfp8MatrixMemory[]) => sumLinearMemory(matrices);
const sumFp8Matrices = (matrices: Fp8BlockMemory[]) => sumLinearMemory(matrices);

export function calculateMiniMaxM3Weight({
  profile,
  hiddenSize: H,
  expertCount,
  tpSize: tp,
  epSize: ep,
  enableSharedExpertTp,
}: MiniMaxM3WeightInput): MiniMaxM3WeightBreakdown {
  const I = profile.expertIntermediateSize;
  const localExperts = expertCount / ep;

  const routedPerExpert = sumMatrices([
    mxfp8MatrixMemory(2 * I, H),
    mxfp8MatrixMemory(H, I),
  ]);
  const routedFactor = profile.moeLayers * localExperts;
  const routedExpertPayload = routedFactor * routedPerExpert.payload;
  const routedExpertScales = routedFactor * routedPerExpert.scales;

  const qRank = profile.attentionHeads / tp * profile.headDim;
  const kvRank = profile.kvHeads / tp * profile.headDim;
  const indexQRank = profile.indexerHeads / tp * profile.indexerHeadDim;
  // The index K projection produces one full index head and is replicated.
  const indexKRank = profile.indexerHeadDim;
  const attentionPerLayer = sumMatrices([
    mxfp8MatrixMemory(qRank, H),
    mxfp8MatrixMemory(kvRank, H),
    mxfp8MatrixMemory(kvRank, H),
    mxfp8MatrixMemory(H, qRank),
  ]);
  const indexerPerLayer = sumMatrices([
    mxfp8MatrixMemory(indexQRank, H),
    mxfp8MatrixMemory(indexKRank, H),
  ]);
  const attentionPayload = profile.totalLayers * attentionPerLayer.payload
    + profile.moeLayers * indexerPerLayer.payload;
  const attentionScales = profile.totalLayers * attentionPerLayer.scales
    + profile.moeLayers * indexerPerLayer.scales;
  const attentionMetadata = 119_808;
  const attention = attentionPayload + attentionScales + attentionMetadata;

  const denseRank = profile.denseIntermediateSize / tp;
  const densePerLayer = sumMatrices([
    mxfp8MatrixMemory(2 * denseRank, H),
    mxfp8MatrixMemory(H, denseRank),
  ]);
  const denseMlpPayload = profile.denseLayers * densePerLayer.payload;
  const denseMlpScales = profile.denseLayers * densePerLayer.scales;
  const denseMlp = denseMlpPayload + denseMlpScales;

  const sharedTp = enableSharedExpertTp ? tp : 1;
  const sharedRank = I / sharedTp;
  const sharedPerExpert = sumMatrices([
    mxfp8MatrixMemory(2 * sharedRank, H),
    mxfp8MatrixMemory(H, sharedRank),
  ]);
  const sharedFactor = profile.moeLayers * profile.sharedExperts;
  const sharedExpertPayload = sharedFactor * sharedPerExpert.payload;
  const sharedExpertScales = sharedFactor * sharedPerExpert.scales;
  const sharedExperts = sharedExpertPayload + sharedExpertScales;

  const router = profile.moeLayers * expertCount * (H + 1) * 4;
  const norms = 2_961_408;
  const paddedVocab = align(profile.vocabSize, profile.vocabPaddingSize);
  const embedding = paddedVocab / tp * H * 2;
  const lmHead = embedding;
  const misc = 256;
  const total = routedExpertPayload + routedExpertScales + attention + denseMlp
    + sharedExperts + router + norms + embedding + lmHead + misc;

  return {
    routedExpertPayload,
    routedExpertScales,
    attentionPayload,
    attentionScales,
    attentionMetadata,
    attention,
    denseMlpPayload,
    denseMlpScales,
    denseMlp,
    sharedExpertPayload,
    sharedExpertScales,
    sharedExperts,
    router,
    norms,
    embedding,
    lmHead,
    misc,
    paddedVocab,
    total,
  };
}

export type Fp8BlockMemory = {
  payload: number;
  scales: number;
  total: number;
};

export function fp8BlockMemory(out: number, input: number, block = 128): Fp8BlockMemory {
  const payload = out * input;
  const scales = Math.ceil(out / block) * Math.ceil(input / block);
  return { payload, scales, total: payload + scales };
}

export function mxfp4K32Memory(out: number, input: number): Fp8BlockMemory {
  const payload = (out * input) / 2;
  const scales = out * (input / 32);
  return { payload, scales, total: payload + scales };
}

export type DeepseekV4WeightBreakdown = {
  routedExpertPayload: number;
  routedExpertScales: number;
  sharedExpertPayload: number;
  sharedExpertScales: number;
  sharedExperts: number;
  router: number;
  fusedWqaWkv: number;
  wqB: number;
  woA: number;
  woB: number;
  compressorCsa: number;
  compressorHca: number;
  indexerWqB: number;
  indexerWeightsProj: number;
  indexerCompressor: number;
  indexer: number;
  attentionTp: number;
  attentionReplicated: number;
  attention: number;
  norms: number;
  mhc: number;
  embedding: number;
  lmHead: number;
  mtpPerLayer: number;
  mtpWeight: number;
  paddedVocab: number;
  total: number;
};

type DeepseekV4WeightInput = {
  profile: WeightProfile;
  hiddenSize: number;
  expertCount: number;
  topK: number;
  tpSize: number;
  epSize: number;
  mtpLayers?: number;
};

const bf16Bytes = (elements: number) => elements * 2;
const fp32Bytes = (elements: number) => elements * 4;

function compressorBytes(
  layers: number,
  coff: number,
  headDim: number,
  compressRatio: number,
  hiddenSize: number,
) {
  const fused = bf16Bytes(2 * coff * headDim * hiddenSize);
  const ape = fp32Bytes(compressRatio * coff * headDim);
  const norm = bf16Bytes(headDim);
  return layers * (fused + ape + norm);
}

export function isDeepseekV4Breakdown(
  breakdown: MiniMaxM3WeightBreakdown | DeepseekV4WeightBreakdown,
): breakdown is DeepseekV4WeightBreakdown {
  return "fusedWqaWkv" in breakdown;
}

export function calculateDeepseekV4Weight({
  profile,
  hiddenSize: H,
  expertCount: E,
  topK,
  tpSize: tp,
  epSize: ep,
  mtpLayers = 1,
}: DeepseekV4WeightInput): DeepseekV4WeightBreakdown {
  if (
    profile.architecture !== "deepseek-v4"
    || profile.qLoraRank == null
    || profile.oGroups == null
    || profile.oLoraRank == null
    || profile.hashLayers == null
    || profile.csaLayers == null
    || profile.hcaLayers == null
    || profile.hcMult == null
  ) {
    throw new Error("DeepSeek V4 weight profile is incomplete");
  }

  const I = profile.expertIntermediateSize;
  const L = profile.totalLayers;
  const localExperts = E / ep;
  const Irank = align(I, 128) / tp;
  const c = profile.headDim;
  const dc = profile.qLoraRank;
  const g = profile.oGroups;
  const dg = profile.oLoraRank;
  const heads = profile.attentionHeads;
  const indexHeads = profile.indexerHeads;
  const indexDim = profile.indexerHeadDim;
  const csaLayers = profile.csaLayers;
  const hcaLayers = profile.hcaLayers;
  const hashLayers = profile.hashLayers;
  const hcMult = profile.hcMult;
  const mixHc = (2 + hcMult) * hcMult;
  const hcDim = hcMult * H;
  const mtpCount = Math.max(0, Math.floor(mtpLayers));

  const routedPerExpert = sumFp8Matrices([
    mxfp4K32Memory(I, H),
    mxfp4K32Memory(I, H),
    mxfp4K32Memory(H, I),
  ]);
  const routedFactor = L * localExperts;
  const routedExpertPayload = routedFactor * routedPerExpert.payload;
  const routedExpertScales = routedFactor * routedPerExpert.scales;

  const sharedPerLayer = sumFp8Matrices([
    fp8BlockMemory(2 * Irank, H),
    fp8BlockMemory(H, Irank),
  ]);
  const sharedExpertPayload = L * profile.sharedExperts * sharedPerLayer.payload;
  const sharedExpertScales = L * profile.sharedExperts * sharedPerLayer.scales;
  const sharedExperts = sharedExpertPayload + sharedExpertScales;

  const router = L * E * H * 2
    + (L - hashLayers) * E * 4
    + hashLayers * profile.vocabSize * topK * 4;

  const fusedWqaWkv = L * fp8BlockMemory(dc + c, H).total;
  const qNorm = L * bf16Bytes(dc);
  const kvNorm = L * bf16Bytes(c);
  const wqB = L * fp8BlockMemory(heads / tp * c, dc).total;
  const woA = L * fp8BlockMemory(g / tp * dg, heads * c / g).total;
  const woB = L * fp8BlockMemory(H, g / tp * dg).total;
  const sink = L * (heads / tp) * 4;
  const compressorCsa = compressorBytes(csaLayers, 2, c, 4, H);
  const compressorHca = compressorBytes(hcaLayers, 1, c, 128, H);
  const indexerWqB = csaLayers * fp8BlockMemory(indexHeads * indexDim, dc).total;
  const indexerWeightsProj = csaLayers * bf16Bytes(indexHeads * H);
  const indexerCompressor = compressorBytes(csaLayers, 2, indexDim, 4, H);
  const indexer = indexerWqB + indexerWeightsProj + indexerCompressor;
  const attentionTp = wqB + woA + woB + sink;
  const attentionReplicated = fusedWqaWkv + qNorm + kvNorm + compressorCsa
    + compressorHca + indexer;
  const attention = attentionTp + attentionReplicated;

  const norms = (2 * L + 1) * bf16Bytes(H);
  const mhc = L * (
    2 * fp32Bytes(mixHc * hcDim)
    + 2 * fp32Bytes(mixHc)
    + 2 * fp32Bytes(3)
  ) + fp32Bytes(hcMult * hcDim) + fp32Bytes(hcMult) + fp32Bytes(1);

  const paddedVocab = align(profile.vocabSize, profile.vocabPaddingSize);
  const embedding = paddedVocab / tp * bf16Bytes(H);
  const lmHead = embedding;

  const swaAttention = fp8BlockMemory(dc + c, H).total
    + bf16Bytes(dc)
    + fp8BlockMemory(heads / tp * c, dc).total
    + bf16Bytes(c)
    + fp8BlockMemory(g / tp * dg, heads * c / g).total
    + fp8BlockMemory(H, g / tp * dg).total
    + (heads / tp) * 4;
  const mtpMhc = 2 * fp32Bytes(mixHc * hcDim) + 2 * fp32Bytes(mixHc) + 2 * fp32Bytes(3);
  const mtpPerLayer = routedPerExpert.total * localExperts
    + sharedPerLayer.total * profile.sharedExperts
    + E * H * 2 + E * 4
    + swaAttention
    + 2 * bf16Bytes(H)
    + mtpMhc
    + 2 * fp8BlockMemory(H, H).total
    + 3 * bf16Bytes(H)
    + fp32Bytes(hcMult * hcDim) + fp32Bytes(hcMult) + fp32Bytes(1);
  const mtpWeight = mtpCount * mtpPerLayer;

  const total = routedExpertPayload + routedExpertScales + sharedExperts + router
    + attention + norms + mhc + embedding + lmHead + mtpWeight;

  return {
    routedExpertPayload,
    routedExpertScales,
    sharedExpertPayload,
    sharedExpertScales,
    sharedExperts,
    router,
    fusedWqaWkv,
    wqB,
    woA,
    woB,
    compressorCsa,
    compressorHca,
    indexerWqB,
    indexerWeightsProj,
    indexerCompressor,
    indexer,
    attentionTp,
    attentionReplicated,
    attention,
    norms,
    mhc,
    embedding,
    lmHead,
    mtpPerLayer,
    mtpWeight,
    paddedVocab,
    total,
  };
}
