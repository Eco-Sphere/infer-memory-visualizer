export type ModelConfig = {
  id: string;
  family: string;
  label: string;
  hiddenSize: number;
  expertCount: number;
  topK: number;
  source?: string;
  weightProfile?: WeightProfile;
  kvCacheModelId?: string;
  supportsMtp?: boolean;
};

export type MxWeightFormat = "mxfp8" | "mxfp4";

export type WeightProfile = {
  vocabSize: number;
  totalLayers: number;
  sparseAttentionLayers?: number;
  denseLayers: number;
  moeLayers: number;
  expertIntermediateSize: number;
  denseIntermediateSize: number;
  attentionHeads: number;
  kvHeads: number;
  headDim: number;
  valueHeadDim?: number;
  indexerHeads: number;
  indexerHeadDim: number;
  sharedExperts: number;
  vocabPaddingSize: number;
  moeWeightFormat?: MxWeightFormat;
  sharedExpertWeightFormat?: MxWeightFormat;
  sharedKv?: boolean;
  qkvProjection?: "qkvo" | "qk";
};

// Curated from the same official Hugging Face config sources used by
// kv-cache-calculator. This calculator intentionally lists MoE models only.
const BASE_MODELS: Omit<ModelConfig, "topK">[] = [
  { id: "deepseek-v4-pro", family: "DeepSeek", label: "DeepSeek V4 Pro", hiddenSize: 7168, expertCount: 384, source: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json" },
  { id: "deepseek-v4-flash", family: "DeepSeek", label: "DeepSeek V4 Flash", hiddenSize: 4096, expertCount: 256, source: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json" },
  { id: "deepseek-v3.2", family: "DeepSeek", label: "DeepSeek V3.2", hiddenSize: 7168, expertCount: 256, source: "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json" },
  { id: "deepseek-v3", family: "DeepSeek", label: "DeepSeek V3", hiddenSize: 7168, expertCount: 256, source: "https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json" },
  { id: "deepseek-r1", family: "DeepSeek", label: "DeepSeek R1", hiddenSize: 7168, expertCount: 256, source: "https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json" },
  { id: "glm-5.2", family: "GLM", label: "GLM-5.2", hiddenSize: 6144, expertCount: 256, source: "https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json" },
  { id: "glm-5.1", family: "GLM", label: "GLM-5.1", hiddenSize: 6144, expertCount: 256, source: "https://huggingface.co/zai-org/GLM-5.1/raw/main/config.json" },
  { id: "glm-5", family: "GLM", label: "GLM-5", hiddenSize: 6144, expertCount: 256, source: "https://huggingface.co/zai-org/GLM-5/raw/main/config.json" },
  { id: "kimi-k2.6", family: "Kimi", label: "Kimi K2.6", hiddenSize: 7168, expertCount: 384, source: "https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json" },
  { id: "kimi-k2.5", family: "Kimi", label: "Kimi K2.5", hiddenSize: 7168, expertCount: 384, source: "https://huggingface.co/moonshotai/Kimi-K2.5/raw/main/config.json" },
  { id: "qwen3.6-35b-a3b", family: "Qwen3.6", label: "Qwen3.6-35B-A3B", hiddenSize: 2048, expertCount: 256, source: "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json" },
  { id: "qwen3.5-397b-a17b", family: "Qwen3.5", label: "Qwen3.5-397B-A17B", hiddenSize: 4096, expertCount: 512, source: "https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json" },
  { id: "qwen3.5-122b-a10b", family: "Qwen3.5", label: "Qwen3.5-122B-A10B", hiddenSize: 3072, expertCount: 256, source: "https://huggingface.co/Qwen/Qwen3.5-122B-A10B/raw/main/config.json" },
  { id: "qwen3.5-35b-a3b", family: "Qwen3.5", label: "Qwen3.5-35B-A3B", hiddenSize: 2048, expertCount: 256, source: "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json" },
  { id: "qwen3-235b-a22b", family: "Qwen3", label: "Qwen3-235B-A22B", hiddenSize: 4096, expertCount: 128, source: "https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json" },
  { id: "qwen3-30b-a3b", family: "Qwen3", label: "Qwen3-30B-A3B", hiddenSize: 2048, expertCount: 128, source: "https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json" },
  { id: "gemma-4-26b-a4b", family: "Gemma", label: "Gemma 4 26B-A4B", hiddenSize: 2816, expertCount: 128, source: "https://huggingface.co/google/gemma-4-26B-A4B/raw/main/config.json" },
  { id: "cohere-command-a-plus-05-2026", family: "Cohere", label: "Command A Plus 05-2026", hiddenSize: 4096, expertCount: 128, source: "https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16/raw/main/config.json" },
  { id: "mimo-v2.5-pro", family: "MiMo", label: "MiMo-V2.5-Pro", hiddenSize: 6144, expertCount: 384, source: "https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro/raw/main/config.json" },
  { id: "mimo-v2.5", family: "MiMo", label: "MiMo-V2.5", hiddenSize: 4096, expertCount: 256, source: "https://huggingface.co/XiaomiMiMo/MiMo-V2.5/raw/main/config.json" },
  { id: "minimax-m3", family: "MiniMax", label: "MiniMax M3", hiddenSize: 6144, expertCount: 128, source: "https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8/blob/main/config.json" },
  { id: "minimax-m2.7", family: "MiniMax", label: "MiniMax M2.7", hiddenSize: 3072, expertCount: 256, source: "https://huggingface.co/MiniMaxAI/MiniMax-M2.7/raw/main/config.json" },
  { id: "minimax-m2.5", family: "MiniMax", label: "MiniMax M2.5", hiddenSize: 3072, expertCount: 256, source: "https://huggingface.co/MiniMaxAI/MiniMax-M2.5/raw/main/config.json" },
  { id: "minimax-m2.1", family: "MiniMax", label: "MiniMax M2.1", hiddenSize: 3072, expertCount: 256, source: "https://huggingface.co/MiniMaxAI/MiniMax-M2.1/raw/main/config.json" },
  { id: "minimax-m2", family: "MiniMax", label: "MiniMax M2", hiddenSize: 3072, expertCount: 256, source: "https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json" },
];

const SPECIAL_TOP_K: Record<string, number> = {
  "deepseek-v4-pro": 6,
  "deepseek-v4-flash": 6,
  "qwen3.5-397b-a17b": 10,
  "minimax-m3": 4,
};

export const MODELS: ModelConfig[] = BASE_MODELS.map((model) => ({
  ...model,
  topK: SPECIAL_TOP_K[model.id] ?? 8,
  weightProfile: model.id === "minimax-m3" ? {
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
    moeWeightFormat: "mxfp8",
    qkvProjection: "qkvo",
  } : undefined,
  kvCacheModelId: model.id === "minimax-m3" ? "minimax-m3" : undefined,
}));

export const DEFAULT_MODEL_ID = "minimax-m3";
