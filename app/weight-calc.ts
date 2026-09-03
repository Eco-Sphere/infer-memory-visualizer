// 张量级权重显存计算：从 safetensors header 解析出的张量清单，按模块归类后
// 计算全量权重与单卡（按 TP/EP 切分后）权重。
//
// 全量权重（TP=EP=1）为 numel × dtype 字节数之和，与模块分类无关，绝对精确；
// 单卡权重按张量所属模块应用通用切分规则（expert 按 EP、embedding/LM Head 按 TP、
// attention 的 QKV 按 attentionTp、O-Proj 按 oprojTp、dense 按 TP、其余复制），
// 面向不同架构为通用近似，后续可逐架构细化。

import type { SafetensorInfo } from "./safetensors";

export type WeightTensor = Pick<SafetensorInfo, "name" | "dtype" | "shape" | "bytes">;

export type TensorModule =
  | "routed_expert"
  | "shared_expert"
  | "dense_mlp"
  | "attention"
  | "embedding"
  | "lm_head"
  | "router"
  | "norm"
  | "vision"
  | "other";

export type WeightParallel = {
  tp: number;
  ep: number;
  attentionTp: number;
  oprojTp: number;
  embeddingTp: number;
  lmHeadTp: number;
  sharedExpertTp: number;
};

export type ModuleBreakdown = {
  full: number;
  perDevice: number;
  tensors: number;
};

export type WeightResult = {
  fullWeight: number;
  perDeviceWeight: number;
  breakdown: Record<TensorModule, ModuleBreakdown>;
};

const VISION = ["vision", "visual", "projector", "patch_merge", "vae", "vision_model"];
const EMBEDDING = ["embed_tokens", "word_embedding", "wte."];
const LM_HEAD = ["lm_head", "final_fc", "score_head"];
const SHARED_EXPERT = ["shared_expert", "shared_experts"];
const ROUTER = ["router", "e_score", "tid2eid", ".gate.weight", ".gate.bias"];
const NORM = [
  "layernorm", "rmsnorm", "norm.weight", "norm.", "norm1", "norm2", "q_norm",
  "k_norm", "kv_norm", "ln_1", "ln_2", "final_norm",
];
const ATTENTION = [
  "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj", "wq", "wkv",
  "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "index", "linear_attn",
  "mamba", "a_log", "conv1d", "dt_bias", "mixer",
];
const DENSE_MLP = [
  "gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h",
  "fc1", "fc2", "linear_fc1", "linear_fc2", "mlp", "ffn",
];
const OPROJ = ["o_proj", "out_proj", "wo_a", "wo_b"];

function includes(name: string, words: string[]): boolean {
  const n = name.toLowerCase();
  return words.some((w) => n.includes(w));
}

/** 将张量名归类到模块类型。量化辅助张量（weight_scale 等）跟随主权重归类。 */
export function classifyTensor(name: string): TensorModule {
  if (includes(name, VISION)) return "vision";
  if (includes(name, EMBEDDING)) return "embedding";
  if (includes(name, LM_HEAD)) return "lm_head";
  if (includes(name, SHARED_EXPERT)) return "shared_expert";
  if (name.toLowerCase().includes("expert")) return "routed_expert";
  if (includes(name, ROUTER)) return "router";
  if (includes(name, NORM)) return "norm";
  if (includes(name, ATTENTION)) return "attention";
  if (includes(name, DENSE_MLP)) return "dense_mlp";
  return "other";
}

function isOproj(name: string): boolean {
  return includes(name, OPROJ);
}

function newBreakdown(): Record<TensorModule, ModuleBreakdown> {
  const empty: ModuleBreakdown = { full: 0, perDevice: 0, tensors: 0 };
  return {
    routed_expert: { ...empty },
    shared_expert: { ...empty },
    dense_mlp: { ...empty },
    attention: { ...empty },
    embedding: { ...empty },
    lm_head: { ...empty },
    router: { ...empty },
    norm: { ...empty },
    vision: { ...empty },
    other: { ...empty },
  };
}

export function calculateWeightFromTensors(
  tensors: WeightTensor[],
  p: WeightParallel,
): WeightResult {
  const safe = (v: number, dflt: number) => (Number.isFinite(v) && v > 0 ? v : dflt);
  const tp = safe(p.tp, 1);
  const ep = safe(p.ep, 1);
  const attentionTp = safe(p.attentionTp, tp);
  const oprojTp = safe(p.oprojTp, tp);
  const embeddingTp = safe(p.embeddingTp, tp);
  const lmHeadTp = safe(p.lmHeadTp, tp);
  const sharedExpertTp = safe(p.sharedExpertTp, 1);

  const divisorFor = (module: TensorModule, name: string): number => {
    switch (module) {
      case "routed_expert":
        return ep;
      case "shared_expert":
        return sharedExpertTp;
      case "attention":
        return isOproj(name) ? oprojTp : attentionTp;
      case "dense_mlp":
        return tp;
      case "embedding":
        return embeddingTp;
      case "lm_head":
        return lmHeadTp;
      default:
        return 1;
    }
  };

  const breakdown = newBreakdown();
  let fullWeight = 0;

  for (const t of tensors) {
    const tensorModule = classifyTensor(t.name);
    const bytes = t.bytes || 0;
    fullWeight += bytes;
    const b = breakdown[tensorModule];
    b.full += bytes;
    b.tensors += 1;
    b.perDevice += bytes / divisorFor(tensorModule, t.name);
  }

  const perDeviceWeight = Object.values(breakdown).reduce((s, b) => s + b.perDevice, 0);
  return { fullWeight, perDeviceWeight, breakdown };
}