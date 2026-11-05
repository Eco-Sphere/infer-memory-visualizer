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
  /** 未传时计算完整 checkpoint；界面按启用的 MTP 层筛选权重。 */
  mtp?: { layers: number; mainLayers: number };
  /** Step-5 GQA 与 DSA provider group 的复制/切分布局。 */
  step5?: boolean;
  tp: number;
  ep: number;
  attentionTp: number;
  oprojTp: number;
  embeddingTp: number;
  lmHeadTp: number;
  sharedExpertTp: number;
};

export type WeightFormulaGroup = {
  name: string;
  dtype: string;
  shape: number[];
  count: number;
  bytesPerElement: number;
  divisor: number;
  parallelism: string;
  full: number;
  perDevice: number;
};

export function weightGroupFormula(group: WeightFormulaGroup): string {
  const shape = group.shape.length ? group.shape.join(" × ") : "1";
  return `${group.count} × (${shape}) × ${group.bytesPerElement} B ÷ ${group.parallelism} ${group.divisor} ÷ 1024³`;
}

export type ModuleBreakdown = {
  full: number;
  perDevice: number;
  tensors: number;
  groups: WeightFormulaGroup[];
};

export type WeightResult = {
  fullWeight: number;
  perDeviceWeight: number;
  breakdown: Record<TensorModule, ModuleBreakdown>;
};

const VISION = ["vision", "visual", "projector", "patch_merge", "vae", "vision_model"];
const EMBEDDING = ["embed_tokens", "word_embedding", "wte."];
const LM_HEAD = ["lm_head", "final_fc", "score_head"];
const SHARED_EXPERT = ["shared_expert", "share_expert", "shared_mlp"];
const ROUTER = ["router", "expert_bias", "e_score", "tid2eid", ".gate.weight", ".gate.bias"];
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
  if (includes(name, EMBEDDING) || /(^|\.)(embed|tok_emb)\./i.test(name)) return "embedding";
  if (includes(name, LM_HEAD) || /(^|\.)head\./i.test(name)
    || /(^|\.)shared_head\.(head|output)\./i.test(name)) return "lm_head";
  // Qwen 的共享专家输出门控是独立的标量门控，不随共享专家 TP 切分。
  if (includes(name, ["shared_expert_gate"])) return "router";
  if (includes(name, SHARED_EXPERT)) return "shared_expert";
  // Kimi-K3 latent MoE 的公共投影/Norm 位于专家集合外，不按 EP 切分。
  if (/(^|\.)routed_expert_(down|up)_proj\./i.test(name)) return "dense_mlp";
  if (/(^|\.)routed_expert_norm\./i.test(name)) return "norm";
  // 专家集合内的张量先归专家；集合外的 expert_bias 是 Router 的校正偏置。
  if (/(^|\.)experts?\./i.test(name)) return "routed_expert";
  if (includes(name, ROUTER)) return "router";
  if (name.toLowerCase().includes("expert")) return "routed_expert";
  if (includes(name, NORM)) return "norm";
  // Step 等模型将打包的路由专家命名为 moe.{gate,up,down}_proj，
  // 没有 experts 路径；Router/Norm/共享专家需先排除。
  if (/(^|[._])moe(?=[._]|$)/i.test(name)) return "routed_expert";
  if (includes(name, ATTENTION)) return "attention";
  if (includes(name, DENSE_MLP)) return "dense_mlp";
  return "other";
}

function isOproj(name: string): boolean {
  // Ling/Ring 将 Attention 输出投影命名为 attention.dense。
  return includes(name, OPROJ) || /(^|\.)attention\.dense\./i.test(name);
}

function newBreakdown(): Record<TensorModule, ModuleBreakdown> {
  const empty = () => ({ full: 0, perDevice: 0, tensors: 0, groups: [] });
  return {
    routed_expert: empty(),
    shared_expert: empty(),
    dense_mlp: empty(),
    attention: empty(),
    embedding: empty(),
    lm_head: empty(),
    router: empty(),
    norm: empty(),
    vision: empty(),
    other: empty(),
  };
}

/** 识别追加在主干后的层、独立 MTP 模块，以及各 MTP 层共用的投影。 */
function mtpTensorGroup(name: string, mainLayers: number): { family: string; index: number } | "shared" | null {
  if (includes(name, VISION)) return null;
  const named = name.match(/(?:^|\.)(mtp|mtp_layers|nextn_predict_layers)(?:\.layers)?\.(\d+)(?:\.|$)/i);
  if (named) return { family: named[1].toLowerCase(), index: Number(named[2]) };
  if (/(?:^|\.)(mtp|mtp_layers|nextn_predict_layers)(?:\.|$)/i.test(name)) return "shared";
  const layer = name.match(/(?:^|\.)layers\.(\d+)(?:\.|$)/i);
  if (mainLayers > 0 && layer && Number(layer[1]) >= mainLayers) {
    return { family: "appended", index: Number(layer[1]) };
  }
  return null;
}

function selectMtpTensors(tensors: WeightTensor[], selection: NonNullable<WeightParallel["mtp"]>): WeightTensor[] {
  const enabled = Number.isFinite(selection.layers) ? Math.max(0, Math.floor(selection.layers)) : 0;
  const entries = tensors.map((tensor) => ({ tensor, group: mtpTensorGroup(tensor.name, selection.mainLayers) }));
  const families = new Map<string, number[]>();
  for (const { group } of entries) {
    if (!group || group === "shared") continue;
    if (!families.has(group.family)) families.set(group.family, []);
    families.get(group.family)!.push(group.index);
  }
  // 按实际存在的层编号排序，兼容从 0 或 1 开始的独立 MTP 层。
  const ranks = new Map([...families].map(([family, indices]) => [
    family, new Map([...new Set(indices)].sort((a, b) => a - b).map((index, rank) => [index, rank])),
  ]));
  return entries.filter(({ group }) => {
    if (!group) return true;
    if (group === "shared") return enabled > 0;
    return ranks.get(group.family)!.get(group.index)! < enabled;
  }).map(({ tensor }) => tensor);
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
        if (p.step5) {
          if (/\.sparse_indexer_(k|z)\.|\.ssmax_s$/i.test(name)) return 1;
          if (/\.(k_proj|v_proj|sparse_indexer_q|sparse_indexer_w)\./i.test(name)) return Math.min(attentionTp, 4);
        }
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

  const parallelismFor = (module: TensorModule, name: string): string => {
    switch (module) {
      case "routed_expert": return "EP";
      case "shared_expert": return "Shared Expert TP";
      case "attention":
        if (p.step5) {
          if (/\.sparse_indexer_(k|z)\.|\.ssmax_s$/i.test(name)) return "复制";
          if (/\.(k_proj|v_proj|sparse_indexer_q|sparse_indexer_w)\./i.test(name)) return "KV/Provider TP（最多 4）";
        }
        return isOproj(name) ? "O-Proj TP" : "Attention TP";
      case "dense_mlp": return "TP";
      case "embedding": return "Embedding TP";
      case "lm_head": return "LM Head TP";
      default: return "复制";
    }
  };

  const breakdown = newBreakdown();
  const groups = new Map<string, WeightFormulaGroup>();
  let fullWeight = 0;

  for (const t of p.mtp ? selectMtpTensors(tensors, p.mtp) : tensors) {
    const tensorModule = classifyTensor(t.name);
    const bytes = t.bytes || 0;
    fullWeight += bytes;
    const b = breakdown[tensorModule];
    b.full += bytes;
    b.tensors += 1;
    const divisor = divisorFor(tensorModule, t.name);
    b.perDevice += bytes / divisor;

    // 只合并重复层/专家编号；不同投影、dtype、shape 和切分规则分别展示。
    const name = t.name.replace(/\.\d+(?=\.|$)/g, ".*");
    const parallelism = parallelismFor(tensorModule, t.name);
    const numel = t.shape.reduce((product, dim) => product * dim, 1);
    const bytesPerElement = numel ? bytes / numel : 0;
    const key = JSON.stringify([tensorModule, name, t.dtype, t.shape, bytesPerElement, divisor, parallelism]);
    let group = groups.get(key);
    if (!group) {
      group = { name, dtype: t.dtype, shape: [...t.shape], count: 0, bytesPerElement,
        divisor, parallelism, full: 0, perDevice: 0 };
      groups.set(key, group);
      b.groups.push(group);
    }
    group.count += 1;
    group.full += bytes;
    group.perDevice += bytes / divisor;
  }

  const perDeviceWeight = Object.values(breakdown).reduce((s, b) => s + b.perDevice, 0);
  return { fullWeight, perDeviceWeight, breakdown };
}