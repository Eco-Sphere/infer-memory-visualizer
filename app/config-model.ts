// 从模型 config.json 提取统一的结构参数。
// 不同架构字段名与嵌套层级不同（MiniMax 用 text_config 嵌套，DeepSeek/GLM/Qwen
// 多为顶层平铺），这里 flatten 后按候选字段名匹配。

export type ModelStructure = {
  hiddenSize: number;
  numLayers: number;
  attentionHeads: number;
  kvHeads: number;
  headDim: number;
  vocabSize: number;
  /** 0 表示 dense（无 routed expert） */
  expertCount: number;
  topK: number;
  sharedExperts: number;
  intermediateSize: number;
  moeIntermediateSize: number;
  architectures: string[];
  modelType: string;
  hasVision: boolean;
};

function flatten(
  obj: unknown,
  out: Record<string, unknown>,
  prefix = "",
): void {
  if (Array.isArray(obj)) return;
  if (obj && typeof obj === "object") {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      const key = prefix ? `${prefix}.${k}` : k;
      out[key.toLowerCase()] = v;
      flatten(v, out, key);
    }
  }
}

function num(out: Record<string, unknown>, candidates: string[]): number {
  for (const c of candidates) {
    const v = out[c.toLowerCase()];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
      return Number(v);
    }
  }
  return 0;
}

/**
 * 优先取 text_config 下的语言部分字段；没有 text_config 时回退顶层。
 * 做法：先整体 flatten（含 text_config.*），候选名里把 text_config 前缀放前面。
 */
function langNum(
  flat: Record<string, unknown>,
  plainCandidates: string[],
  aliases: string[] = plainCandidates,
): number {
  // text_config.<field> 优先，其次顶层 <field>，再其次别名。
  const all: string[] = [];
  for (const c of plainCandidates) all.push(`text_config.${c}`, c);
  all.push(...aliases);
  return num(flat, all);
}

export function extractModelStructure(config: Record<string, unknown>): ModelStructure {
  const flat: Record<string, unknown> = {};
  flatten(config, flat);

  const architectures = (() => {
    const v = flat["architectures"] ?? config.architectures;
    return Array.isArray(v) ? (v as string[]) : [];
  })();
  const modelType = String(flat["model_type"] ?? config.model_type ?? "");

  const hiddenSize = langNum(flat, ["hidden_size"]);
  const numLayers = langNum(flat, ["num_hidden_layers"], ["num_layers", "n_layer"]);
  const attentionHeads = langNum(flat, ["num_attention_heads"], ["n_head"]);
  const kvHeads = langNum(
    flat,
    ["num_key_value_heads"],
    ["num_kv_heads", "n_head_kv", "kv_heads"],
  );
  const headDim = langNum(flat, ["head_dim"], ["head_size"]);
  const vocabSize = langNum(flat, ["vocab_size"], ["padded_vocab_size"]);
  const intermediateSize = langNum(flat, ["intermediate_size"], ["ffn_hidden_size"]);
  const moeIntermediateSize = langNum(
    flat,
    ["moe_intermediate_size"],
    ["expert_intermediate_size", "expert_hidden_size"],
  );
  // expert 数量字段名分歧最大
  const expertCount = langNum(
    flat,
    ["num_experts", "num_local_experts", "n_routed_experts"],
    ["n_moe_experts", "moe_num_experts", "num_moe_experts"],
  );
  const topK = langNum(
    flat,
    ["num_experts_per_tok", "num_experts_per_token"],
    ["top_k", "moe_topk", "num_selected_experts"],
  );
  const sharedExperts = langNum(flat, ["n_shared_experts"], ["num_shared_experts"]);

  // 判定是否含视觉塔
  const hasVision = Object.keys(flat).some(
    (k) =>
      k.includes("vision_config") ||
      k.includes("vision_tower") ||
      k.includes(".visual.") ||
      k.includes("patch_embedding") ||
      k.includes("multi_modal_projector"),
  );

  return {
    hiddenSize,
    numLayers,
    attentionHeads,
    kvHeads,
    headDim,
    vocabSize,
    expertCount,
    topK,
    sharedExperts,
    intermediateSize,
    moeIntermediateSize,
    architectures,
    modelType,
    hasVision,
  };
}