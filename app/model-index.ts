// 静态模型清单索引：一级按厂商（family/品牌）收束，二级为原始权重及其 Eco-Tech 量化变体。
// 由 scripts 生成脚本从 ModelScope 实测清单加工而来；数据面向 w*a* 量化权重，
// 已过滤 xllm / mindie / 310 / eagle / pdmix / nopdmix / LAOS / 非 w*a* 量化。

export type RawModel = {
  /** 原始权重名（ModelScope repo 后半段），如 "DeepSeek-V3.2" */
  name: string;
  /** 该原始权重下的 Eco-Tech 量化权重名（owner 均为 Eco-Tech） */
  quantized: string[];
};

export type Vendor = {
  /** 一级显示名，如 "DeepSeek" */
  brand: string;
  /** 原始权重的 ModelScope owner，如 "deepseek-ai" */
  owner: string;
  models: RawModel[];
};

/** 所有量化权重的统一 owner */
export const QUANT_OWNER = "Eco-Tech";

export const MODEL_INDEX: Vendor[] = [
  {
    brand: "DeepSeek",
    owner: "deepseek-ai",
    models: [
      { name: "DeepSeek-Math-V2", quantized: ["DeepSeek-Math-V2-w8a8-mtp-QuaRot"] },
      { name: "DeepSeek-R1-0528", quantized: ["DeepSeek-R1-0528-w4a8-mtp-QuaRot", "DeepSeek-R1-0528-w4a8c8-mtp-QuaRot", "DeepSeek-R1-0528-w8a8-mtp-QuaRot", "DeepSeek-R1-0528-w8a8c8-mtp-QuaRot"] },
      { name: "DeepSeek-V3", quantized: ["DeepSeek-V3-w4a8-mtp-QuaRot", "DeepSeek-V3-w4a8c8-mtp-QuaRot", "DeepSeek-V3-w8a8-mtp-QuaRot", "DeepSeek-V3-w8a8c8-mtp-QuaRot"] },
      { name: "DeepSeek-V3-0324", quantized: ["DeepSeek-V3-0324-w4a8-mtp-QuaRot", "DeepSeek-V3-0324-w4a8c8-mtp-QuaRot", "DeepSeek-V3-0324-w8a8-mtp-QuaRot", "DeepSeek-V3-0324-w8a8c8-mtp-QuaRot"] },
      { name: "DeepSeek-V3.1", quantized: ["DeepSeek-V3.1-w4a8-mtp-QuaRot", "DeepSeek-V3.1-w4a8c8-mtp-QuaRot", "DeepSeek-V3.1-w8a8-mtp-QuaRot", "DeepSeek-V3.1-w8a8c8-mtp-QuaRot"] },
      { name: "DeepSeek-V3.1-Terminus", quantized: ["DeepSeek-V3.1-Terminus-w4a8-mtp-QuaRot", "DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot", "DeepSeek-V3.1-Terminus-w8a8-mtp-QuaRot", "DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot"] },
      { name: "DeepSeek-V3.2", quantized: ["DeepSeek-V3.2-w8a8-mtp-QuaRot"] },
      { name: "DeepSeek-V3.2-Exp", quantized: ["DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot"] },
      { name: "DeepSeek-V3.2-Speciale", quantized: ["DeepSeek-V3.2-Speciale-w8a8-mtp-QuaRot"] },
      { name: "DeepSeek-V4-Flash", quantized: ["DeepSeek-V4-Flash-DSpark-w8a8", "DeepSeek-V4-Flash-w8a8-mtp"] },
      { name: "DeepSeek-V4-Flash-0731", quantized: ["DeepSeek-V4-Flash-0731-w8a8"] },
      { name: "DeepSeek-V4-Pro", quantized: ["DeepSeek-V4-Pro-w4a8-mtp"] },
      { name: "DeepSeek-V4-Pro-0813", quantized: ["DeepSeek-V4-Pro-0813-w4a8"] },
    ],
  },
  {
    brand: "GLM",
    owner: "ZhipuAI",
    models: [
      { name: "GLM-4.7", quantized: ["GLM-4.7-W8A8", "GLM-4.7-W8A8-floatmtp"] },
      { name: "GLM-5", quantized: ["GLM-5-w4a8", "GLM-5-w4a8-mtp-QuaRot", "GLM-5-w8a8"] },
      { name: "GLM-5.1", quantized: ["GLM-5.1-w4a4c8-mxfp4", "GLM-5.1-w4a8", "GLM-5.1-w8a8", "GLM-5.1-w8a8c8", "GLM-5.1-w8a8c8-MTP"] },
      { name: "GLM-5.2", quantized: ["GLM-5.2-w4a8", "GLM-5.2-w4a8c8", "GLM-5.2-w8a8", "GLM-5.2-w8a8c8"] },
      { name: "GLM-5.3", quantized: ["GLM-5.3-w8a8", "GLM-5.3-w8a8c8"] },
    ],
  },
  {
    brand: "Hy",
    owner: "Tencent-Hunyuan",
    models: [
      { name: "Hy3", quantized: ["Hy3-w8a8"] },
      { name: "Hy4-preview", quantized: ["Hy4-preview-w8a8"] },
    ],
  },
  {
    brand: "InternVL",
    owner: "OpenGVLab",
    models: [
      { name: "InternVL3_5-241B-A28B", quantized: ["InternVL3_5-241B-A28B-w8a8"] },
      { name: "InternVL3_5-38B", quantized: ["InternVL3_5-38B-w8a8"] },
    ],
  },
  {
    brand: "Kimi",
    owner: "moonshotai",
    models: [
      { name: "KIMI-k2-Instruct-0905", quantized: ["KIMI-k2-Instruct-0905-W4A8-QuaRot", "KIMI-k2-Instruct-0905-W8A8-QuaRot"] },
      { name: "KIMI-k2-Thinking", quantized: ["KIMI-k2-Thinking-W8A8-QuaRot"] },
      { name: "Kimi-K2.5", quantized: ["Kimi-K2.5-w4a8"] },
      { name: "Kimi-K2.6", quantized: ["Kimi-K2.6-w4a8"] },
      { name: "Kimi-K2.7-Code", quantized: ["Kimi-K2.7-Code-w4a8"] },
      { name: "Kimi-K3", quantized: ["Kimi-K3-w4a8"] },
    ],
  },
  {
    brand: "Ling",
    owner: "inclusionAI",
    models: [
      { name: "Ling-2.5-1T", quantized: ["Ling-2.5-1T-w4a8"] },
      { name: "Ling-2.6-1T", quantized: ["Ling-2.6-1T-w4a8"] },
      { name: "Ling-3.0-flash", quantized: ["Ling-3.0-flash-w4a8"] },
    ],
  },
  {
    brand: "MiniMax",
    owner: "MiniMax",
    models: [
      { name: "MiniMax-M2.5", quantized: ["MiniMax-M2.5-w8a8-QuaRot"] },
      { name: "MiniMax-M2.7", quantized: ["MiniMax-M2.7-w8a8-QuaRot"] },
      { name: "MiniMax-M3", quantized: ["MiniMax-M3-w8a8", "MiniMax-M3-w8a8-0626"] },
    ],
  },
  {
    brand: "Qwen",
    owner: "Qwen",
    models: [
      { name: "Qwen2.5-VL-72B-Instruct", quantized: ["Qwen2.5-VL-72B-Instruct-w8a8"] },
      { name: "Qwen2.5-VL-7B-Instruct", quantized: ["Qwen2.5-VL-7B-Instruct-w8a8"] },
      { name: "Qwen3-235B-A22B", quantized: ["Qwen3-235B-A22B-w8a8-QuaRot"] },
      { name: "Qwen3-235B-A22B-Instruct-2507", quantized: ["Qwen3-235B-A22B-Instruct-2507-w8a8-QuaRot", "Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot"] },
      { name: "Qwen3-30B-A3B", quantized: ["Qwen3-30B-A3B-w8a8"] },
      { name: "Qwen3-30B-A3B-Thinking-2507", quantized: ["Qwen3-30B-A3B-Thinking-2507-w8a8"] },
      { name: "Qwen3-32B", quantized: ["Qwen3-32B-w8a8c8"] },
      { name: "Qwen3-Coder-30B-A3B-Instruct", quantized: ["Qwen3-Coder-30B-A3B-Instruct-w8a8"] },
      { name: "Qwen3-Coder-480B-A35B-Instruct", quantized: ["Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"] },
      { name: "Qwen3-Next-80B-A3B-Instruct", quantized: ["Qwen3-Next-80B-A3B-Instruct-w8a8-mtp"] },
      { name: "Qwen3-VL-235B-A22B-Instruct", quantized: ["Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot"] },
      { name: "Qwen3-VL-30B-A3B-Instruct", quantized: ["Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot"] },
      { name: "Qwen3-VL-32B-Instruct", quantized: ["Qwen3-VL-32B-Instruct-w8a8-QuaRot"] },
      { name: "Qwen3-VL-8B-Instruct", quantized: ["Qwen3-VL-8B-Instruct-w8a8-QuaRot"] },
      { name: "Qwen3.5-122B-A10B", quantized: ["Qwen3.5-122B-A10B-w8a8-mtp"] },
      { name: "Qwen3.5-27B", quantized: ["Qwen3.5-27B-w8a8-mtp"] },
      { name: "Qwen3.5-35B-A3B", quantized: ["Qwen3.5-35B-A3B-w8a8-mtp"] },
      { name: "Qwen3.5-397B-A17B", quantized: ["Qwen3.5-397B-A17B-w4a4-mxfp4", "Qwen3.5-397B-A17B-w4a8-mtp", "Qwen3.5-397B-A17B-w8a8-mtp", "Qwen3.5-397B-A17B-w8a8-mxfp8"] },
      { name: "Qwen3.6-27B", quantized: ["Qwen3.6-27B-w8a8", "Qwen3.6-27B-w8a8-mxfp8"] },
      { name: "Qwen3.6-35B-A3B", quantized: ["Qwen3.6-35B-A3B-w8a8", "Qwen3.6-35B-A3B-w8a8-mxfp8"] },
      { name: "Qwen3.8-2.4T-A95B", quantized: ["Qwen3.8-2.4T-A95B-w8a8"] },
      { name: "Qwen3.8-27B", quantized: ["Qwen3.8-27B-w8a8", "Qwen3.8-27B-w8a8-mxfp8"] },
    ],
  },
  {
    brand: "Ring",
    owner: "inclusionAI",
    models: [
      { name: "Ring-2.5-1T", quantized: ["Ring-2.5-1T-w4a8"] },
    ],
  },
  {
    brand: "Step",
    owner: "stepfun-ai",
    models: [
      { name: "Step-3.5-Flash", quantized: ["Step-3.5-Flash-w8a8-mtp"] },
      { name: "Step-3.7-Flash", quantized: ["Step-3.7-Flash-w8a8-mtp"] },
    ],
  },
];

/**
 * 原始权重名 → kv-cache-calculator 的 model id 映射。
 * 仅覆盖上游 52 个模型里能精确对应（或日期版本可近似）的条目；
 * 未列出的原始权重无 KV Cache 公式，前端相应隐藏 KV Cache 部分。
 */
export const KV_MODEL_ID: Record<string, string> = {
  "MiniMax-M3": "minimax-m3",
  "MiniMax-M2.7": "minimax-m2.7",
  "MiniMax-M2.5": "minimax-m2.5",
  "DeepSeek-V4-Pro": "deepseek-v4-pro",
  "DeepSeek-V4-Pro-0813": "deepseek-v4-pro",
  "DeepSeek-V4-Flash": "deepseek-v4-flash",
  "DeepSeek-V4-Flash-0731": "deepseek-v4-flash",
  "DeepSeek-V3.2": "deepseek-v3.2",
  "DeepSeek-V3.2-Exp": "deepseek-v3.2",
  "DeepSeek-V3.2-Speciale": "deepseek-v3.2",
  "DeepSeek-V3": "deepseek-v3",
  "DeepSeek-V3-0324": "deepseek-v3",
  "DeepSeek-R1-0528": "deepseek-r1",
  "GLM-5": "glm-5",
  "GLM-5.1": "glm-5.1",
  "GLM-5.2": "glm-5.2",
  "Kimi-K2.5": "kimi-k2.5",
  "Kimi-K2.6": "kimi-k2.6",
  "Qwen3-32B": "qwen3-32b",
  "Qwen3-30B-A3B": "qwen3-30b-a3b",
  "Qwen3-235B-A22B": "qwen3-235b-a22b",
  "Qwen3.5-397B-A17B": "qwen3.5-397b-a17b",
  "Qwen3.5-122B-A10B": "qwen3.5-122b-a10b",
  "Qwen3.5-35B-A3B": "qwen3.5-35b-a3b",
  "Qwen3.5-27B": "qwen3.5-27b",
  "Qwen3.6-27B": "qwen3.6-27b",
  "Qwen3.6-35B-A3B": "qwen3.6-35b-a3b",
};

/** 由原始权重名取 KV Cache 模型 id；无公式时返回 undefined。 */
export function kvCacheModelIdOf(rawName: string): string | undefined {
  return KV_MODEL_ID[rawName];
}

export function sortKey(name: string): string {
  return name.toLowerCase().replace(/_/g, "-").replace(/^minimax(\d)/, "minimax-$1");
}

export function findVendor(brand: string): Vendor | undefined {
  return MODEL_INDEX.find((v) => v.brand === brand);
}

/** 由任意权重名（raw 或量化）判定其厂商品牌前缀。 */
export function brandOf(name: string): string | undefined {
  const n = sortKey(name);
  const order = ["deepseek","glm","qwen","kimi","minimax","step","ling","ring","hy","internvl","wan"];
  const found = order.find((p) => n.startsWith(p));
  return found ? (MODEL_INDEX.find((v) => v.brand.toLowerCase() === found)?.brand ?? undefined) : undefined;
}
