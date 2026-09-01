# 旧版 MiniMax 权重手写公式（已归档）

> 本文件是旧版 `app/weight-model.ts` 的归档说明。该文件里硬编码了 MiniMax-M3
> 的权重计算公式，现已被 `app/weight-calc.ts` 的**逐张量（per-tensor）引擎**
> 取代。保留此文档仅为说明其当年的数据逻辑与计算公式，不再在代码中使用。

## 为什么被取代

旧公式按“模块 × 矩阵形状 × 精度”手推每个模块的字节数，属于**近似估算**：

- 旧公式 MiniMax-M3 总权重 ≈ **442 GiB**（假定 MXFP8 全量化）。
- 新引擎直接从 `safetensors` header 逐张量读 `dtype × shape`，得到真实值
  **405.74 GiB**（w8a8 实际存储），张量总数 67,912，与仓库一致。

`header` 是权威来源：真实存储可能是 W8A8（INT8 + scale），而非旧公式假定的
MXFP8（FP8 + [1,32] block scale）。逐张量求和天然精确，无需再维护手工公式。

## 核心计算公式（存档）

### MXFP 矩阵内存

```ts
// MXFP8：每个元素 1 字节 payload；每 [1, 32] block 1 字节 scale。
function mxfp8MatrixMemory(out, input) {
  const payload = out * input;                       // 1 B / element
  const scales = Math.ceil(out) * Math.ceil(input / 32); // 1 B / block
  return { payload, scales, total: payload + scales };
}

// MXFP4：每 2 个元素打包 1 字节 payload（0.5 B / element）；scale 同 MXFP8。
function mxMatrixMemory(out, input, format) {
  const payload = format === "mxfp4"
    ? Math.ceil(out * input / 2)      // 4-bit packed
    : out * input;
  const scales = Math.ceil(out) * Math.ceil(input / 32);
  return { payload, scales, total: payload + scales };
}
```

### 各模块公式（MiniMax-M3，`H` = hidden size，`I` = expert intermediate）

- **Routed Expert**（每 expert 2 个矩阵：`2I×H` 的 gate-up 合并 + `H×I` 的 down）：
  `routedExpertPayload = moeLayers × (expertCount ÷ EP) × 3 × H × I × 每元素字节`
- **Attention QKV / O-proj / Indexer**：按 `heads ÷ TP × headDim` 计算 rank，
  再 `rank × H × 字节`；O-proj 单独按 `oprojTp` 切；QKV 按 `attentionTp` 切。
- **Dense MLP**（`denseLayers` 层）：`2 × (denseIntermediate ÷ TP) × H + H × (denseIntermediate ÷ TP)`。
- **Shared Expert**：与 routed 同形，`moeLayers × sharedExperts`，按 `sharedExpertTp` 切。
- **Router**：`moeLayers × expertCount × (H + 1) × 4 B`（FP32）。
- **Norm**：`(4 × totalLayers + 1) × H × 2 B`（BF16）。
- **Embedding / LM Head**：`align(vocabSize, 64) ÷ TP × H × 2 B`。
- **MTP**：复用 sparse MoE block + projection `H×2H` + norm，`mtpLayers` 层。

这些公式的**数据逻辑**（哪些模块、按什么 rank 切分、量化 payload/scale 如何算）
仍可参考，但数值应一律以 safetensors header 的真实 `dtype × shape` 求和为准。