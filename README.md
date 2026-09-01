# Infer Memory Visualizer

面向分布式 MoE 大模型推理的单卡显存估算与可视化工具。通过选择厂商与权重、调整并行策略、批量规模和缓存精度，可以快速拆解权重、KV Cache、激活值、通信缓冲区及运行时开销，辅助部署前的容量规划。

![Infer Memory Visualizer 界面预览](./public/og.png)

## 在线使用

[打开 Infer Memory Visualizer](https://eco-sphere.github.io/infer-memory-visualizer/)

所有计算都在浏览器中完成，不需要上传配置或推理数据；模型结构参数与权重张量清单会直接从 ModelScope 实时拉取。

## 使用方法

1. **一级选择厂商**：下拉默认显示 `厂商（owner）`，例如 `DeepSeek（deepseek-ai）`、`Ling（inclusionAI）`、`Ring（inclusionAI）`。
2. **二级选择权重**：列出该厂商下的**原始权重**（`deepseek-ai/DeepSeek-V3.1`）及其 **Eco-Tech 量化权重**（`Eco-Tech/DeepSeek-V3.1-w4a8-mtp-QuaRot`），原始权重按字母序、量化权重紧跟在所属原始权重之下按字母序排列。
3. 选中后页面自动拉取该权重的 `config.json`（结构参数）与全部 `safetensors` 分片头部（逐张量 `dtype × shape`），并计算单卡权重占用。
4. 配置 DP / TP / EP 及高级切分（Attention、O-Proj、Embedding、LM Head、Shared Expert 的独立 TP），设置 batch、KV Cache 精度与运行时开销，实时查看各部分占比与总显存。

## 功能

- 按「厂商 → 原始权重 / 量化权重」两级选择，覆盖 DeepSeek、GLM、Qwen、Kimi、MiniMax、Step、Ling、Ring、Hunyuan、InternVL 等厂商。
- 动态从 ModelScope 拉取结构参数与权重张量清单，无需内置写死的模型参数。
- 逐张量计算权重显存（全量精确），并按模块通用切分估算单卡权重。
- 已拉取的权重清单在会话内缓存：切换权重（含切回看过的权重）复用缓存，调整 DP/TP 等并行参数只重算切分、不重复请求；分片 header 并发拉取。
- 支持 `w*a*` 量化权重（含 INT8/INT4 及 `mxfp8`/`mxfp4` 的 payload + scale）与原始 BF16/FP8 权重。
- 估算激活值、HCCL buffer、ACLGraph、CANN/PTA/算子与 Device OS 开销。
- 对 `kv-cache-calculator` 已收录的模型，独立配置 KV Cache 与 Index Cache 的上下文长度、序列数和精度。
- 提供浅色/深色界面和响应式布局。

> [!NOTE]
> 权重显存对清单内所有权重均支持（逐张量精确）；KV Cache 与 Index Cache 仅对上游 `kv-cache-calculator` 已收录的模型（MiniMax M3/M2.7、DeepSeek V3/V3.2/V4、GLM-5/5.1/5.2、Kimi K2.5/K2.6、Qwen3 系列等）展示，其余模型暂不支持。

## 核心原理

### 权重显存：逐张量、从 safetensors header 计算

本项目不再按「模块 × 矩阵形状 × 精度」手推近似公式，而是直接读取权重文件的 **safetensors header**：

1. 读取文件头 8 字节的 little-endian u64，得到 header JSON 长度；
2. 读取 header JSON，其中每个张量给出 `{ dtype, shape, data_offsets }`；
3. 对每个张量计算字节数 `numel × dtypeBytes(dtype)` 并求和。

这样得到的是**完整模型权重的精确值**，对任何架构、任何命名都成立，无需维护手写公式。量化权重的 `scale`、`offset`、`quant_bias` 等辅助张量也一并作为独立张量计入。

**MXFP 格式的存储结构**（已实测）：

| 量化命名 | weight payload | scale |
|---|---|---|
| `w8a8-mxfp8` | `F8_E4M3`（1 B/元素） | `U8` [out, in/32] |
| `w4a4-mxfp4` | `U8` 打包（shape 已减半，等价 0.5 B/元素） | `U8` [out, in/32] |

因为 payload 与 scale 一开始就被拆成独立张量存在 header 里，所以字节数仍是 `numel × dtypeBytes` 直接求和，无需额外推理 block scale。

**单卡权重**按张量所属模块应用通用切分规则（近似，后续可逐架构细化）：

- 路由专家 `÷ EP`；
- 注意力 QKV `÷ attentionTp`，O-Proj `÷ oprojTp`；
- Dense MLP `÷ TP`，Embedding / LM Head `÷ embeddingTp / lmHeadTp`；
- Shared Expert、Router、Norm、视觉塔等复制（`÷ 1`）。

### 权重清单来源

量化权重清单来自 ModelScope 上的 Eco-Tech 仓库，经过选型过滤后静态维护在 [`app/model-index.ts`](./app/model-index.ts) 中：只保留 `w*a*` 量化，并排除 `xllm`、`mindie`、`310`、`eagle`、`pdmix`/`nopdmix`、`LAOS` 等非通用或硬件专属版本。原始权重的 owner 通过厂商前缀映射（如 `DeepSeek-*`、`GLM-*`、`Ling-*`/`Ring-*`）。

### 其余开销

- KV Cache 基算复用 [`kv-cache-calculator`](https://github.com/Eco-Sphere/kv-cache-calculator)。
- 激活值与 HCCL buffer 根据 token 数、并行规模、专家数量和最大 batch size 估算。
- ACLGraph、CANN 及算子开销允许手动配置；Device OS 当前按固定值（4.25 GiB）计入。

## 计算范围

```text
单卡显存 = 权重 + KV/Index Cache + 激活值 + HCCL buffer
         + ACLGraph + CANN/PTA/算子 + Device OS
```

本工具结果可用于开发阶段验证显存占用，以及部署阶段估算最大并发。实际占用还会受到推理框架版本和算子实现的影响，上线前请使用目标环境进行实测。

## 本地开发

### 环境要求

- Node.js `>=22.13.0`
- npm

### 启动项目

```bash
git clone https://github.com/Eco-Sphere/infer-memory-visualizer.git
cd infer-memory-visualizer
npm install
npm run dev
```

开发服务器启动后会在终端输出本地访问地址。项目基于 React、TypeScript 和 vinext 构建。

## 常用命令

```bash
npm run dev          # 启动开发服务器
npm run lint         # 运行 ESLint
npm test             # 构建 Pages 产物并运行全部测试
npm run build        # 创建生产构建
npm run build:pages  # 创建 GitHub Pages 静态产物
```

开发计算逻辑时，也可以单独运行测试文件：

```bash
node --test tests/weight-calc.test.mjs      # 张量分类与权重切分
node --test tests/config-model.test.mjs     # config.json 结构参数提取
node --test tests/model-index.test.mjs      # 模型清单数据完整性
node --test tests/kv-cache-model.test.mjs   # KV Cache 计算适配
```

## 项目结构

```text
app/
├── page.tsx             # 计算器界面与整体显存建模
├── model-index.ts       # 厂商 → 原始权重 → 量化权重的静态清单与 owner 映射
├── modelscope.ts        # ModelScope resolve 拉取（config + safetensors header + 分片定位）
├── safetensors.ts       # safetensors header 解析（dtype/shape/字节数）
├── weight-calc.ts       # 张量归类与权重（全量/单卡）计算
├── config-model.ts      # config.json → 统一结构参数
├── kv-cache-model.ts    # KV/Index Cache 计算适配
└── globals.css          # 全局样式
tests/                   # 计算逻辑与渲染测试
docs/
└── legacy-minimax-weight-formula.md  # 旧版 MiniMax 手写公式（已归档）
scripts/build-pages.mjs  # GitHub Pages 构建脚本
worker/                  # Cloudflare Worker 入口
```

## 数据与精度

- 结构参数与权重张量**动态来自 ModelScope** 的 `config.json` 与 `safetensors` header，不在代码中写死模型参数。
- 权重显存以 safetensors header 为权威来源，全量值精确；单卡切分为通用近似，详见上文「核心原理」。
- 模型权重清单（厂商/owner/量化变体）静态维护在 [`app/model-index.ts`](./app/model-index.ts)，新增权重需同步补充过滤规则与回归测试。
- 旧版手写的 MiniMax 权重公式已归档至 [`docs/legacy-minimax-weight-formula.md`](./docs/legacy-minimax-weight-formula.md)，仅作数据逻辑与计算公式参考。

如果你发现模型参数、计算假设或结果存在问题，欢迎提交 [Issue](https://github.com/Eco-Sphere/infer-memory-visualizer/issues)。请附上输入配置、预期结果、实际结果和可验证的数据来源。

## 参与贡献

欢迎贡献模型支持、计算修正、测试、界面优化和文档改进。开发流程及 Pull Request 要求请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 致谢

- [`kv-cache-calculator`](https://github.com/Eco-Sphere/kv-cache-calculator)：KV Cache 占用计算工具。
- [`vinext`](https://github.com/cloudflare/vinext)：基于 Vite 的 Next.js 兼容运行时。

## License

本项目采用 [MIT License](./LICENSE) 开源。