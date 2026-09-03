# 参与贡献

感谢你帮助改进 Infer Memory Visualizer。无论是修正模型数据与计算逻辑、完善测试，还是优化界面与文档，我们都欢迎。

## 开始之前

- 先搜索已有的 Issue 和 Pull Request，避免重复工作。
- 如果计划添加大型功能或修改显存计算公式，请先创建 Issue，说明目标、方案和计算假设。
- 报告问题时，请提供模型、输入参数、预期结果、实际结果，以及相关截图或数据来源。

## 开发环境

需要安装：

- Node.js `>=22.13.0`
- npm

Fork 仓库后，将其克隆到本地并安装依赖：

```bash
git clone <你的 Fork 地址>
cd infer-memory-visualizer
npm install
npm run dev
```

开发服务器启动后会在终端输出本地访问地址。

## 项目结构

- `app/page.tsx`：计算器主界面与整体显存建模。
- `app/model-index.ts`：厂商 → 原始权重 → 量化权重的静态清单、owner 映射与 KV Cache 模型映射。
- `app/modelscope.ts`：ModelScope resolve 拉取（config + safetensors header + 分片定位）。
- `app/safetensors.ts`：safetensors header 解析（dtype/shape/字节数）。
- `app/weight-calc.ts`：张量归类与权重（全量/单卡）计算。
- `app/config-model.ts`：`config.json` → 统一结构参数。
- `app/kv-cache-model.ts`：KV Cache 计算引擎适配。
- `app/globals.css`：应用样式。
- `docs/legacy-minimax-weight-formula.md`：旧版 MiniMax 手写公式（已归档）。
- `tests/`：计算逻辑与渲染结果测试。
- `public/`：静态资源。

## 进行修改

基于最新的默认分支创建一个用途明确的分支：

```bash
git switch -c feat/简短描述
```

每个 Pull Request 应尽量只解决一个问题。请遵循仓库现有的 TypeScript、React 和 CSS 风格，避免夹带无关的格式化修改。

### 修改模型数据或计算逻辑

- 优先使用模型发布方的官方配置或文档作为依据。
- 结构参数动态读取 ModelScope 的 `config.json`，无需在代码中写死；模型清单（厂商/owner/量化变体）在 `app/model-index.ts` 中维护。
- 在代码或测试中明确单位、精度、对齐方式和并行切分假设。
- 为公式、边界情况和并行策略添加回归测试。
- 计算测试应尽量断言精确的整数字节数，仅在展示层转换为可读单位。

### 修改界面

- 同时检查桌面端和窄屏布局。
- 保持键盘可操作性和清晰可见的焦点状态。
- 如果修改了可见界面，请在 Pull Request 中提供修改前后的截图。

## 新增模型

新增模型前，请先确定支持范围：

- **基础支持**：完成模型注册、权重计算以及 KV/Index Cache 计算，并提供对应的显存明细与测试。
- **完整支持**：在基础支持之外，支持 O-Proj、Embedding、LM Head 和 Shared Expert 的独立 TP 切分，以及投机推理模型的显存计算。

本章节重点介绍如何实现基础支持。

### 1. 注册模型与权重清单

在 `app/model-index.ts` 的 `MODEL_INDEX` 中添加厂商条目：

- `brand`：一级显示名（厂商），如 `DeepSeek`；`Ling` 与 `Ring` 一级分开、owner 均为 `inclusionAI`。
- `owner`：该厂商原始权重的 ModelScope owner，如 `deepseek-ai`。
- `models`：原始权重列表，每个含 `name`（原始权重 repo 名）与 `quantized`（Eco-Tech 量化权重名列表，owner 统一为 `Eco-Tech`）。

量化权重需满足选型过滤：必须为 `w*a*` 量化，且不含 `xllm` / `mindie` / `310` / `eagle` / `pdmix`（含 `nopdmix`）/ `LAOS`。原始权重与量化权重按「原始权重字母序、量化子项在组内字母序」排列。

结构参数（hidden size、层数、专家数、TopK 等）**无需在代码中登记**，运行时从该权重的 `config.json` 自动提取（见 `app/config-model.ts`）。

### 2. 权重计算（逐张量）

权重显存从 safetensors header 逐张量计算：`numel × dtypeBytes(dtype)` 求和得到全量权重（精确）。单卡权重由 `app/weight-calc.ts` 的 `classifyTensor` 将张量归类为模块，再按通用规则切分：

- 路由专家 `÷ EP`；注意力 QKV `÷ attentionTp`、O-Proj `÷ oprojTp`；Dense MLP `÷ TP`；Embedding / LM Head `÷ embeddingTp / lmHeadTp`；Shared Expert、Router、Norm、视觉塔等复制。

量化 metadata（scale / offset / quant_bias）是独立张量，随主权重张量归类并计入。若新架构的张量命名无法被现有关键词正确归类，请在 `classifyTensor` 中补充关键词或新增模块并加测试；不要用「总参数量 × dtype ÷ 设备数」的均值近似。

#### 权重验证（必须）

1. **全量对账**：`TP=1`、`DP=1`（即 `EP=1`）时全量权重应与 `safetensors` header 逐张量求和一致，并与官方权重文件字节数对照。
2. **单卡切分验证**：按真实部署设置 TP/DP/EP 及各模块独立 TP，确认各模块按预期切分；能与推理框架打屏的单卡权重对照更佳。
3. 测试写在 `tests/weight-calc.test.mjs` 与 `tests/config-model.test.mjs`，断言精确字节数。由于推理框架可能只打印 GiB 或经过舍入的数值，无法获得精确字节数时，应根据其显示精度设置最小合理误差范围，并在测试和 Pull Request 中解释换算方式与误差来源。未经说明的偏差不能直接忽略。

此外，权重测试还应覆盖：

- `classifyTensor` 对多种架构张量名的归类。
- 新量化格式（如 `F8_E4M3`、`U8` scale、4-bit 打包）的字节数。

### 3. 完成基础 KV/Index Cache 计算

KV/Index Cache 可以通过以下两种方案接入：

#### 方案一：贡献给上游（推荐）

向 [`kv-cache-calculator`](https://github.com/Eco-Sphere/kv-cache-calculator) 添加模型配置和缓存计算支持。上游变更发布后，在本仓库中更新依赖版本，并在 `app/model-index.ts` 的 `KV_MODEL_ID` 中设置原始权重名到上游 model id 的映射。

推荐优先采用该方案，使模型数据和通用缓存公式能够由其他项目复用，并避免两个仓库维护重复配置。Pull Request 中请附上对应的上游提交或 Pull Request 链接。

#### 方案二：在本仓库直接实现

如果上游暂时不接受该模型、模型逻辑只适用于本项目，或上游发布周期无法满足需求，可以直接在本仓库补充模型配置和计算逻辑。实现应放在 `app/kv-cache-model.ts` 或拆分出的专用模块中，并保持现有调用接口和返回明细一致。

采用本地实现时，请在 Pull Request 中说明。如果后续上游已提供等价能力，应优先迁移回上游实现并删除重复逻辑。

无论采用哪种方案，都必须根据模型真实结构处理缓存行为。如果模型没有 Index Cache、K/V 并非各存一份、Index Cache 需要切分，或 KV head 采用特殊复制策略，请扩展相应配置和计算逻辑，不要直接套用现有假设。

在 `tests/kv-cache-model.test.mjs` 中至少覆盖：

- 层数、KV head 数、head dimension 等关键字段。
- 一个固定 tokens、sequences 和精度组合下的精确字节数。
- TP 切分行为，以及 KV 与 Index 精度独立选择的行为。
- 模型特有的缓存规则；如果模型没有 Index Cache，也应明确处理并测试其占用为零。

### 4. 支持独立 TP 切分（可选）

完整支持必须允许以下模块使用独立于主 TP 的切分大小：

- O-Proj
- Embedding
- LM Head
- Shared Expert

请在模型配置、`app/page.tsx` 和权重计算函数之间正确传递各模块的 TP 参数，并为每个模块校验维度是否可整除。改变某个独立 TP 时，只有对应模块及依赖该模块的总量应发生变化，其他模块的字节数必须保持不变。

在 `tests/weight-calc.test.mjs` 中分别覆盖上述模块，至少验证：

- 独立 TP 为默认值时，结果与主 TP 行为一致。
- 增大某个模块的 TP 后，该模块按预期切分。
- Routed Expert、Attention QKV 等无关模块不受影响。
- 不可整除的 TP 配置不会产生看似有效的结果。

### 5. 支持投机推理模型（可选）

完整支持还必须覆盖模型配套的投机推理结构，例如 MTP。不要只增加界面输入项；投机层新增的权重、KV Cache 和 Index Cache 都必须计入单卡总显存。

请根据官方模型结构确认每个投机层包含的模块、权重格式、切分策略和缓存层数；当前 MTP 层数通过界面输入影响 KV/Index Cache 的有效层数，投机层权重已随 safetensors 张量逐张量计入，无需额外标记。同时验证 MTP 层对主 KV 层数和 Index 层数的影响。如果投机模型不能用现有 MTP 公式描述，请扩展 KV Cache 适配逻辑，避免复用不适用的结构假设。

测试至少应验证：

- 投机层数为 `0` 时与基础模型结果一致。
- 每增加一层时，权重增量符合该投机层的真实结构。
- KV Cache 和 Index Cache 的有效层数及字节增量正确。
- 投机层遵循各模块约定的 TP/EP 切分方式。
- 多层投机配置按预期累加，且最终计入总显存。

### 6. 完成自检

新增模型的 Pull Request 应完成以下检查：

- 官方来源链接可以访问，并能支持提交中的关键参数和计算假设。
- 模型在选择器中归入正确家族，切换和重置行为正常。
- 权重与 KV/Index Cache 均有详细计算、界面展示和回归测试。
- 声明完整支持时，O-Proj、Embedding、LM Head 和 Shared Expert 均可独立配置 TP。
- 声明完整支持时，投机推理模型的权重和缓存占用已计入总显存。
- 非法 TP/EP 组合能够被正确限制或提示，不会产生看似有效的错误结果。
- 已运行 `npm run lint` 和 `npm test`，并在 Pull Request 中写明验证结果。

## 验证修改

提交前请运行与 Pull Request CI 相同的核心检查：

```bash
npm run lint
npm test
```

`npm test` 会先生成生产环境的 Pages 构建，再运行 Node.js 测试，因此可能比单独执行测试文件耗时更长。开发过程中可以运行特定测试：

```bash
node --test tests/weight-calc.test.mjs
node --test tests/config-model.test.mjs
node --test tests/kv-cache-model.test.mjs
```

如果修改会影响生产构建，请额外运行：

```bash
npm run build
```

## Commit 规范

Commit 信息应简短、明确，并使用祈使语气。仓库通常采用 Conventional Commits 前缀，例如：

```text
feat: 添加模型配置
fix: 修正 KV Cache 切分逻辑
test: 覆盖 FP4 scale metadata
docs: 完善贡献流程
```

常用前缀包括：

- `feat`：新增功能。
- `fix`：修复问题。
- `test`：新增或修改测试。
- `docs`：修改文档。
- `refactor`：不改变外部行为的代码重构。
- `chore`：构建、依赖或其他维护工作。

## Pull Request 要求

提交 Pull Request 时，请：

- 说明修改内容及原因。
- 关联对应的 Issue（如有）。
- 为模型参数或公式引用可靠来源。
- 说明验证方法与测试结果。
- 为可见的界面变化提供修改前后的截图。
- 明确列出已知限制或后续工作。

请求 Review 前，请确认 lint 和测试均已通过。Review 过程中可能需要补充测试用例，或提供更明确的模型计算依据。

## 安全问题

请勿通过公开 Issue 披露安全漏洞。请使用 GitHub Security Advisory，或通过仓库所有者提供的私密联系方式报告。

## 许可证

提交贡献即表示你同意按照本项目的 [MIT License](./LICENSE) 授权你的贡献。
