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
- `app/models.ts`：支持的模型参数及数据来源。
- `app/weight-model.ts`：模型权重显存计算。
- `app/kv-cache-model.ts`：KV Cache 计算引擎适配。
- `app/globals.css`：应用样式。
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
- 在 `app/models.ts` 中添加或更新模型的 `source` 链接。
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

### 1. 注册基础模型

在 `app/models.ts` 的 `BASE_MODELS` 中添加模型，至少填写：

- `id`：稳定且唯一的小写标识，通常使用模型仓库名的规范化形式。
- `family`：界面中的模型家族名称；同一系列应复用相同名称。
- `label`：界面展示名称。
- `hiddenSize`：隐藏层维度。
- `expertCount`：Routed Expert 总数。
- `source`：模型发布方的官方配置或文档链接，优先使用可直接查看的原始配置文件。

`topK` 默认值为 `8`。如果模型使用其他值，请在 `SPECIAL_TOP_K` 中显式配置。提交前应在本地界面确认模型家族、名称和基础参数展示正确，并检查常用 DP/TP 组合下的本地专家数是否合理。

### 2. 完成基础权重计算

权重必须按照目标推理框架实际加载到每个 Device 的张量逐项计算，不能使用“总参数量 × dtype ÷ Device 数”的平均值。实现前先确定对齐的推理框架、版本、量化配置和并行方案，并为模型提供 `weightProfile` 或独立的权重配置类型。

按以下步骤计算：

1. **列出全部权重模块**：至少检查 Routed Expert、Shared Expert、Dense MLP、Attention Q/K/V/O、Indexer、Router、Norm、Embedding、LM Head、投机层及模型特有模块，避免遗漏 checkpoint 中实际加载的参数和 buffer。
2. **确定单 Device 的实际 shape**：根据推理框架实现判断每个张量是按 TP、EP 或独立 TP 切分，还是在各 Rank 完整复制。`EP = TP × DP` 时，`DP` 只通过 EP 影响本地 Routed Expert 数，不能再次除到其他权重模块上。
3. **按实际 dtype 计算 payload**：对每个 Rank-local tensor 使用 `numel × bytesPerElement`。BF16/FP16、FP8、FP4、FP32、INT32 等必须分别处理，不能为整个模型假设单一精度。
4. **计算量化 metadata**：按照 checkpoint 的真实 block shape，对切分后的 tensor shape 计算 scale、zero point 等 metadata，并保留 `ceil` 和 padding。不能使用固定比例或其他模型的 block 公式近似。例如 MXFP8 `[1, 32]` 与 FP8 `[128, 128]` 的 scale 公式不同。
5. **计算复制项和辅助项**：Router bias、Norm、Compressor、Indexer、RoPE buffer、Hash 表等是否切分必须以推理框架代码为准；参数量较小也不能无依据省略。
6. **逐项求和**：内部统一使用 Bytes，分别返回 payload、metadata 和模块小计，最后得到单 Device 总权重；仅在展示层转换为 GiB。

模型常量至少应核对词表大小、层数、Dense/MoE 层分布、FFN 中间维度、Attention/KV/Indexer head、head dimension、专家数量、词表对齐方式和各模块权重格式。并行配置必须校验专家数、head 数、中间维度和 padded vocab 是否能被对应的 TP/EP 整除。

如果现有 `WeightProfile` 或 `app/weight-model.ts` 无法准确描述新架构，请新增配置类型和独立计算函数，不要通过填入近似参数强行复用现有公式。还应同步检查 `app/page.tsx` 中的参数合法性判断、输入项和明细公式是否适用于该模型。

可以参考以下设计与实现：

- [RFC #1：支持 MiniMax M3 权重占用计算](https://github.com/Eco-Sphere/infer-memory-visualizer/issues/1)：逐项计算 MXFP8 payload/scale、`EP = TP × DP`、Shared Expert、Attention、Embedding 和 LM Head。
- [RFC #2：支持 DeepSeek V4 权重显存计算](https://github.com/Eco-Sphere/infer-memory-visualizer/issues/2)：展示新架构如何定义独立 dtype、block scale、复制项、并行语义和验收数值。
- [PR #3：实现 DeepSeek V4 权重显存建模](https://github.com/Eco-Sphere/infer-memory-visualizer/pull/3)：RFC #2 对应的代码示例，包含模型配置、独立计算函数、界面接入和回归测试。

#### 权重验证（必须）

新增模型必须同时完成以下两种验证，缺少任意一种都不能视为完成基础支持：

1. **TP1/DP1 与总权重验证**：使用 `TP=1`、`DP=1`（即 `EP=1`）计算不切分的整模型权重，并与模型官方权重文件的总权重基准对照。该验证用于确认各模块均已计入，且权重格式、scale metadata 和固定开销没有遗漏或重复计算。
2. **实际部署与推理框架打屏验证**：按照真实部署方案设置 TP、DP 及各模块的独立 TP，计算单卡权重，并与推理框架启动或加载模型时实际打屏的单卡权重对照。Pull Request 必须注明推理框架及版本、硬件或设备数量、完整并行配置，并附上包含权重数值的日志片段或截图。

两组验证都必须在 `tests/weight-model.test.mjs` 中形成回归测试，并断言精确字节数。由于推理框架可能只打印 GiB 或经过舍入的数值，无法获得精确字节数时，应根据其显示精度设置最小合理误差范围，并在测试和 Pull Request 中解释换算方式与误差来源。未经说明的偏差不能直接忽略。

此外，权重测试还应覆盖：

- MODELS 中登记的模型参数是否正确，例如层数、隐藏维度、专家数和量化格式。
- 新模型特有的计算逻辑，例如新的量化格式、特殊 TP 切分等情况。

### 3. 完成基础 KV/Index Cache 计算

KV/Index Cache 可以通过以下两种方案接入：

#### 方案一：贡献给上游（推荐）

向 [`kv-cache-calculator`](https://github.com/Eco-Sphere/kv-cache-calculator) 添加模型配置和缓存计算支持。上游变更发布后，在本仓库中更新依赖版本，并在 `app/models.ts` 中设置对应的 `kvCacheModelId`。

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

在 `tests/weight-model.test.mjs` 中分别覆盖上述模块，至少验证：

- 独立 TP 为默认值时，结果与主 TP 行为一致。
- 增大某个模块的 TP 后，该模块按预期切分。
- Routed Expert、Attention QKV 等无关模块不受影响。
- 不可整除的 TP 配置不会产生看似有效的结果。

### 5. 支持投机推理模型（可选）

完整支持还必须覆盖模型配套的投机推理结构，例如 MTP。不要只增加界面输入项；投机层新增的权重、KV Cache 和 Index Cache 都必须计入单卡总显存。

请根据官方模型结构确认每个投机层包含的模块、权重格式、切分策略和缓存层数，并在模型配置中设置相应的能力标记（当前 MTP 使用 `supportsMtp`），同时验证 MTP 层对主 KV 层数和 Index 层数的影响。如果投机模型不能用现有 MTP 公式描述，请扩展配置类型和计算函数，避免复用不适用的结构假设。

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
node --test tests/weight-model.test.mjs
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
