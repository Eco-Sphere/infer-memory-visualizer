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

开发服务器启动后会在终端输出本地访问地址。本地开发不需要 `wrangler.jsonc`。

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
