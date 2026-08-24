# Infer Memory Visualizer

面向分布式 MoE 大模型推理的单卡显存估算与可视化工具。通过调整模型、并行策略、批量规模和缓存精度，可以快速拆解权重、KV Cache、激活值、通信缓冲区及运行时开销，辅助部署前的容量规划。

![Infer Memory Visualizer 界面预览](./public/og.png)

## 在线使用

[打开 Infer Memory Visualizer](https://eco-sphere.github.io/infer-memory-visualizer/)

所有计算都在浏览器中完成，不需要上传配置或推理数据。

## 功能

- 按模型家族选择主流 MoE 模型，并展示隐藏维度、专家数和 Top-K 等参数。
- 配置 DP、TP、EP，以及 Attention、O-Proj、Embedding、LM Head 和 Shared Expert 的独立 TP 策略。
- 估算单卡权重占用，并支持自定义 Routed Expert、Shared Expert、Attention、Dense MLP、Embedding 和 LM Head的TP切分。
- 独立配置 KV Cache 与 Index Cache 的上下文长度、序列数和精度。
- 估算激活值、HCCL buffer、ACLGraph、CANN/PTA/算子与 Device OS 开销。
- 支持 MTP 层数建模，并实时展示各部分占比和总显存。
- 提供浅色/深色界面和响应式布局。

> [!NOTE]
> 模型列表覆盖多种 MoE 架构，但精细的权重与 KV Cache 拆解目前仅支持部分模型；当前默认且完整支持的是 MiniMax M3。

## 计算范围

最终结果按单设备计算，包含：

```text
单卡显存 = 权重 + KV/Index Cache + 激活值 + HCCL buffer
         + ACLGraph + CANN/PTA/算子 + Device OS
```

其中：

- 权重计算考虑 MXFP8/MXFP4 payload、分块 scale metadata，以及不同组件的并行切分方式。
- KV Cache 的基础计算复用 [`kv-cache-calculator`](https://github.com/Eco-Sphere/kv-cache-calculator)，主 KV 按 Attention TP 切分，Index Cache 按当前模型假设保留副本。
- 激活值与 HCCL buffer 根据 token 数、并行规模、专家数量和最大 batch size 估算。
- ACLGraph、CANN 及算子开销允许手动配置；Device OS 当前按固定值计入。
- 界面使用 GiB/MiB 展示结果，内部计算保留字节精度。

这些结果用于部署规划和方案比较，不等同于运行时峰值的严格保证。实际占用还会受到推理框架版本、算子实现、内存对齐、动态工作区、调度策略和硬件环境影响。上线前请使用目标环境进行实测。

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

开发服务器启动后会在终端输出本地访问地址。项目基于 React、TypeScript、vinext 和 Cloudflare Vite 插件构建，本地开发不需要 `wrangler.jsonc`。

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
node --test tests/weight-model.test.mjs
node --test tests/kv-cache-model.test.mjs
```

## 项目结构

```text
app/
├── page.tsx             # 计算器界面与整体显存建模
├── models.ts            # 模型参数、能力声明与数据来源
├── weight-model.ts      # 权重显存计算
├── kv-cache-model.ts    # KV/Index Cache 计算适配
└── globals.css          # 全局样式
tests/                   # 计算逻辑与渲染测试
scripts/build-pages.mjs  # GitHub Pages 构建脚本
worker/                  # Cloudflare Worker 入口
```

## 数据与精度

模型参数应优先来自模型发布方的官方配置或文档，并在 [`app/models.ts`](./app/models.ts) 中保留来源链接。修改公式或新增模型时，请同时添加覆盖关键字节数、精度和并行切分行为的回归测试。

如果你发现模型参数、计算假设或结果存在问题，欢迎提交 [Issue](https://github.com/Eco-Sphere/infer-memory-visualizer/issues)。请附上输入配置、预期结果、实际结果和可验证的数据来源。

## 参与贡献

欢迎贡献模型支持、计算修正、测试、界面优化和文档改进。开发流程及 Pull Request 要求请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 致谢

- [`kv-cache-calculator`](https://github.com/Eco-Sphere/kv-cache-calculator)：KV Cache 占用计算工具。
- [`vinext`](https://github.com/cloudflare/vinext)：基于 Vite 的 Next.js 兼容运行时。

## License

本项目采用 [MIT License](./LICENSE) 开源。
