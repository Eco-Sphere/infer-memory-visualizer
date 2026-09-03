import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the inference memory planner shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /推理显存建模/);
  assert.match(html, /模型与负载/);
  assert.match(html, /模型配置/);
  // 一级厂商下拉与「请按菜单栏开始选择」占位（静态 MODEL_INDEX 可服务端渲染）
  assert.match(html, /DeepSeek（deepseek-ai）/);
  assert.match(html, /请按菜单栏开始选择/);
  // 初始不加载权重，SSR 阶段显示空态文案
  assert.match(html, /请先选择厂商与权重/);
  assert.match(html, /尚未选择权重|请选择权重/);
  // 不依赖权重数据的运行时项仍可服务端渲染
  assert.match(html, /HCCL buffer/);
  assert.match(html, /EP buffer/);
  assert.match(html, /MC2 buffer/);
  assert.match(html, /CANN \+ PTA \+ 算子/);
  assert.match(html, /Device OS/);
  assert.match(html, /我要贡献/);
  assert.match(html, /infer-memory-visualizer\/blob\/main\/CONTRIBUTING\.md/);
  assert.match(html, /<strong>\d+\.\d{2}<\/strong><span>GiB<\/span>/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("includes accessible numeric controls and a live result region", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /type="number"/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /切换到深色模式/);
  assert.match(html, /<select/);
});

test("GitHub Pages output uses the repository base path", async () => {
  const html = await readFile(new URL("../pages-dist/index.html", import.meta.url), "utf8");
  assert.match(html, /\/infer-memory-visualizer\/assets\//);
  assert.match(html, /\/infer-memory-visualizer\/og\.png/);
  assert.doesNotMatch(html, /(?:href|src)="\/assets\//);
});