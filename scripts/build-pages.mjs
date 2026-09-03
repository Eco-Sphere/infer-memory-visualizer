import { cp, mkdir, rm, writeFile } from "node:fs/promises";

const output = new URL("../pages-dist/", import.meta.url);
const client = new URL("../dist/client/", import.meta.url);
const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("pages", `${process.pid}-${Date.now()}`);

const base = process.env.GITHUB_PAGES_BASE ?? "/infer-memory-visualizer/";
const origin = process.env.GITHUB_PAGES_ORIGIN ?? "https://eco-sphere.github.io";
const { default: worker } = await import(workerUrl.href);

const response = await worker.fetch(
  new Request(`${origin}/`, { headers: { accept: "text/html" } }),
  { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
  { waitUntil() {}, passThroughOnException() {} },
);

if (!response.ok) {
  throw new Error(`Static render failed with HTTP ${response.status}`);
}

let html = await response.text();
html = html.replaceAll("/assets/", `${base}assets/`);

for (const asset of ["favicon.svg", "file.svg", "globe.svg", "og.png", "window.svg"]) {
  html = html.replaceAll(`/${asset}`, `${base}${asset}`);
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(client, output, { recursive: true });
await writeFile(new URL("index.html", output), html);
await writeFile(new URL("404.html", output), html);
await writeFile(new URL(".nojekyll", output), "");

console.log(`GitHub Pages output: ${output.pathname}`);
