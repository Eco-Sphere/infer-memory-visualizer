// ModelScope 数据获取层：拼接 resolve 直链并拉取 config 与 safetensors header。
// resolve 端点已实测开放 CORS（Access-Control-Allow-Origin: *）且支持 HTTP Range，
// 因此可在浏览器与 Node 中直接使用，符合“纯 HTTP + 前端动态获取”。
//
// 性能策略：
// - 每个分片 header 用「一次较大范围的 Range 请求」覆盖常见情况（避免两次串行 Range），
//   仅当 header 异常超长时回退到精确范围请求。
// - 多个分片按固定并发度并行拉取。
// - 拉取结果按 `${owner}/${name}` 缓存在内存中，会话内切换权重命中缓存直接复用。

import { parseSafetensorsHeader, type SafetensorsHeader, type SafetensorInfo } from "./safetensors.ts";
import { extractModelStructure, type ModelStructure } from "./config-model.ts";

/** 乐观覆盖范围：绝大多数 safetensors header JSON 远小于 1 MiB。 */
const HEADER_MAX_BYTES = 1024 * 1024;
/** 分片 header 拉取的并发度。 */
const CONCURRENCY = 8;

export function modelscopeResolveUrl(
  owner: string,
  name: string,
  file: string,
  ref = "master",
): string {
  return `https://modelscope.cn/models/${owner}/${name}/resolve/${ref}/${file}`;
}

/** 拉取模型 config.json（resolve 直链，开放 CORS）。 */
export async function fetchConfigJson(
  owner: string,
  name: string,
  ref = "master",
): Promise<Record<string, unknown>> {
  const url = modelscopeResolveUrl(owner, name, "config.json", ref);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`拉取 config.json 失败: HTTP ${res.status} (${url})`);
  }
  return (await res.json()) as Record<string, unknown>;
}

// 常见 index.json 文件名，按可能性排序；用于从 weight_map 推导分片列表。
const INDEX_CANDIDATES = [
  "model.safetensors.index.json",
  "quant_model_weights.safetensors.index.json",
  "pytorch_model.safetensors.index.json",
];

// 「可选附加权重」文件：不在 index.json 的 weight_map 里，但推理时同样会加载，
// 例如 QuaRot 的全局旋转矩阵（optional/quarot.safetensors）。用 8 字节 Range 探测
// 是否存在，命中则并入分片列表，避免漏算这部分显存。
const OPTIONAL_WEIGHT_CANDIDATES = [
  "optional/quarot.safetensors",
  "quarot.safetensors",
];

/**
 * 定位模型的所有 safetensors 分片文件名。
 * 权重分片列表无法走 ModelScope 的 repo/files 列表 API（该 API 无 CORS 头），
 * 因此改为从 index.json 的 weight_map 推导；缺失 index 时回退到单文件试探。
 */
export async function fetchSafetensorsShards(
  owner: string,
  name: string,
  ref = "master",
): Promise<string[]> {
  for (const indexFile of INDEX_CANDIDATES) {
    const url = modelscopeResolveUrl(owner, name, indexFile, ref);
    let res: Response;
    try {
      res = await fetch(url);
    } catch {
      continue;
    }
    if (!res.ok) continue;
    try {
      const index = (await res.json()) as { weight_map?: Record<string, string> };
      const weightMap = index.weight_map;
      if (weightMap && typeof weightMap === "object") {
        const shards = Array.from(new Set(Object.values(weightMap)))
          .filter((s) => s.endsWith(".safetensors"))
          .sort();
        if (shards.length > 0) return shards;
      }
    } catch {
      // index 解析失败，继续尝试下一个候选
    }
  }

  // 回退：尝试常见单文件权重。
  for (const single of ["model.safetensors", "quant_model_weights.safetensors"]) {
    const url = modelscopeResolveUrl(owner, name, single, ref);
    try {
      const res = await fetch(url, { headers: { Range: "bytes=0-7" } });
      if (res.ok) return [single];
    } catch {
      // ignore
    }
  }
  throw new Error(`无法定位 ${owner}/${name} 的 safetensors 权重分片`);
}

/**
 * 探测并返回存在的「可选附加权重」文件。只用 8 字节 Range 判断存在性，
 * 不下载文件体，成本远低于一次完整 GET。
 */
async function fetchOptionalShards(
  owner: string,
  name: string,
  ref = "master",
): Promise<string[]> {
  const found: string[] = [];
  await mapWithConcurrency(OPTIONAL_WEIGHT_CANDIDATES, OPTIONAL_WEIGHT_CANDIDATES.length, async (file) => {
    const url = modelscopeResolveUrl(owner, name, file, ref);
    try {
      const res = await fetch(url, { headers: { Range: "bytes=0-7" } });
      if (res.ok) found.push(file);
    } catch {
      // 忽略探测失败（不存在或网络错误），视为无该可选权重。
    }
  });
  return found;
}

/**
 * 只拉取 safetensors 文件的头部（header JSON），不下载整个权重。
 * 先用一次「8 字节长度字段 + 1 MiB 余量」的 Range 覆盖常见 header，避免两次串行
 * 请求；仅当 header 声明长度超过该余量时，才回退到精确范围再拉一次。
 */
export async function fetchSafetensorsHeader(
  owner: string,
  name: string,
  file: string,
  ref = "master",
): Promise<SafetensorsHeader> {
  const url = modelscopeResolveUrl(owner, name, file, ref);

  const res = await fetch(url, { headers: { Range: `bytes=0-${8 + HEADER_MAX_BYTES - 1}` } });
  if (!res.ok) {
    throw new Error(`拉取 safetensors header 失败: HTTP ${res.status} (${url})`);
  }
  const buf = await res.arrayBuffer();
  if (buf.byteLength < 8) {
    throw new Error(`safetensors header 长度字段不足 8 字节: ${buf.byteLength}`);
  }
  const headerLen = Number(new DataView(buf).getBigUint64(0, true));
  if (buf.byteLength >= 8 + headerLen) {
    return parseSafetensorsHeader(buf);
  }

  // 罕见：单个分片 header 超过 1 MiB，精确再拉一次。
  const exact = await fetch(url, { headers: { Range: `bytes=0-${8 + headerLen - 1}` } });
  if (!exact.ok) {
    throw new Error(`拉取 safetensors header 失败: HTTP ${exact.status} (${url})`);
  }
  return parseSafetensorsHeader(await exact.arrayBuffer());
}

export type ModelWeightData = {
  struct: ModelStructure;
  tensors: SafetensorInfo[];
  shardCount: number;
  optionalShards: string[];
};

/** 权重加载进度：index 阶段总数为 0（未知）；shards 阶段 done/total 为已拉取分片数。 */
export type WeightLoadProgress = {
  phase: "index" | "shards";
  done: number;
  total: number;
};
export type WeightProgressCallback = (progress: WeightLoadProgress) => void;

function modelDataCacheKey(owner: string, name: string): string {
  return `${owner}/${name}`;
}

const dataCache = new Map<string, ModelWeightData>();
const pendingCache = new Map<string, Promise<ModelWeightData>>();

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = cursor++;
      if (i >= items.length) break;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}

/** 命中会话内缓存时直接返回缓存数据，不发起网络请求。 */
export function getCachedModelWeightData(owner: string, name: string): ModelWeightData | undefined {
  return dataCache.get(modelDataCacheKey(owner, name));
}

/**
 * 拉取并解析某个权重的结构参数与全部张量清单，带会话内内存缓存与在途去重：
 * 首次拉取的网络结果会被缓存，后续（含切回已看过的权重）直接复用，不再请求。
 * 可选 onProgress 会在「拉索引」与「拉分片 header」两个阶段上报进度：
 * 命中缓存时不会调用 onProgress（数据同步返回）。
 */
export function fetchModelWeightData(
  owner: string,
  name: string,
  ref = "master",
  onProgress?: WeightProgressCallback,
): Promise<ModelWeightData> {
  const key = modelDataCacheKey(owner, name);
  const cached = dataCache.get(key);
  if (cached) return Promise.resolve(cached);
  const pending = pendingCache.get(key);
  if (pending) return pending;

  const promise = (async () => {
    onProgress?.({ phase: "index", done: 0, total: 0 });
    const config = await fetchConfigJson(owner, name, ref);
    const struct = extractModelStructure(config);
    const mainShards = await fetchSafetensorsShards(owner, name, ref);
    const optionalShards = await fetchOptionalShards(owner, name, ref);
    const shards = [...mainShards, ...optionalShards];

    let done = 0;
    onProgress?.({ phase: "shards", done: 0, total: shards.length });
    const headers = await mapWithConcurrency(shards, CONCURRENCY, async (shard) => {
      const header = await fetchSafetensorsHeader(owner, name, shard, ref);
      done += 1;
      onProgress?.({ phase: "shards", done, total: shards.length });
      return header;
    });
    const tensors = headers.flatMap((header) => header.tensors);
    return { struct, tensors, shardCount: shards.length, optionalShards } satisfies ModelWeightData;
  })();

  pendingCache.set(key, promise);
  // 无论成败，结束后清除在途标记；失败不缓存。用 then 双分支处理，避免额外 rejection 分支泄露。
  promise.then(
    (data) => {
      dataCache.set(key, data);
      pendingCache.delete(key);
    },
    () => {
      pendingCache.delete(key);
    },
  );
  return promise;
}