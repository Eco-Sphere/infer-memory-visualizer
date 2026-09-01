"use client";

import { useEffect, useMemo, useState } from "react";
import { MODEL_INDEX, QUANT_OWNER, kvCacheModelIdOf, type Vendor } from "./model-index";
import { CACHE_PRECISIONS, calculateKvCache, getKvCacheModelInfo, type CachePrecision } from "./kv-cache-model";
import { fetchModelWeightData, type ModelWeightData, type WeightLoadProgress } from "./modelscope";
import { calculateWeightFromTensors, type TensorModule } from "./weight-calc";

type Inputs = {
  maxBatchedTokens: number;
  dpSize: number;
  tpSize: number;
  attentionTpSize: number | null;
  oprojTpSize: number | null;
  embeddingTpSize: number | null;
  lmHeadTpSize: number | null;
  sharedExpertTpSize: number;
  mtpLayers: number;
  kvCacheTokens: number;
  kvCacheSequences: number;
  kvPrecision: CachePrecision;
  indexCachePrecision: CachePrecision;
  maxBS: number;
  graphCount: number;
  cannGB: number;
};

const DEFAULTS: Inputs = {
  maxBatchedTokens: 4096,
  dpSize: 8,
  tpSize: 4,
  attentionTpSize: null,
  oprojTpSize: null,
  embeddingTpSize: null,
  lmHeadTpSize: null,
  sharedExpertTpSize: 1,
  mtpLayers: 0,
  kvCacheTokens: 131072,
  kvCacheSequences: 8,
  kvPrecision: "fp8_int8",
  indexCachePrecision: "fp8_int8",
  maxBS: 128,
  graphCount: 5,
  cannGB: 1,
};

const GB = 1_000_000_000;
const GIB = 1024 ** 3;
const MB = 1_000_000;
const MIB = 1024 ** 2;
const align = (value: number, boundary: number) =>
  Math.ceil(value / boundary) * boundary;
const align480To512 = (value: number) => Math.ceil(value / 480) * 512;

const MODULE_LABEL: Record<TensorModule, string> = {
  routed_expert: "路由专家",
  shared_expert: "共享专家",
  dense_mlp: "Dense MLP",
  attention: "Attention",
  embedding: "Embedding",
  lm_head: "LM Head",
  router: "Router",
  norm: "Norm",
  vision: "视觉塔",
  other: "其他",
};

function formatGiB(bytes: number) {
  return `${(bytes / GIB).toLocaleString("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} GiB`;
}

function formatMiB(bytes: number) {
  return `${(bytes / MIB).toLocaleString("zh-CN", {
    maximumFractionDigits: 1,
  })} MiB`;
}

function formatCompact(bytes: number) {
  return bytes >= GIB ? formatGiB(bytes) : formatMiB(bytes);
}

function safe(value: number, fallback = 0) {
  return Number.isFinite(value) ? Math.max(0, value) : fallback;
}

/** 某个原始权重（模型）下的全部权重选项：原始权重 + 其 Eco-Tech 量化权重。 */
function modelWeightOptions(vendor: Vendor, modelName: string): { owner: string; name: string; kind: "raw" | "quantized"; rawName: string }[] {
  const model = vendor.models.find((m) => m.name === modelName);
  if (!model) return [];
  const out: { owner: string; name: string; kind: "raw" | "quantized"; rawName: string }[] = [
    { owner: vendor.owner, name: model.name, kind: "raw", rawName: model.name },
  ];
  for (const q of model.quantized) {
    out.push({ owner: QUANT_OWNER, name: q, kind: "quantized", rawName: model.name });
  }
  return out;
}

export default function Home() {
  const [inputs, setInputs] = useState(DEFAULTS);
  const [dark, setDark] = useState(false);

  // 一级：厂商（空串表示「请按菜单栏开始选择」占位，不加载任何权重）
  const [vendorBrand, setVendorBrand] = useState<string>("");
  // 二级：模型（原始权重名）
  const [modelName, setModelName] = useState<string>("");
  // 三级：权重（owner/name）
  const [weightKey, setWeightKey] = useState<string>("");

  const vendor = useMemo(
    () => (vendorBrand ? MODEL_INDEX.find((v) => v.brand === vendorBrand) : undefined),
    [vendorBrand],
  );
  const modelOptions = useMemo(
    () => (vendor
      ? [{ value: "", label: "请选择模型" }, ...vendor.models.map((m) => ({ value: m.name, label: m.name }))]
      : []),
    [vendor],
  );
  const weightList = useMemo(
    () => (vendor && modelName ? modelWeightOptions(vendor, modelName) : []),
    [vendor, modelName],
  );
  const weightOptions = useMemo(
    () => (vendor && modelName
      ? [{ value: "", label: "请选择权重" }, ...weightList.map((w) => ({ value: `${w.owner}/${w.name}`, label: `${w.owner}/${w.name}` }))]
      : []),
    [vendor, modelName, weightList],
  );
  const weight = weightList.find((w) => `${w.owner}/${w.name}` === weightKey);

  // 每一级都不自动预选下一级：只有用户在三级明确选择了某个权重后，才触发加载。
  const selectVendor = (brand: string) => {
    setVendorBrand(brand);
    setModelName("");
    setWeightKey("");
  };

  const selectModel = (name: string) => {
    setModelName(name);
    setWeightKey("");
  };

  const [modelData, setModelData] = useState<ModelWeightData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const [progress, setProgress] = useState<WeightLoadProgress | null>(null);

  // 仅在切换权重时拉取数据（结果由 fetchModelWeightData 会话内缓存）；
  // 并行参数变化不在此依赖中，因此不会触发任何网络请求。
  useEffect(() => {
    if (!weight) {
      setModelData(null);
      setLoading(false);
      setError(undefined);
      setProgress(null);
      return;
    }
    let cancelled = false;
    // 切换权重时总是先清空，避免上一权重数据在加载期间被「预填充」到新权重上。
    setModelData(null);
    setLoading(true);
    setError(undefined);
    setProgress(null);
    (async () => {
      try {
        const data = await fetchModelWeightData(weight.owner, weight.name, "master", (p) => {
          if (!cancelled) setProgress(p);
        });
        if (!cancelled) {
          setModelData(data);
          setLoading(false);
          setProgress(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError((e as Error).message);
          setLoading(false);
          setProgress(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [weightKey]);

  const struct = modelData?.struct;

  // 权重切分仅依赖已缓存的张量清单与并行参数：改 DP/TP 时只重算、不重拉。
  const weightResult = useMemo(() => {
    if (!modelData) return null;
    const tp = Math.max(1, Math.floor(safe(inputs.tpSize, 1)));
    return calculateWeightFromTensors(modelData.tensors, {
      tp,
      ep: tp * Math.max(1, Math.floor(safe(inputs.dpSize, 1))),
      attentionTp: Math.max(1, Math.floor(safe(inputs.attentionTpSize ?? tp, 1))),
      oprojTp: Math.max(1, Math.floor(safe(inputs.oprojTpSize ?? tp, 1))),
      embeddingTp: Math.max(1, Math.floor(safe(inputs.embeddingTpSize ?? tp, 1))),
      lmHeadTp: Math.max(1, Math.floor(safe(inputs.lmHeadTpSize ?? tp, 1))),
      sharedExpertTp: Math.max(1, Math.floor(safe(inputs.sharedExpertTpSize, 1))),
    });
  }, [modelData, inputs.tpSize, inputs.dpSize, inputs.attentionTpSize, inputs.oprojTpSize, inputs.embeddingTpSize, inputs.lmHeadTpSize, inputs.sharedExpertTpSize]);

  const kvCacheModelId = weight ? kvCacheModelIdOf(weight.rawName) : undefined;
  const epSize = Math.max(1, Math.floor(safe(inputs.tpSize, 1)) * Math.floor(safe(inputs.dpSize, 1)));

  const result = useMemo(() => {
    // 仅在「已选权重 且 权重数据已加载」后显示非零占用；加载期间保持全 0，避免预填充。
    const active = weightKey && struct ? 1 : 0;
    const H = struct?.hiddenSize ?? 0;
    const T = safe(inputs.maxBatchedTokens);
    const dp = Math.max(1, safe(inputs.dpSize, 1));
    const ep = epSize;
    const K = struct?.topK ?? 0;
    const expertCount = struct?.expertCount ?? 0;
    const localExperts = expertCount > 0 && ep > 0 ? expertCount / ep : 0;
    const maxBS = safe(inputs.maxBS);

    const hiddenResidual = 2 * 2 * T * H;
    const moeBuffers = K > 0 ? 4 * 2 * dp * T * K / ep * H : 0;
    const activation = hiddenResidual + moeBuffers;

    const hcclDP = Math.max(Math.ceil(((dp + 1) * 4) / 1024 ** 2), 50) * 2 * MB;
    const hcclTP = 200 * 2 * MB;
    const hcclEP = 200 * 2 * MB;
    const alignedDispatch = align480To512(align(2 * H, 32) + 64);
    const alignedCombine = align(2 * H, 512);
    const epDispatch = localExperts * maxBS * ep * alignedDispatch;
    const epCombine = K * maxBS * alignedCombine;
    const hcclMC2 = 2 * (epDispatch + epCombine);
    const hccl = hcclDP + hcclTP + hcclEP + hcclMC2;

    const graph = (safe(inputs.graphCount) / 5) * 0.27 * GB;
    const cann = safe(inputs.cannGB) * GIB;
    const deviceOS = 4.25 * GIB;

    const tp = Math.max(1, Math.floor(safe(inputs.tpSize, 1)));
    const attentionTp = Math.max(1, Math.floor(safe(inputs.attentionTpSize ?? tp, 1)));
    const kvCacheInfo = kvCacheModelId ? getKvCacheModelInfo(kvCacheModelId) : undefined;
    const kvCacheBreakdown = kvCacheModelId ? calculateKvCache({
      modelId: kvCacheModelId,
      tokens: safe(inputs.kvCacheTokens, DEFAULTS.kvCacheTokens),
      sequences: safe(inputs.kvCacheSequences, DEFAULTS.kvCacheSequences),
      kvPrecision: inputs.kvPrecision ?? DEFAULTS.kvPrecision,
      indexPrecision: inputs.indexCachePrecision ?? DEFAULTS.indexCachePrecision,
      mtpLayers: Math.floor(safe(inputs.mtpLayers)),
      tpSize: attentionTp,
    }) ?? null : null;
    const kvCache = kvCacheBreakdown?.total ?? 0;

    const weight = weightResult?.perDeviceWeight ?? 0;
    const fullWeight = weightResult?.fullWeight ?? 0;
    const total = weight + kvCache + activation + hccl + graph + cann + deviceOS;

    return {
      activation: active * activation,
      hiddenResidual: active * hiddenResidual,
      moeBuffers: active * moeBuffers,
      hccl: active * hccl,
      hcclDP: active * hcclDP,
      hcclTP: active * hcclTP,
      hcclEP: active * hcclEP,
      hcclMC2: active * hcclMC2,
      epDispatch: active * 2 * epDispatch,
      epCombine: active * 2 * epCombine,
      alignedDispatch: active * alignedDispatch,
      alignedCombine: active * alignedCombine,
      attentionTp, graph: active * graph, cann: active * cann, deviceOS: active * deviceOS,
      weight: active * weight, fullWeight: active * fullWeight, kvCache: active * kvCache,
      kvCacheBreakdown, kvCacheInfo, weightResult,
      total: active * total, localExperts,
    };
  }, [inputs, epSize, struct, weightResult, kvCacheModelId, weightKey]);

  const update = (key: keyof Inputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: Number(value) }));
  };
  const updateOptional = (key: keyof Inputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: value === "" ? null : Number(value) }));
  };
  const reset = () => {
    setVendorBrand("");
    setModelName("");
    setWeightKey("");
    setInputs(DEFAULTS);
  };

  const categories = [
    ...(weightResult ? [{ label: "权重占用", value: result.weight, color: "var(--rose)", display: formatGiB(result.weight) }] : []),
    ...(kvCacheModelId ? [{ label: "KV + Index Cache", value: result.kvCache, color: "var(--cyan)", display: formatGiB(result.kvCache) }] : []),
    { label: "激活占用", value: result.activation, color: "var(--coral)", display: formatGiB(result.activation) },
    { label: "HCCL buffer", value: result.hccl, color: "var(--blue)", display: formatGiB(result.hccl) },
    { label: "ACLGraph 占用", value: result.graph, color: "var(--violet)", display: formatGiB(result.graph) },
    { label: "CANN + PTA + 算子", value: result.cann, color: "var(--green)", display: formatGiB(result.cann) },
    { label: "Device OS", value: result.deviceOS, color: "var(--amber)", display: formatGiB(result.deviceOS) },
  ];

  return (
    <main className={dark ? "app dark" : "app"}>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="shell">
        <header className="topbar">
          <div className="brand-block">
            <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
            <div>
              <h1>推理显存建模</h1>
              <p>Memory planner for distributed MoE inference</p>
            </div>
          </div>
          <div className="top-actions">
            <a className="top-link" href="https://github.com/Eco-Sphere/infer-memory-visualizer" target="_blank" rel="noreferrer" aria-label="GitHub 仓库">
              <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
              </svg>
              GitHub
            </a>
            <a className="top-link" href="https://github.com/Eco-Sphere/infer-memory-visualizer/issues/new" target="_blank" rel="noreferrer" aria-label="反馈问题">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <line x1="12" y1="8" x2="12" y2="13" />
                <line x1="12" y1="16.5" x2="12.01" y2="16.5" />
              </svg>
              反馈
            </a>
            <a className="top-link" href="https://github.com/Eco-Sphere/infer-memory-visualizer/blob/main/CONTRIBUTING.md" target="_blank" rel="noreferrer" aria-label="查看贡献指南">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="6" cy="6" r="2" />
                <circle cx="18" cy="6" r="2" />
                <circle cx="6" cy="18" r="2" />
                <path d="M6 8v8" />
                <path d="M8 6h5a5 5 0 0 1 5 5v5" />
              </svg>
              我要贡献
            </a>
            <button className="theme-toggle" type="button" onClick={() => setDark((value) => !value)} aria-label={dark ? "切换到浅色模式" : "切换到深色模式"}>
              <span>{dark ? "☀" : "☾"}</span>{dark ? "浅色" : "深色"}
            </button>
          </div>
        </header>

        <section className="workspace">
          <aside className="control-card">
            <div className="section-title">
              <div>
                <span className="eyebrow">CONFIGURATION</span>
                <h2>模型与负载</h2>
              </div>
              <button className="reset" type="button" onClick={reset}>重置</button>
            </div>

            <fieldset>
              <legend>模型配置</legend>
              <div className="model-picker-grid">
                <SelectField
                  label="厂商"
                  value={vendorBrand}
                  onChange={selectVendor}
                  options={[{ value: "", label: "请按菜单栏开始选择" }, ...MODEL_INDEX.map((v) => ({ value: v.brand, label: `${v.brand}（${v.owner}）` }))]}
                />
                <SelectField
                  label="模型"
                  value={modelName}
                  onChange={selectModel}
                  options={modelOptions}
                  disabled={!vendor}
                />
                <SelectField
                  label="权重"
                  value={weightKey}
                  onChange={(value) => setWeightKey(value)}
                  options={weightOptions}
                  disabled={!vendor || !modelName}
                />
              </div>
              <p className="field-note">权重数据实时从 ModelScope 拉取。</p>
            </fieldset>

            {loading && <p className="field-note">正在拉取 {weight?.owner}/{weight?.name} 的权重清单…</p>}
            {error && <p className="field-note" style={{ color: "var(--rose)" }}>加载失败：{error}</p>}

            <fieldset>
              <legend>负载参数</legend>
              <NumberField label="Max batched tokens" value={inputs.maxBatchedTokens} onChange={(v) => update("maxBatchedTokens", v)} />
            </fieldset>

            {kvCacheModelId && struct && (
              <fieldset>
                <legend>KV Cache</legend>
                <div className="field-grid">
                  <NumberField label="Tokens / sequence" value={inputs.kvCacheTokens ?? DEFAULTS.kvCacheTokens} onChange={(v) => update("kvCacheTokens", v)} />
                  <NumberField label="Concurrent sequences" value={inputs.kvCacheSequences ?? DEFAULTS.kvCacheSequences} onChange={(v) => update("kvCacheSequences", v)} />
                </div>
                <div className="field-grid cache-precision-fields">
                  <SelectField label="KV precision" value={inputs.kvPrecision ?? DEFAULTS.kvPrecision} onChange={(value) => setInputs((current) => ({ ...current, kvPrecision: value as CachePrecision }))} options={Object.entries(CACHE_PRECISIONS).map(([value, item]) => ({ value, label: item.label }))} />
                  <SelectField label="Index precision" value={inputs.indexCachePrecision ?? DEFAULTS.indexCachePrecision} onChange={(value) => setInputs((current) => ({ ...current, indexCachePrecision: value as CachePrecision }))} options={Object.entries(CACHE_PRECISIONS).map(([value, item]) => ({ value, label: item.label }))} />
                </div>
                <p className="field-note">基于上游 kv-cache-calculator 的 {kvCacheModelId} 结构参数。</p>
              </fieldset>
            )}

            <fieldset>
              <legend>并行策略</legend>
              <div className="field-grid">
                <NumberField label="DP size" value={inputs.dpSize} onChange={(v) => update("dpSize", v)} />
                <NumberField label="TP size" value={inputs.tpSize} onChange={(v) => update("tpSize", v)} />
              </div>
              <div className="parallel-basic">
                <NumberField label="Shared Expert TP" value={inputs.sharedExpertTpSize} onChange={(v) => update("sharedExpertTpSize", v)} />
              </div>
              {kvCacheModelId && (
                <div className="parallel-basic">
                  <NumberField label="MTP layers" value={inputs.mtpLayers} onChange={(v) => update("mtpLayers", v)} />
                </div>
              )}
              <p className="field-note">Shared Expert TP 为 1 表示不切分，不影响 EP size。</p>
              <details className="advanced-tp">
                <summary>
                  高级切分配置
                  <span className="help-icon" role="note" aria-label="默认跟随主 TP size（与 vLLM 一致），输入数值后单独切分，清空输入框恢复跟随；各项均不影响 EP size。"
                    onClick={(event) => event.preventDefault()}
                  >?<span className="help-tooltip">默认跟随主 TP size（与 vLLM 一致），输入数值后单独切分，清空输入框恢复跟随；各项均不影响 EP size。</span></span>
                </summary>
                <div className="advanced-tp-body">
                  <div className="field-grid">
                    <NumberField label="QK / Indexer TP" value={result.attentionTp} onChange={(v) => updateOptional("attentionTpSize", v)} />
                    <NumberField label="O-proj TP" value={Math.max(1, Math.floor(safe(inputs.oprojTpSize ?? inputs.tpSize, 1)))} onChange={(v) => updateOptional("oprojTpSize", v)} />
                  </div>
                  <div className="field-grid">
                    <NumberField label="Embedding TP" value={Math.max(1, Math.floor(safe(inputs.embeddingTpSize ?? inputs.tpSize, 1)))} onChange={(v) => updateOptional("embeddingTpSize", v)} />
                    <NumberField label="LM Head TP" value={Math.max(1, Math.floor(safe(inputs.lmHeadTpSize ?? inputs.tpSize, 1)))} onChange={(v) => updateOptional("lmHeadTpSize", v)} />
                  </div>
                </div>
              </details>
            </fieldset>

            <fieldset>
              <legend>运行时</legend>
              <div className="field-grid">
                <NumberField label="Max BS" value={inputs.maxBS} onChange={(v) => update("maxBS", v)} />
                <NumberField label="ACLGraph 个数" value={inputs.graphCount} onChange={(v) => update("graphCount", v)} />
              </div>
              <NumberField label="CANN + PTA + 算子预估（GiB）" value={inputs.cannGB} step="0.1" onChange={(v) => update("cannGB", v)} />
              <p className="field-note">该项默认按 1 GiB 预留，实际占用通常低于此值。</p>
            </fieldset>
          </aside>

          <section className="results" aria-live="polite">
            <article className="hero-card">
              <div className="hero-copy">
                <span className="eyebrow">ESTIMATED PER DEVICE</span>
                <div className="total-line"><strong>{formatGiB(result.total).replace(" GiB", "")}</strong><span>GiB</span></div>
                <p>{weightResult ? "单卡总显存预估（含权重）" : weight ? "加载权重数据后显示完整预估" : "请先选择厂商与权重"}</p>
              </div>
            </article>

            <article className="model-context-card">
              <div className="model-identity">
                <span className="eyebrow">ACTIVE MODEL</span>
                {weight ? (
                  <a className="model-link" href={`https://modelscope.cn/models/${weight.owner}/${weight.name}`} target="_blank" rel="noreferrer">
                    {weight.owner}/{weight.name}
                  </a>
                ) : (
                  <strong className="model-link model-link-empty">尚未选择权重</strong>
                )}
              </div>
              {struct ? (
                <>
                  <div className="model-facts">
                    <div className="model-fact"><span>Hidden size</span><strong>{struct.hiddenSize.toLocaleString("zh-CN")}</strong></div>
                    <div className="model-fact"><span>层数</span><strong>{struct.numLayers.toLocaleString("zh-CN")}</strong></div>
                    <div className="model-fact"><span>专家总数</span><strong>{struct.expertCount.toLocaleString("zh-CN")}</strong><small>{struct.expertCount > 0 ? "MoE" : "Dense"}</small></div>
                    <div className="model-fact"><span>TopK 专家</span><strong>{struct.topK.toLocaleString("zh-CN")}</strong></div>
                    <div className="model-fact"><span>EP size</span><strong>{epSize.toLocaleString("zh-CN")}</strong><small>TP {inputs.tpSize} × DP {inputs.dpSize}</small></div>
                    <div className="model-fact"><span>本地专家数</span><strong>{result.localExperts.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}</strong></div>
                  </div>
                  {modelData?.optionalShards && modelData.optionalShards.length > 0 && (
                    <p className="field-note">已并入可选权重：{modelData.optionalShards.join("、")}</p>
                  )}
                </>
              ) : loading && progress ? (
                progress.phase === "shards" ? (
                  <div className="load-progress" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={progress.done}>
                    <div className="load-progress-track">
                      <div className="load-progress-fill" style={{ width: `${progress.total ? Math.round((progress.done / progress.total) * 100) : 0}%` }} />
                    </div>
                    <span>加载权重分片 {progress.done}/{progress.total}</span>
                  </div>
                ) : (
                  <p className="field-note">正在加载权重索引…</p>
                )
              ) : (
                <p className="field-note">{loading ? "加载中…" : "请选择权重"}</p>
              )}
            </article>

            <div className="metric-grid">
              {categories.map((item) => (
                <article className="metric-card" key={item.label}>
                  <div className="metric-label"><i style={{ background: item.color }} />{item.label}</div>
                  <strong>{item.display}</strong>
                  <span>{result.total ? `${(item.value / result.total * 100).toFixed(1)}%` : "0%"} of total</span>
                </article>
              ))}
            </div>

            <article className="breakdown-card">
              <div className="panel-heading">
                <div><span className="eyebrow">MEMORY MAP</span><h2>显存构成</h2></div>
                <span className="unit-pill">GiB</span>
              </div>
              <div className="stack" aria-label="显存构成比例图">
                {categories.map((item) => (
                  <div key={item.label} title={`${item.label}: ${formatGiB(item.value)}`}
                    style={{ width: `${result.total ? item.value / result.total * 100 : 0}%`, background: item.color }} />
                ))}
              </div>
              <div className="legend">
                {categories.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}
              </div>

              <div className="detail-sections">
                {weightResult && (
                  <DetailSection title="权重占用" value={result.weight} tone="rose">
                    <div className="sub-detail weight-summary">
                      <span>全量模型 {formatGiB(result.fullWeight)}</span>
                      <span>当前单卡 {formatGiB(result.weight)}</span>
                      <span>{weight?.kind === "quantized" ? `Eco-Tech 量化 · ${weight.name}` : `原始权重 · ${weight?.name}`}</span>
                    </div>
                    {Object.entries(weightResult.breakdown).filter(([, b]) => b.tensors > 0).map(([module, b]) => (
                      <DetailRow
                        key={module}
                        label={`${MODULE_LABEL[module as TensorModule]}（${b.tensors} 张量）`}
                        value={b.perDevice}
                        formula={`全量 ${formatCompact(b.full)}，逐张量 dtype × shape 求和`}
                      />
                    ))}
                  </DetailSection>
                )}

                {result.kvCacheInfo && result.kvCacheBreakdown && (
                  <DetailSection title="KV + Index Cache" value={result.kvCache} tone="cyan">
                    <DetailRow
                      label={`K/V Cache（${CACHE_PRECISIONS[inputs.kvPrecision ?? DEFAULTS.kvPrecision].label}）`}
                      value={result.kvCacheBreakdown.kvCache}
                      formula={`${inputs.kvCacheTokens ?? DEFAULTS.kvCacheTokens} × ${inputs.kvCacheSequences ?? DEFAULTS.kvCacheSequences} × ${result.kvCacheInfo.layers} + MTP ${inputs.mtpLayers} × ${result.kvCacheBreakdown.kvCopies} × (${result.kvCacheInfo.kvHeads} ÷ TP ${result.attentionTp}) × ${result.kvCacheInfo.headDim} × ${result.kvCacheBreakdown.kvBytesPerElement} B`}
                    />
                    <DetailRow
                      label={`Index Cache（${CACHE_PRECISIONS[inputs.indexCachePrecision ?? DEFAULTS.indexCachePrecision].label}）`}
                      value={result.kvCacheBreakdown.indexCache}
                      formula={`${inputs.kvCacheTokens ?? DEFAULTS.kvCacheTokens} × ${inputs.kvCacheSequences ?? DEFAULTS.kvCacheSequences} × ${result.kvCacheInfo.sparseLayers} × ${result.kvCacheInfo.indexHeadDim} × ${result.kvCacheBreakdown.indexBytesPerElement} B`}
                    />
                  </DetailSection>
                )}

                <DetailSection title="激活占用" value={result.activation} tone="coral">
                  <DetailRow label="Hidden states + residual" value={result.hiddenResidual} formula={`2 × 2 B × ${inputs.maxBatchedTokens} × ${struct?.hiddenSize ?? 0}`} />
                  <DetailRow label="4 份 MoE 激活 buffer" value={result.moeBuffers} formula={`4 × 2 B × ${inputs.dpSize} × ${inputs.maxBatchedTokens} × ${struct?.topK ?? 0} ÷ ${epSize} × ${struct?.hiddenSize ?? 0}`} />
                </DetailSection>

                <DetailSection title="HCCL buffer" value={result.hccl} tone="blue">
                  <DetailRow label="DP buffer" value={result.hcclDP} formula={`max(ceil((${inputs.dpSize} + 1) × 4 ÷ 1024²), 50) × 2 MB`} />
                  <DetailRow label="TP buffer" value={result.hcclTP} formula="200 MB × 2" />
                  <DetailRow label="EP buffer" value={result.hcclEP} formula="200 MB × 2" />
                  <DetailRow label={`MC2 buffer（本地专家 ${result.localExperts.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}）`} value={result.hcclMC2} formula={`2 × (本地专家 × Max BS × EP × 480Align512 + K × Max BS × Align512)`} />
                  <div className="sub-detail">
                    <span>Dispatch {formatMiB(result.epDispatch)}</span>
                    <span>Combine {formatMiB(result.epCombine)}</span>
                    <span>480Align512 = {result.alignedDispatch.toLocaleString("zh-CN")} B</span>
                    <span>Align512 = {result.alignedCombine.toLocaleString("zh-CN")} B</span>
                  </div>
                </DetailSection>

                <DetailSection title="其他运行时" value={result.graph + result.cann + result.deviceOS} tone="violet">
                  <DetailRow label={`ACLGraph 占用（${inputs.graphCount} 张）`} value={result.graph} formula={`${inputs.graphCount} ÷ 5 × 0.27 GB`} />
                  <DetailRow label="CANN + PTA + 算子" value={result.cann} formula={`${inputs.cannGB} GiB 预估值`} />
                  <DetailRow label="Device OS 固定占用" value={result.deviceOS} formula="4.25 × 1024³ bytes（固定值）" />
                </DetailSection>
              </div>
            </article>

            <p className="method-note"><strong>口径说明</strong> 所有结果均为单卡估算并统一显示为 GiB。权重从 safetensors header 逐张量（dtype × shape）求和，全量精确；单卡按模块通用切分（专家÷EP、注意力/Embedding/LM Head÷TP、其余复制）为近似。KV Cache 仅对上游已收录的模型展示。EP 自动等于 TP × DP。</p>
          </section>
        </section>
      </div>
    </main>
  );
}

function NumberField({ label, value, onChange, step = "1" }: { label: string; value: number; onChange: (value: string) => void; step?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" min="0" step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function SelectField({ label, value, onChange, options, disabled = false }: { label: string; value: string; onChange: (value: string) => void; options: { value: string; label: string }[]; disabled?: boolean }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
        {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  );
}

function DetailSection({ title, value, tone, children }: { title: string; value: number; tone: string; children: React.ReactNode }) {
  return (
    <section className={`detail-section ${tone}`}>
      <div className="detail-title"><span>{title}</span><strong>{formatGiB(value)}</strong></div>
      {children}
    </section>
  );
}

function DetailRow({ label, value, formula }: { label: string; value: number; formula: string }) {
  return (
    <div className="detail-row">
      <div><span>{label}</span><code>{formula}</code></div>
      <strong>{formatCompact(value)}</strong>
    </div>
  );
}