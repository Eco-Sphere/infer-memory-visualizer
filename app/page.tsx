"use client";

import { useMemo, useState } from "react";
import { DEFAULT_MODEL_ID, MODELS } from "./models";
import { CACHE_PRECISIONS, calculateKvCache, getKvCacheModelInfo, type CachePrecision } from "./kv-cache-model";
import { calculateMiniMaxWeight } from "./weight-model";

type Inputs = {
  maxBatchedTokens: number;
  dpSize: number;
  tpSize: number;
  // null means the projection follows the main TP size, mirroring vLLM's
  // default of sharding over the global TP group.
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

export default function Home() {
  const [inputs, setInputs] = useState(DEFAULTS);
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);
  const [dark, setDark] = useState(false);
  const model = MODELS.find((item) => item.id === modelId) ?? MODELS[0];
  const families = Array.from(new Set(MODELS.map((item) => item.family)));
  const [family, setFamily] = useState(model.family);
  const familyModels = MODELS.filter((item) => item.family === family);
  const epSize = Math.max(1, Math.floor(safe(inputs.tpSize, 1)) * Math.floor(safe(inputs.dpSize, 1)));
  const localExpertNum = model.expertCount / epSize;

  const result = useMemo(() => {
    const H = model.hiddenSize;
    const T = safe(inputs.maxBatchedTokens);
    const dp = Math.max(1, safe(inputs.dpSize, 1));
    const ep = epSize;
    const K = model.topK;
    const localExperts = localExpertNum;
    const maxBS = safe(inputs.maxBS);

    const hiddenResidual = 2 * 2 * T * H;
    const moeBuffers = 4 * 2 * dp * T * K / ep * H;
    const activation = hiddenResidual + moeBuffers;

    const hcclDP = Math.max(Math.ceil(((dp + 1) * 4) / 1024 ** 2), 50) * 2 * MB;
    const hcclTP = 200 * 2 * MB;
    const alignedDispatch = align480To512(align(2 * H, 32) + 64);
    const alignedCombine = align(2 * H, 512);
    const epDispatch = localExperts * maxBS * ep * alignedDispatch;
    const epCombine = K * maxBS * alignedCombine;
    const hcclEP = 2 * (epDispatch + epCombine);
    const hccl = hcclDP + hcclTP + hcclEP;

    const graph = (safe(inputs.graphCount) / 5) * 0.27 * GB;
    const cann = safe(inputs.cannGB) * GIB;
    const deviceOS = 4.25 * GIB;
    const profile = model.weightProfile;
    const kvCacheInfo = model.kvCacheModelId
      ? getKvCacheModelInfo(model.kvCacheModelId)
      : undefined;
    const tp = Math.max(1, Math.floor(safe(inputs.tpSize, 1)));
    const attentionTp = Math.max(1, Math.floor(safe(inputs.attentionTpSize ?? tp, 1)));
    const oprojTp = Math.max(1, Math.floor(safe(inputs.oprojTpSize ?? tp, 1)));
    const embeddingTp = Math.max(1, Math.floor(safe(inputs.embeddingTpSize ?? tp, 1)));
    const lmHeadTp = Math.max(1, Math.floor(safe(inputs.lmHeadTpSize ?? tp, 1)));
    const sharedExpertTp = Math.max(1, Math.floor(safe(inputs.sharedExpertTpSize, 1)));
    const paddedVocab = profile ? align(profile.vocabSize, profile.vocabPaddingSize) : 0;
    const weightConfigValid = Boolean(
      profile
      && model.expertCount % ep === 0
      && profile.attentionHeads % attentionTp === 0
      && profile.attentionHeads % oprojTp === 0
      && (profile.sharedKv || profile.kvHeads % attentionTp === 0)
      && profile.indexerHeads % attentionTp === 0
      && profile.denseIntermediateSize % tp === 0
      && profile.expertIntermediateSize % sharedExpertTp === 0
      && paddedVocab % embeddingTp === 0
      && paddedVocab % lmHeadTp === 0
    );

    let weight = 0;
    let weightBreakdown = null;
    let fullWeight = 0;
    if (profile && weightConfigValid) {
      weightBreakdown = calculateMiniMaxWeight({
        profile,
        hiddenSize: H,
        expertCount: model.expertCount,
        tpSize: tp,
        epSize: ep,
        attentionTpSize: attentionTp,
        oprojTpSize: oprojTp,
        embeddingTpSize: embeddingTp,
        lmHeadTpSize: lmHeadTp,
        sharedExpertTpSize: sharedExpertTp,
        mtpLayers: Math.floor(safe(inputs.mtpLayers)),
      });
      weight = weightBreakdown.total;
      fullWeight = calculateMiniMaxWeight({
        profile,
        hiddenSize: H,
        expertCount: model.expertCount,
        tpSize: 1,
        epSize: 1,
        attentionTpSize: 1,
        oprojTpSize: 1,
        embeddingTpSize: 1,
        lmHeadTpSize: 1,
        sharedExpertTpSize: 1,
        mtpLayers: Math.floor(safe(inputs.mtpLayers)),
      }).total;
    }

    const kvCacheBreakdown = model.kvCacheModelId ? calculateKvCache({
      modelId: model.kvCacheModelId,
      tokens: safe(inputs.kvCacheTokens, DEFAULTS.kvCacheTokens),
      sequences: safe(inputs.kvCacheSequences, DEFAULTS.kvCacheSequences),
      kvPrecision: inputs.kvPrecision ?? DEFAULTS.kvPrecision,
      indexPrecision: inputs.indexCachePrecision ?? DEFAULTS.indexCachePrecision,
      mtpLayers: Math.floor(safe(inputs.mtpLayers)),
      tpSize: attentionTp,
    }) ?? null : null;
    const kvCache = kvCacheBreakdown?.total ?? 0;

    const total = weight + kvCache + activation + hccl + graph + cann + deviceOS;

    return {
      activation,
      hiddenResidual,
      moeBuffers,
      hccl,
      hcclDP,
      hcclTP,
      hcclEP,
      epDispatch: 2 * epDispatch,
      epCombine: 2 * epCombine,
      alignedDispatch,
      alignedCombine,
      attentionTp,
      oprojTp,
      embeddingTp,
      lmHeadTp,
      sharedExpertTp,
      graph,
      cann,
      deviceOS,
      weight,
      fullWeight,
      kvCache,
      kvCacheBreakdown,
      kvCacheInfo,
      weightBreakdown,
      weightConfigValid,
      total,
    };
  }, [inputs, epSize, localExpertNum, model]);

  const update = (key: keyof Inputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: Number(value) }));
  };

  const updateOptional = (key: keyof Inputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: value === "" ? null : Number(value) }));
  };

  const changeFamily = (nextFamily: string) => {
    const nextModel = MODELS.find((item) => item.family === nextFamily) ?? MODELS[0];
    setFamily(nextFamily);
    setModelId(nextModel.id);
  };

  const changeModel = (nextId: string) => {
    setModelId(nextId);
  };

  const reset = () => {
    const defaultModel = MODELS.find((item) => item.id === DEFAULT_MODEL_ID) ?? MODELS[0];
    setFamily(defaultModel.family);
    setModelId(defaultModel.id);
    setInputs(DEFAULTS);
  };

  const categories = [
    ...(model.weightProfile ? [{ label: "权重占用", value: result.weight, color: "var(--rose)", display: result.weightConfigValid ? formatGiB(result.weight) : "配置无效" }] : []),
    ...(model.kvCacheModelId ? [{ label: "KV + Index Cache", value: result.kvCache, color: "var(--cyan)", display: formatGiB(result.kvCache) }] : []),
    { label: "激活占用", value: result.activation, color: "var(--coral)", display: formatGiB(result.activation) },
    { label: "HCCL buffer", value: result.hccl, color: "var(--blue)", display: formatGiB(result.hccl) },
    { label: "图占用", value: result.graph, color: "var(--violet)", display: formatGiB(result.graph) },
    { label: "CANN + PTA + 算子", value: result.cann, color: "var(--green)", display: formatGiB(result.cann) },
    { label: "Device OS", value: result.deviceOS, color: "var(--amber)", display: "4.25 GiB" },
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
            <span className="live"><i />实时估算</span>
            <button
              className="theme-toggle"
              type="button"
              onClick={() => setDark((value) => !value)}
              aria-label={dark ? "切换到浅色模式" : "切换到深色模式"}
            >
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
                <SelectField label="模型族" value={family} onChange={changeFamily} options={families.map((item) => ({ value: item, label: item }))} />
                <SelectField label="模型" value={modelId} onChange={changeModel} options={familyModels.map((item) => ({ value: item.id, label: item.label }))} />
              </div>
              <p className="field-note">模型参数由内置配置自动参与计算。</p>
            </fieldset>

            <fieldset>
              <legend>负载参数</legend>
              <NumberField label="Max batched tokens" value={inputs.maxBatchedTokens} onChange={(v) => update("maxBatchedTokens", v)} />
            </fieldset>

            {model.kvCacheModelId && (
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
                <p className="field-note">标准 GQA，K/V 各存一份 Cache。</p>
              </fieldset>
            )}

            <fieldset>
              <legend>并行策略</legend>
              <div className="field-grid">
                <NumberField label="DP size" value={inputs.dpSize} onChange={(v) => update("dpSize", v)} />
                <NumberField label="TP size" value={inputs.tpSize} onChange={(v) => update("tpSize", v)} />
              </div>
              {model.supportsMtp ? (
                <div className="field-grid parallel-basic">
                  <NumberField label="Shared Expert TP" value={inputs.sharedExpertTpSize} onChange={(v) => update("sharedExpertTpSize", v)} />
                  <NumberField label="MTP layers" value={inputs.mtpLayers} onChange={(v) => update("mtpLayers", v)} />
                </div>
              ) : (
                <div className="parallel-basic">
                  <NumberField label="Shared Expert TP" value={inputs.sharedExpertTpSize} onChange={(v) => update("sharedExpertTpSize", v)} />
                </div>
              )}
              <p className="field-note">Shared Expert TP 为 1 表示不切分，不影响 EP size。</p>
              <details className="advanced-tp">
                <summary>
                  高级切分配置
                  <span
                    className="help-icon"
                    role="note"
                    aria-label="默认跟随主 TP size（与 vLLM 一致），输入数值后单独切分，清空输入框恢复跟随；各项均不影响 EP size。"
                    onClick={(event) => event.preventDefault()}
                  >?<span className="help-tooltip">默认跟随主 TP size（与 vLLM 一致），输入数值后单独切分，清空输入框恢复跟随；各项均不影响 EP size。</span></span>
                </summary>
                <div className="advanced-tp-body">
                  <div className="field-grid">
                    <NumberField label="QK / Indexer TP" value={result.attentionTp} onChange={(v) => updateOptional("attentionTpSize", v)} />
                    <NumberField label="O-proj TP" value={result.oprojTp} onChange={(v) => updateOptional("oprojTpSize", v)} />
                  </div>
                  <div className="field-grid">
                    <NumberField label="Embedding TP" value={result.embeddingTp} onChange={(v) => updateOptional("embeddingTpSize", v)} />
                    <NumberField label="LM Head TP" value={result.lmHeadTp} onChange={(v) => updateOptional("lmHeadTpSize", v)} />
                  </div>
                </div>
              </details>
            </fieldset>

            <fieldset>
              <legend>运行时</legend>
              <div className="field-grid">
                <NumberField label="Max BS" value={inputs.maxBS} onChange={(v) => update("maxBS", v)} />
                <NumberField label="图个数" value={inputs.graphCount} onChange={(v) => update("graphCount", v)} />
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
                <p>{model.weightProfile ? "单卡总显存预估（含权重）" : "单卡非权重显存预估"}</p>
              </div>
            </article>

            <article className="model-context-card">
              <div className="model-identity">
                <span className="eyebrow">ACTIVE MODEL</span>
                <strong>{model.label}</strong>
              </div>
              <div className="model-fact"><span>Hidden size</span><strong>{model.hiddenSize.toLocaleString("zh-CN")}</strong></div>
              <div className="model-fact"><span>专家总数</span><strong>{model.expertCount.toLocaleString("zh-CN")}</strong></div>
              <div className="model-fact"><span>TopK 专家</span><strong>{model.topK.toLocaleString("zh-CN")}</strong></div>
              <div className="model-fact"><span>EP size</span><strong>{epSize.toLocaleString("zh-CN")}</strong><small>TP {inputs.tpSize} × DP {inputs.dpSize}</small></div>
              <div className="model-fact"><span>本地专家数</span><strong>{localExpertNum.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}</strong><small>{model.expertCount} ÷ EP {epSize}</small></div>
              {model.source && <a className="model-source" href={model.source} target="_blank" rel="noopener noreferrer" aria-label={`查看 ${model.label} 官方配置`}>官方配置 ↗</a>}
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
                  <div
                    key={item.label}
                    title={`${item.label}: ${formatGiB(item.value)}`}
                    style={{ width: `${result.total ? item.value / result.total * 100 : 0}%`, background: item.color }}
                  />
                ))}
              </div>
              <div className="legend">
                {categories.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}
              </div>

              <div className="detail-sections">
                {model.weightProfile && (
                  <DetailSection title="权重占用" value={result.weight} tone="rose">
                    {result.weightBreakdown ? (
                      <>
                        <div className="sub-detail weight-summary">
                          <span>全量模型 {formatGiB(result.fullWeight)}</span>
                          <span>当前单卡 {formatGiB(result.weight)}</span>
                          <span>本地专家 {localExpertNum.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}</span>
                        </div>
                        <DetailRow label={`Routed Experts ${(model.weightProfile.moeWeightFormat ?? "mxfp8").toUpperCase()} payload`} value={result.weightBreakdown.routedExpertPayload} formula={`${model.weightProfile.moeLayers} × (${model.expertCount} ÷ ${epSize}) × 3 × ${model.hiddenSize} × ${model.weightProfile.expertIntermediateSize} × ${(model.weightProfile.moeWeightFormat ?? "mxfp8") === "mxfp4" ? "0.5" : "1"} B`} />
                        <DetailRow label={`Routed Experts ${(model.weightProfile.moeWeightFormat ?? "mxfp8").toUpperCase()} scale`} value={result.weightBreakdown.routedExpertScales} formula="随本地专家数 E ÷ EP 切分；每 [1, 32] block 1 B" />
                        <DetailRow label={`Attention ${model.weightProfile.qkvProjection === "qk" ? "Q+K" : "QKV"}（TP ${result.attentionTp}）`} value={result.weightBreakdown.attentionQkv} formula={model.weightProfile.qkvProjection === "qk" ? `${model.weightProfile.totalLayers} 层 × [Q: ${model.weightProfile.attentionHeads} heads ÷ TP ${result.attentionTp}；K: ${model.weightProfile.kvHeads} head 复制] × head dim ${model.weightProfile.headDim}；MXFP8 + scale，无 V 投影` : `Q/K/V 按 Attention TP ${result.attentionTp} 切分`} />
                        <DetailRow label={`Attention O-proj（TP ${result.oprojTp}）`} value={result.weightBreakdown.attentionOproj} formula={`${model.weightProfile.totalLayers} × ${model.hiddenSize} × (${model.weightProfile.attentionHeads} × ${model.weightProfile.valueHeadDim ?? model.weightProfile.headDim} ÷ TP ${result.oprojTp})；MXFP8 payload ${formatCompact(result.weightBreakdown.attentionOprojPayload)} + scale ${formatCompact(result.weightBreakdown.attentionOprojScales)}`} />
                        <DetailRow label={`Attention Indexer Q/K（TP ${result.attentionTp}）`} value={result.weightBreakdown.attentionIndexer} formula={`${model.weightProfile.sparseAttentionLayers ?? model.weightProfile.moeLayers} 层 × [Q: ${model.weightProfile.indexerHeads} heads ÷ TP ${result.attentionTp}；K: 1 head 复制] × head dim ${model.weightProfile.indexerHeadDim}；MXFP8 + scale`} />
                        <DetailRow label="Attention 辅助张量" value={result.weightBreakdown.attentionMetadata} formula="Q/K Norm、Indexer Q/K Norm 等" />
                        <DetailRow label="Dense MLP FP8 payload" value={result.weightBreakdown.denseMlpPayload} formula={`${model.weightProfile.denseLayers} × 3 × H × (${model.weightProfile.denseIntermediateSize} ÷ ${inputs.tpSize})`} />
                        <DetailRow label="Dense MLP MXFP8 scale" value={result.weightBreakdown.denseMlpScales} formula="按 TP 切分后的实际矩阵 shape 计算" />
                        <DetailRow label={`Shared Experts ${(model.weightProfile.sharedExpertWeightFormat ?? model.weightProfile.moeWeightFormat ?? "mxfp8").toUpperCase()} payload（TP ${result.sharedExpertTp}）`} value={result.weightBreakdown.sharedExpertPayload} formula={`Lm × Ns × 3 × H × I × ${(model.weightProfile.sharedExpertWeightFormat ?? model.weightProfile.moeWeightFormat ?? "mxfp8") === "mxfp4" ? "0.5" : "1"} B${result.sharedExpertTp > 1 ? ` ÷ ${result.sharedExpertTp}` : "；不切分"}`} />
                        <DetailRow label={`Shared Experts ${(model.weightProfile.sharedExpertWeightFormat ?? model.weightProfile.moeWeightFormat ?? "mxfp8").toUpperCase()} scale`} value={result.weightBreakdown.sharedExpertScales} formula={result.sharedExpertTp > 1 ? `按独立 TP ${result.sharedExpertTp} 切分后的矩阵 shape 计算` : "TP 1，不随 EP 切分，每个 Device 保留完整 scale"} />
                        <DetailRow label="Router（FP32）" value={result.weightBreakdown.router} formula={`${model.weightProfile.moeLayers} × ${model.expertCount} × (${model.hiddenSize} + 1) × 4 B`} />
                        <DetailRow label="Norm" value={result.weightBreakdown.norms} formula="BF16 Norm tensors" />
                        <DetailRow label="Embedding" value={result.weightBreakdown.embedding} formula={`${result.weightBreakdown.paddedVocab} ÷ ${result.embeddingTp} × ${model.hiddenSize} × 2 B`} />
                        <DetailRow label="LM Head" value={result.weightBreakdown.lmHead} formula={`${result.weightBreakdown.paddedVocab} ÷ ${result.lmHeadTp} × ${model.hiddenSize} × 2 B`} />
                        {model.supportsMtp && (
                          <DetailRow label={`MTP 权重（${inputs.mtpLayers} 层）`} value={result.weightBreakdown.mtpWeight} formula={`${inputs.mtpLayers} × 单层 ${formatCompact(result.weightBreakdown.mtpPerLayer)}；复制稀疏 MoE Block + MTP projection/norm，复用 Embedding/LM Head`} />
                        )}
                        <DetailRow label="Misc" value={result.weightBreakdown.misc} formula="RoPE inv_freq 等 buffer" />
                      </>
                    ) : (
                      <p className="validation-note">当前并行组合不满足专家数、Attention Heads、中间维度或词表大小的整除要求。</p>
                    )}
                  </DetailSection>
                )}

                {result.kvCacheInfo && result.kvCacheBreakdown && (
                  <DetailSection title="KV + Index Cache" value={result.kvCache} tone="cyan">
                    <DetailRow
                      label={`K/V Cache（${CACHE_PRECISIONS[inputs.kvPrecision ?? DEFAULTS.kvPrecision].label}）`}
                      value={result.kvCacheBreakdown.kvCache}
                      formula={`${inputs.kvCacheTokens ?? DEFAULTS.kvCacheTokens} × ${inputs.kvCacheSequences ?? DEFAULTS.kvCacheSequences} × ${result.kvCacheInfo.layers}${model.supportsMtp ? ` + MTP ${inputs.mtpLayers}` : ""} × ${result.kvCacheBreakdown.kvCopies} × (${result.kvCacheInfo.kvHeads} ÷ TP ${result.attentionTp}) × ${result.kvCacheInfo.headDim} × ${result.kvCacheBreakdown.kvBytesPerElement} B；有效 ${result.kvCacheBreakdown.effectiveKvLayers} 层，K/V 各存一份，随 TP 切分`}
                    />
                    <DetailRow
                      label={`Index Cache（${CACHE_PRECISIONS[inputs.indexCachePrecision ?? DEFAULTS.indexCachePrecision].label}）`}
                      value={result.kvCacheBreakdown.indexCache}
                      formula={`${inputs.kvCacheTokens ?? DEFAULTS.kvCacheTokens} × ${inputs.kvCacheSequences ?? DEFAULTS.kvCacheSequences} × ${result.kvCacheInfo.sparseLayers}${model.supportsMtp ? ` + MTP ${inputs.mtpLayers}` : ""} × ${result.kvCacheInfo.indexHeadDim} × ${result.kvCacheBreakdown.indexBytesPerElement} B；有效 ${result.kvCacheBreakdown.effectiveIndexLayers} 层，Index K 随 TP 复制，不切分`}
                    />
                  </DetailSection>
                )}

                <DetailSection title="激活占用" value={result.activation} tone="coral">
                  <DetailRow label="Hidden states + residual" value={result.hiddenResidual} formula={`2 × 2 B × ${inputs.maxBatchedTokens} × ${model.hiddenSize}`} />
                  <DetailRow label="4 份 MoE 激活 buffer" value={result.moeBuffers} formula={`4 × 2 B × ${inputs.dpSize} × ${inputs.maxBatchedTokens} × ${model.topK} ÷ ${epSize} × ${model.hiddenSize}`} />
                </DetailSection>

                <DetailSection title="HCCL buffer" value={result.hccl} tone="blue">
                  <DetailRow label="DP buffer" value={result.hcclDP} formula={`max(ceil((${inputs.dpSize} + 1) × 4 ÷ 1024²), 50) × 2 MB`} />
                  <DetailRow label="TP buffer" value={result.hcclTP} formula="200 MB × 2" />
                  <DetailRow label={`EP buffer（本地专家 ${localExpertNum.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}）`} value={result.hcclEP} formula={`2 × (${model.expertCount} ÷ ${epSize} × Max BS × EP × 480Align512 + K × Max BS × Align512)`} />
                  <div className="sub-detail">
                    <span>Dispatch {formatMiB(result.epDispatch)}</span>
                    <span>Combine {formatMiB(result.epCombine)}</span>
                    <span>480Align512 = {result.alignedDispatch.toLocaleString("zh-CN")} B</span>
                    <span>Align512 = {result.alignedCombine.toLocaleString("zh-CN")} B</span>
                  </div>
                </DetailSection>

                <DetailSection title="其他运行时" value={result.graph + result.cann + result.deviceOS} tone="violet">
                  <DetailRow label={`图占用（${inputs.graphCount} 张）`} value={result.graph} formula={`${inputs.graphCount} ÷ 5 × 0.27 GB`} />
                  <DetailRow label="CANN + PTA + 算子" value={result.cann} formula={`${inputs.cannGB} GiB 预估值`} />
                  <DetailRow label="Device OS 固定占用" value={result.deviceOS} formula="4.25 × 1024³ bytes = 4.25 GiB" />
                </DetailSection>
              </div>
            </article>

            <p className="method-note"><strong>口径说明</strong> 所有结果均为单卡估算并统一显示为 GiB。MiniMax M3 权重包含 MXFP8 scale；EP 自动等于 TP × DP。KV Cache 与 Index Cache 按缓存 token 总数估算，不包含分配器开销。</p>
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

function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: { value: string; label: string }[] }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
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
