"use client";

import { useMemo, useState } from "react";
import { DEFAULT_MODEL_ID, MODELS } from "./models";
import { calculateMiniMaxM3Weight } from "./weight-model";

type Inputs = {
  maxBatchedTokens: number;
  dpSize: number;
  tpSize: number;
  maxBS: number;
  graphCount: number;
  cannGB: number;
  enableSharedExpertTp: boolean;
};

const DEFAULTS: Inputs = {
  maxBatchedTokens: 4096,
  dpSize: 8,
  tpSize: 8,
  maxBS: 128,
  graphCount: 5,
  cannGB: 1,
  enableSharedExpertTp: false,
};

const GB = 1_000_000_000;
const GIB = 1024 ** 3;
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

    const hcclDP = Math.max(Math.ceil(((dp + 1) * 4) / MIB), 50) * 2 * MIB;
    const hcclTP = 200 * 2 * MIB;
    const alignedDispatch = align480To512(align(2 * H, 32) + 64);
    const alignedCombine = align(2 * H, 512);
    const epDispatch = localExperts * maxBS * ep * alignedDispatch;
    const epCombine = K * maxBS * alignedCombine;
    const hcclEP = 2 * (epDispatch + epCombine);
    const hcclBuffsizeMiB = Math.max(1, Math.ceil(hcclEP / MIB));
    const hccl = hcclDP + hcclTP + hcclEP;

    const graph = (safe(inputs.graphCount) / 5) * 0.27 * GB;
    const cann = safe(inputs.cannGB) * GB;
    const deviceOS = 4.25 * GIB;
    const profile = model.weightProfile;
    const tp = Math.max(1, Math.floor(safe(inputs.tpSize, 1)));
    const weightConfigValid = Boolean(
      profile
      && model.expertCount % ep === 0
      && profile.attentionHeads % tp === 0
      && profile.kvHeads % tp === 0
      && profile.indexerHeads % tp === 0
      && profile.denseIntermediateSize % tp === 0
      && (!inputs.enableSharedExpertTp || profile.expertIntermediateSize % tp === 0)
    );

    let weight = 0;
    let weightBreakdown = null;
    if (profile && weightConfigValid) {
      weightBreakdown = calculateMiniMaxM3Weight({
        profile,
        hiddenSize: H,
        expertCount: model.expertCount,
        tpSize: tp,
        epSize: ep,
        enableSharedExpertTp: inputs.enableSharedExpertTp,
      });
      weight = weightBreakdown.total;
    }

    const total = weight + activation + hccl + graph + cann + deviceOS;

    return {
      activation,
      hiddenResidual,
      moeBuffers,
      hccl,
      hcclDP,
      hcclTP,
      hcclEP,
      hcclBuffsizeMiB,
      epDispatch: 2 * epDispatch,
      epCombine: 2 * epCombine,
      alignedDispatch,
      alignedCombine,
      graph,
      cann,
      deviceOS,
      weight,
      weightBreakdown,
      weightConfigValid,
      total,
    };
  }, [inputs, epSize, localExpertNum, model]);

  const update = (key: keyof Inputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: Number(value) }));
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
              <div className="field-grid">
                <SelectField label="模型族" value={family} onChange={changeFamily} options={families.map((item) => ({ value: item, label: item }))} />
                <SelectField label="模型" value={modelId} onChange={changeModel} options={familyModels.map((item) => ({ value: item.id, label: item.label }))} />
              </div>
              <p className="field-note">模型参数从官方配置自动读取并参与计算。</p>
            </fieldset>

            <fieldset>
              <legend>负载参数</legend>
              <NumberField label="Max batched tokens" value={inputs.maxBatchedTokens} onChange={(v) => update("maxBatchedTokens", v)} />
            </fieldset>

            <fieldset>
              <legend>并行策略</legend>
              <div className="field-grid">
                <NumberField label="DP size" value={inputs.dpSize} onChange={(v) => update("dpSize", v)} />
                <NumberField label="TP size" value={inputs.tpSize} onChange={(v) => update("tpSize", v)} />
              </div>
              <ToggleField
                label="Shared Expert TP"
                checked={inputs.enableSharedExpertTp}
                onChange={(checked) => setInputs((current) => ({ ...current, enableSharedExpertTp: checked }))}
                disabled={!model.weightProfile}
              />
              <p className="field-note">仅 MiniMax M3 权重建模使用；默认关闭，开启后 Shared Expert 按 TP 切分。</p>
            </fieldset>

            <fieldset>
              <legend>运行时</legend>
              <div className="field-grid">
                <NumberField label="Max BS" value={inputs.maxBS} onChange={(v) => update("maxBS", v)} />
                <NumberField label="图个数" value={inputs.graphCount} onChange={(v) => update("graphCount", v)} />
              </div>
              <NumberField label="CANN + PTA + 算子预估（GB）" value={inputs.cannGB} step="0.1" onChange={(v) => update("cannGB", v)} />
              <p className="field-note">该项默认按 1 GB 预留，实际占用通常低于此值。</p>
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
              <a className="model-source" href={model.source} target="_blank" rel="noopener noreferrer" aria-label={`查看 ${model.label} 官方配置`}>官方配置 ↗</a>
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
                        <DetailRow label="Routed Experts FP8 payload" value={result.weightBreakdown.routedExpertPayload} formula={`${model.weightProfile.moeLayers} × (${model.expertCount} ÷ ${epSize}) × 3 × ${model.hiddenSize} × ${model.weightProfile.expertIntermediateSize}`} />
                        <DetailRow label="Routed Experts MXFP8 scale" value={result.weightBreakdown.routedExpertScales} formula="随本地专家数 E ÷ EP 切分；每 [1, 32] block 1 B" />
                        <DetailRow label="Attention FP8 payload" value={result.weightBreakdown.attentionPayload} formula={`Q/K/V/O 与 Indexer Q 按 TP ${inputs.tpSize} 切分；Indexer K 复制`} />
                        <DetailRow label="Attention MXFP8 scale" value={result.weightBreakdown.attentionScales} formula="按各 Rank 实际 Q/K/V/O/Indexer 矩阵 shape 精确计算" />
                        <DetailRow label="Attention 辅助张量" value={result.weightBreakdown.attentionMetadata} formula="Q/K Norm、Indexer Q/K Norm 等" />
                        <DetailRow label="Dense MLP FP8 payload" value={result.weightBreakdown.denseMlpPayload} formula={`${model.weightProfile.denseLayers} × 3 × H × (${model.weightProfile.denseIntermediateSize} ÷ ${inputs.tpSize})`} />
                        <DetailRow label="Dense MLP MXFP8 scale" value={result.weightBreakdown.denseMlpScales} formula="按 TP 切分后的实际矩阵 shape 计算" />
                        <DetailRow label={`Shared Experts FP8 payload（${inputs.enableSharedExpertTp ? `TP ${inputs.tpSize}` : "不切分"}）`} value={result.weightBreakdown.sharedExpertPayload} formula={`Lm × Ns × 3 × H × I${inputs.enableSharedExpertTp ? ` ÷ ${inputs.tpSize}` : ""}`} />
                        <DetailRow label="Shared Experts MXFP8 scale" value={result.weightBreakdown.sharedExpertScales} formula={inputs.enableSharedExpertTp ? `按 TP ${inputs.tpSize} 切分后的实际矩阵 shape 计算` : "不随 EP 切分，每个 Device 保留完整 scale"} />
                        <DetailRow label="Router（FP32）" value={result.weightBreakdown.router} formula={`${model.weightProfile.moeLayers} × ${model.expertCount} × (${model.hiddenSize} + 1) × 4 B`} />
                        <DetailRow label="Norm" value={result.weightBreakdown.norms} formula="BF16 Norm tensors" />
                        <DetailRow label="Embedding" value={result.weightBreakdown.embedding} formula={`${result.weightBreakdown.paddedVocab} ÷ ${inputs.tpSize} × ${model.hiddenSize} × 2 B`} />
                        <DetailRow label="LM Head" value={result.weightBreakdown.lmHead} formula={`${result.weightBreakdown.paddedVocab} ÷ ${inputs.tpSize} × ${model.hiddenSize} × 2 B`} />
                        <DetailRow label="Misc" value={result.weightBreakdown.misc} formula="RoPE inv_freq 等 buffer" />
                      </>
                    ) : (
                      <p className="validation-note">当前 TP/DP 组合不满足专家数、Attention Heads 或中间维度的整除要求。</p>
                    )}
                  </DetailSection>
                )}

                <DetailSection title="激活占用" value={result.activation} tone="coral">
                  <DetailRow label="Hidden states + residual" value={result.hiddenResidual} formula={`2 × 2 B × ${inputs.maxBatchedTokens} × ${model.hiddenSize}`} />
                  <DetailRow label="4 份 MoE 激活 buffer" value={result.moeBuffers} formula={`4 × 2 B × ${inputs.dpSize} × ${inputs.maxBatchedTokens} × ${model.topK} ÷ ${epSize} × ${model.hiddenSize}`} />
                </DetailSection>

                <DetailSection title="HCCL buffer" value={result.hccl} tone="blue">
                  <DetailRow label="DP buffer" value={result.hcclDP} formula={`max(ceil((${inputs.dpSize} + 1) × 4 ÷ 1024²), 50) × 2 MiB`} />
                  <DetailRow label="TP buffer" value={result.hcclTP} formula="200 MiB × 2" />
                  <DetailRow label={`EP buffer（本地专家 ${localExpertNum.toLocaleString("zh-CN", { maximumFractionDigits: 2 })}）`} value={result.hcclEP} formula={`2 × (${model.expertCount} ÷ ${epSize} × Max BS × EP × 480Align512 + K × Max BS × Align512)`} />
                  <DetailRow label={`建议 export HCCL_BUFFSIZE=${result.hcclBuffsizeMiB}`} value={result.hcclBuffsizeMiB * MIB} formula={`ceil(EP buffer ÷ 1024²) = ${result.hcclBuffsizeMiB}`} />
                  <div className="sub-detail">
                    <span>Dispatch {formatMiB(result.epDispatch)}</span>
                    <span>Combine {formatMiB(result.epCombine)}</span>
                    <span>480Align512 = {result.alignedDispatch.toLocaleString("zh-CN")} B</span>
                    <span>Align512 = {result.alignedCombine.toLocaleString("zh-CN")} B</span>
                  </div>
                </DetailSection>

                <DetailSection title="其他运行时" value={result.graph + result.cann + result.deviceOS} tone="violet">
                  <DetailRow label={`图占用（${inputs.graphCount} 张）`} value={result.graph} formula={`${inputs.graphCount} ÷ 5 × 0.27 GB`} />
                  <DetailRow label="CANN + PTA + 算子" value={result.cann} formula={`${inputs.cannGB} GB 预估值`} />
                  <DetailRow label="Device OS 固定占用" value={result.deviceOS} formula="4.25 × 1024³ bytes = 4.25 GiB" />
                </DetailSection>
              </div>
            </article>

            <p className="method-note"><strong>口径说明</strong> 所有结果均为单卡估算并统一显示为 GiB。MiniMax M3 权重默认包含 MXFP8 scale；EP 自动等于 TP × DP。其他模型暂不计算权重。建议 <code>export HCCL_BUFFSIZE</code> 取 EP buffer 向上取整到 MiB（与 CANN DispatchV2 / vLLM-Ascend MC2 比较；combine 按 topK，不含本卡 shared 专家）。</p>
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

function ToggleField({ label, checked, onChange, disabled = false }: { label: string; checked: boolean; onChange: (checked: boolean) => void; disabled?: boolean }) {
  return (
    <label className={`toggle-field${disabled ? " disabled" : ""}`}>
      <span>{label}</span>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <i aria-hidden="true"><b /></i>
      <em>{checked ? "已开启" : "已关闭"}</em>
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
