// safetensors 头部（header）解析：从文件头部字节提取每个张量的
// name / dtype / shape / data_offsets，并计算字节数。
//
// safetensors 格式：文件开头 8 字节为 little-endian u64（header JSON 的字节
// 长度），随后紧跟该长度的 UTF-8 JSON。JSON 的顶层除 `__metadata__` 外，每个
// key 是一个张量名，value 形如 `{ dtype, shape, data_offsets: [begin, end] }`。

export type SafetensorInfo = {
  name: string;
  /** 存储 dtype，如 "F32" / "BF16" / "I8" / "F8_E4M3"，决定每元素字节数 */
  dtype: string;
  shape: number[];
  dataOffsets: [number, number];
  /** numel = shape 各项乘积 */
  numel: number;
  /** 字节数 = numel × 每元素字节数 */
  bytes: number;
};

export type SafetensorsHeader = {
  metadata: Record<string, string> | null;
  tensors: SafetensorInfo[];
};

const DTYPE_BYTES: Record<string, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
  F8_E4M3: 1,
  F8_E5M2: 1,
  I4: 0.5,
  U4: 0.5,
};

/** dtype → 每元素字节数；未知 dtype 返回 0。 */
export function dtypeBytes(dtype: string): number {
  return DTYPE_BYTES[dtype.toUpperCase()] ?? 0;
}

/**
 * 解析 safetensors header。`buf` 需要包含“8 字节长度字段 + 完整 header JSON”
 * 的字节（即至少 8 + headerLen 字节，多余部分被忽略）。
 */
export function parseSafetensorsHeader(buf: ArrayBuffer): SafetensorsHeader {
  if (buf.byteLength < 8) {
    throw new Error("safetensors header 不足 8 字节");
  }
  const view = new DataView(buf);
  const headerLen = Number(view.getBigUint64(0, true));
  if (buf.byteLength < 8 + headerLen) {
    throw new Error(
      `safetensors header 声明 ${headerLen} 字节，但仅提供 ${buf.byteLength - 8} 字节`,
    );
  }
  const headerBytes = new Uint8Array(buf, 8, headerLen);
  const headerText = new TextDecoder().decode(headerBytes);
  let header: unknown;
  try {
    header = JSON.parse(headerText);
  } catch (error) {
    throw new Error(`safetensors header JSON 解析失败: ${(error as Error).message}`);
  }
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new Error("safetensors header 顶层必须是对象");
  }

  const tensors: SafetensorInfo[] = [];
  let metadata: Record<string, string> | null = null;
  for (const [name, info] of Object.entries(header as Record<string, unknown>)) {
    if (name === "__metadata__") {
      metadata = (info ?? null) as Record<string, string> | null;
      continue;
    }
    const t = info as { dtype?: string; shape?: number[]; data_offsets?: number[] };
    if (!t || !t.dtype || !Array.isArray(t.shape) || !Array.isArray(t.data_offsets)) {
      continue;
    }
    const numel = t.shape.reduce((a, b) => a * b, 1);
    tensors.push({
      name,
      dtype: t.dtype,
      shape: t.shape,
      dataOffsets: [t.data_offsets[0], t.data_offsets[1]],
      numel,
      bytes: numel * dtypeBytes(t.dtype),
    });
  }
  return { metadata, tensors };
}