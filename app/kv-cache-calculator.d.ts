declare module "kv-cache-calculator" {
  export type UpstreamCacheGroup = {
    role: string;
    label: string;
    elements?: number;
    bytes: number;
  };

  export type UpstreamCalculateResult = {
    totalBytes: number;
    kvBytes: number;
    indexerBytes: number;
    cacheGroups: UpstreamCacheGroup[];
  };

  const core: {
    calculate: (
      model: unknown,
      input: Record<string, unknown>,
      options?: unknown,
    ) => UpstreamCalculateResult;
    formatBytes: (bytes: number) => string;
  };
  export default core;
}

declare module "kv-cache-calculator/models" {
  export type UpstreamModel = {
    id: string;
    label: string;
    family: string;
    formula: string;
    default_tokens?: number;
    fields: Record<string, unknown>;
  };

  const data: {
    models: UpstreamModel[];
  };
  export default data;
}
