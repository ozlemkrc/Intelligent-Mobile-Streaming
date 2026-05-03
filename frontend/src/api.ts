import type { Job, AppState, AllMetricsResponse, SimMetrics } from './types'

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? res.statusText)
  }
  return res.json()
}

const post = <T>(path: string, body: object) =>
  req<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  getPersonas:  () => req<Record<string, { name: string }>>('/api/personas'),
  getFeatures:  () => req<{ features: string[]; classes: string[] }>('/api/features'),
  getState:     () => req<AppState>('/api/state'),
  getJob:       (id: string) => req<Job>(`/api/jobs/${id}`),

  // Data
  generateData: (p: { n_sessions: number; session_duration: number; seed: number }) =>
    post<{ job_id: string }>('/api/data/generate', p),
  getDataSummary:      () => req('/api/data/summary'),
  getDistributionChart: (persona: string, feature: string) =>
    req(`/api/data/${persona}/charts/distribution?feature=${feature}`),
  getPieCharts:  (persona: string) => req(`/api/data/${persona}/charts/pies`),
  getScatterChart: (persona: string, features: string) =>
    req(`/api/data/${persona}/charts/scatter?features=${encodeURIComponent(features)}`),

  // Models — all personas
  trainAllPersonas: (p: { lstm_window: number; lstm_epochs: number; lstm_patience: number; knn_k: number; rf_trees: number }) =>
    post<{ job_id: string }>('/api/models/train-all', p),
  getAllMetrics:    () => req<AllMetricsResponse>('/api/models/all-metrics'),
  getPersonaLossChart:      (key: string) => req(`/api/models/${key}/charts/loss`),
  getPersonaConfusionChart: (key: string) => req(`/api/models/${key}/charts/confusion`),

  // Simulation
  runSimulation: (p: { persona_key: string; hour: number; duration: number; seed: number }) =>
    post<{ rule: SimMetrics; threshold: SimMetrics; ml: SimMetrics }>('/api/simulation/run', p),
  getSimulationCharts: () => req('/api/simulation/charts'),

  // Comparison
  getComparisonData: () => req('/api/comparison/data'),

  // Multi-seed
  runMultiSeed: (p: { persona_key: string; n_runs: number; duration: number }) =>
    post<{ job_id: string }>('/api/multiseed/run', p),
  getMultiSeedData: (metric: string, baseline: string) =>
    req(`/api/multiseed/data?metric=${metric}&baseline=${encodeURIComponent(baseline)}`),

  // Cross-user
  runCrossUser: (p: { epochs: number; window: number; patience: number }) =>
    post<{ job_id: string }>('/api/crossuser/run', p),
  getCrossUserData: () => req('/api/crossuser/data'),
}
