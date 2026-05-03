export interface PlotlyFigure {
  data: object[]
  layout: object
}

export interface AppState {
  has_data: boolean
  has_models: boolean
  has_persona_models: boolean
  trained_personas: string[]
  has_simulation: boolean
  has_multiseed: boolean
  has_crossuser: boolean
  active_persona: string
}

export interface Job {
  status: 'running' | 'done' | 'error'
  progress: number
  result: object | null
  error: string | null
}

export interface SimMetrics {
  qoe_score: number
  avg_bitrate_mbps: number
  rebuffer_events: number
  rebuffer_secs: number
  quality_switches: number
  mean_quality_idx: number
  adapt_accuracy: number
  quality_mismatch: number
  quality_distribution: Record<string, number>
}

export interface PersonaMetricRow {
  persona_name: string
  accuracy: number
  f1_macro: number
  epochs_trained: number
}

export interface AllMetricsResponse {
  personas: Record<string, PersonaMetricRow>
  baselines: Record<string, { accuracy: number; f1_macro: number }>
  feature_importance: Record<string, number>
}

export interface CrossUserRow {
  Persona: string
  'Personal Acc': number
  'Generic Acc': number
  'Delta Acc': number
  'Personal F1': number
  'Generic F1': number
  'Delta F1': number
}
