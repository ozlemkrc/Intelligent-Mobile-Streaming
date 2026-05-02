export interface PlotlyFigure {
  data: object[]
  layout: object
}

export interface AppState {
  has_data: boolean
  has_models: boolean
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

export interface SidebarParams {
  activePersona: string
  nSessions: number
  sessionDuration: number
  dataSeed: number
  lstmWindow: number
  lstmEpochs: number
  lstmPatience: number
  knnK: number
  rfTrees: number
  simHour: number
  simDuration: number
  simSeed: number
}
