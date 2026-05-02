import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import { useJob } from './hooks/useJob'
import { Sidebar } from './components/Sidebar'
import { Overview } from './tabs/Overview'
import { PersonaData } from './tabs/PersonaData'
import { ModelTraining } from './tabs/ModelTraining'
import { Streaming } from './tabs/Streaming'
import { Comparison } from './tabs/Comparison'
import { CrossUser } from './tabs/CrossUser'
import type { AppState, SidebarParams } from './types'
import type { SimMetrics } from './tabs/Streaming'

const TAB_LABELS = [
  '🏠 Overview',
  '📊 Persona Data',
  '🤖 Model Training',
  '▶ Streaming',
  '📈 Comparison',
  '🧪 Cross-User',
]

const DEFAULT_PARAMS: SidebarParams = {
  activePersona: '',
  nSessions: 15, sessionDuration: 300, dataSeed: 42,
  lstmWindow: 15, lstmEpochs: 25, lstmPatience: 7,
  knnK: 5, rfTrees: 100,
  simHour: 8.5, simDuration: 300, simSeed: 7,
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [personas, setPersonas] = useState<Record<string, { name: string }>>({})
  const [appState, setAppState] = useState<AppState>({
    has_data: false, has_models: false, has_simulation: false,
    has_multiseed: false, has_crossuser: false, active_persona: '',
  })
  const [params, setParams] = useState<SidebarParams>(DEFAULT_PARAMS)
  const [simMetrics, setSimMetrics] = useState<{
    rule: SimMetrics; threshold: SimMetrics; ml: SimMetrics
  } | null>(null)
  const [simError, setSimError] = useState<string | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)

  const refreshState = useCallback(() => {
    api.getState().then(setAppState).catch(console.error)
  }, [])

  const job = useJob(refreshState)

  useEffect(() => {
    api.getPersonas().then(p => {
      setPersonas(p)
      const first = Object.keys(p)[0]
      if (first) setParams(prev => ({ ...prev, activePersona: first }))
    }).catch(console.error)
    refreshState()
  }, [refreshState])

  const handleParamChange = (key: keyof SidebarParams, value: number | string) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  const handleGenerateData = async () => {
    setJobError(null)
    try {
      const { job_id } = await api.generateData({
        n_sessions: params.nSessions,
        session_duration: params.sessionDuration,
        seed: params.dataSeed,
      })
      job.start(job_id)
    } catch (e: unknown) {
      setJobError(e instanceof Error ? e.message : String(e))
    }
  }

  const handleTrainModels = async () => {
    setJobError(null)
    try {
      const { job_id } = await api.trainModels({
        persona_key: params.activePersona,
        lstm_window: params.lstmWindow,
        lstm_epochs: params.lstmEpochs,
        lstm_patience: params.lstmPatience,
        knn_k: params.knnK,
        rf_trees: params.rfTrees,
      })
      job.start(job_id)
    } catch (e: unknown) {
      setJobError(e instanceof Error ? e.message : String(e))
    }
  }

  const handleRunSimulation = async () => {
    setSimError(null)
    setSimMetrics(null)
    try {
      const metrics = await api.runSimulation({
        persona_key: params.activePersona,
        hour: params.simHour,
        duration: params.simDuration,
        seed: params.simSeed,
      }) as { rule: SimMetrics; threshold: SimMetrics; ml: SimMetrics }
      setSimMetrics(metrics)
      refreshState()
    } catch (e: unknown) {
      setSimError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        params={params}
        personas={personas}
        appState={appState}
        jobProgress={job.progress}
        jobRunning={job.isRunning}
        onParamChange={handleParamChange}
        onGenerateData={handleGenerateData}
        onTrainModels={handleTrainModels}
        onRunSimulation={handleRunSimulation}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Tab bar */}
        <div className="border-b border-slate-800 bg-slate-900 px-4">
          <div className="flex gap-1 overflow-x-auto">
            {TAB_LABELS.map((label, i) => (
              <button
                key={i}
                onClick={() => setActiveTab(i)}
                className={`px-4 py-3 text-sm whitespace-nowrap border-b-2 transition-colors ${
                  activeTab === i
                    ? 'border-indigo-500 text-indigo-400'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Error banners */}
        {(jobError || job.error || simError) && (
          <div className="bg-red-900/40 border-b border-red-700 px-6 py-2 text-sm text-red-300">
            {jobError ?? job.error ?? simError}
          </div>
        )}

        {/* Tab content */}
        <main className="flex-1 overflow-y-auto p-6">
          {activeTab === 0 && <Overview />}
          {activeTab === 1 && (
            <PersonaData
              activePersona={params.activePersona}
              hasData={appState.has_data}
              personas={personas}
            />
          )}
          {activeTab === 2 && (
            <ModelTraining
              hasModels={appState.has_models}
              activePersona={appState.active_persona || params.activePersona}
              personas={personas}
            />
          )}
          {activeTab === 3 && (
            <Streaming
              hasSimulation={appState.has_simulation}
              simMetrics={simMetrics}
            />
          )}
          {activeTab === 4 && (
            <Comparison
              hasSimulation={appState.has_simulation}
              activePersona={params.activePersona}
            />
          )}
          {activeTab === 5 && (
            <CrossUser hasData={appState.has_data} />
          )}
        </main>
      </div>
    </div>
  )
}
