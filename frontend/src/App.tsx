import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import { DataPage } from './pages/DataPage'
import { ModelsPage } from './pages/ModelsPage'
import { StreamingPage } from './pages/StreamingPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { ExperimentPage } from './pages/ExperimentPage'
import type { AppState } from './types'

type Page = 'data' | 'models' | 'streaming' | 'analysis' | 'experiment'

const NAV: { id: Page; label: string }[] = [
  { id: 'data',       label: 'Data' },
  { id: 'models',     label: 'Models' },
  { id: 'streaming',  label: 'Streaming' },
  { id: 'analysis',   label: 'Analysis' },
  { id: 'experiment', label: 'Experiment' },
]

const EMPTY_STATE: AppState = {
  has_data: false, has_models: false, has_persona_models: false,
  trained_personas: [], has_simulation: false,
  has_multiseed: false, has_crossuser: false, active_persona: '',
}

function StatusBadge({ label, ok, partial }: { label: string; ok: boolean; partial?: boolean }) {
  const cls = ok
    ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20'
    : partial
    ? 'text-amber-400 bg-amber-400/10 border-amber-400/20'
    : 'text-zinc-600 bg-zinc-800/60 border-zinc-700/40'
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-xs font-medium ${cls}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-emerald-400' : partial ? 'bg-amber-400' : 'bg-zinc-600'}`} />
      {label}
    </span>
  )
}

export default function App() {
  const [page, setPage]         = useState<Page>('data')
  const [personas, setPersonas] = useState<Record<string, { name: string }>>({})
  const [appState, setAppState] = useState<AppState>(EMPTY_STATE)

  const refreshState = useCallback(() => {
    api.getState().then(setAppState).catch(console.error)
  }, [])

  useEffect(() => {
    api.getPersonas().then(setPersonas).catch(console.error)
    refreshState()
  }, [refreshState])

  const nTrained = appState.trained_personas.length
  const nPersonas = Object.keys(personas).length

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden">

      {/* Left navigation */}
      <nav className="w-48 flex-shrink-0 bg-zinc-900 border-r border-zinc-800 flex flex-col">
        <div className="px-5 py-5 border-b border-zinc-800">
          <div className="text-sm font-semibold text-zinc-100 leading-tight">Intelligent Mobile</div>
          <div className="text-xs text-zinc-500 mt-0.5">Streaming Dashboard</div>
        </div>

        <div className="flex-1 py-2">
          {NAV.map(item => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={`w-full text-left px-5 py-2.5 text-sm transition-colors border-l-2 ${
                page === item.id
                  ? 'border-blue-500 text-zinc-100 bg-zinc-800/60'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/30'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 space-y-1.5">
          <div className="text-xs text-zinc-600 uppercase tracking-widest mb-2">Pipeline</div>
          <div className="flex flex-col gap-1.5">
            <StatusBadge label="Data" ok={appState.has_data} />
            <StatusBadge
              label={nTrained > 0 ? `Models ${nTrained}/${nPersonas}` : 'Models'}
              ok={nTrained === nPersonas && nPersonas > 0}
              partial={nTrained > 0 && nTrained < nPersonas}
            />
            <StatusBadge label="Simulation" ok={appState.has_simulation} />
          </div>
        </div>
      </nav>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="h-11 flex-shrink-0 bg-zinc-900/80 border-b border-zinc-800 flex items-center px-6 gap-2">
          <span className="text-xs text-zinc-600 mr-1">Status</span>
          <StatusBadge label="Data" ok={appState.has_data} />
          <StatusBadge
            label={nTrained > 0 ? `Models ${nTrained}/${nPersonas}` : 'No models'}
            ok={nTrained === nPersonas && nPersonas > 0}
            partial={nTrained > 0 && nTrained < nPersonas}
          />
          <StatusBadge label="Simulation" ok={appState.has_simulation} />
          {appState.has_crossuser && <StatusBadge label="Experiment" ok />}
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          {page === 'data' && (
            <DataPage personas={personas} appState={appState} onStateChange={refreshState} />
          )}
          {page === 'models' && (
            <ModelsPage personas={personas} appState={appState} onStateChange={refreshState} />
          )}
          {page === 'streaming' && (
            <StreamingPage personas={personas} appState={appState} onStateChange={refreshState} />
          )}
          {page === 'analysis' && (
            <AnalysisPage personas={personas} appState={appState} onStateChange={refreshState} />
          )}
          {page === 'experiment' && (
            <ExperimentPage appState={appState} onStateChange={refreshState} />
          )}
        </main>
      </div>
    </div>
  )
}
