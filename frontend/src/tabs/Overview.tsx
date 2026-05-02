export function Overview() {
  const personas = [
    ['Urban Commuter (Alice)', 'Bimodal — excellent at office/home, terrible in subway'],
    ['Suburban Student (Bob)', 'Mostly low congestion (campus 5G), medium at home'],
    ['Rural Remote (Charlie)', 'Consistently weak signal → medium–high congestion'],
    ['Dense Urban (Dana)', 'Strong signal, high cell load → persistent medium'],
    ['Frequent Traveler (Eve)', 'Highly variable; frequent handoffs at speed'],
  ]
  const steps = [
    ['Persona Generator', 'Five user archetypes, each with home / work / commute anchors. Metrics emerge from a capacity model coupling RSSI and cell load — labels are not pre-assigned'],
    ['Personal LSTM', "PyTorch 2-layer LSTM trained on one user's data. Learns that user's specific location fingerprints and schedule"],
    ['KNN / RF baselines', 'Scikit-learn classifiers trained on the same persona data for fair comparison'],
    ['Streaming Engine', 'Buffer-based ABR simulation comparing rule-based, rate-based, and ML-based controllers'],
    ['Cross-User Experiment', "Proves personalisation: personal model vs generic model on each user's held-out test data"],
  ]

  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold text-slate-100 mb-2">Intelligent Mobile Streaming</h1>
        <p className="text-slate-400">Per-user ML-based adaptive bitrate streaming in mobile networks · CSE 476 Term Project</p>
      </div>

      <section>
        <h2 className="text-lg font-semibold text-slate-200 mb-3">System Overview</h2>
        <div className="overflow-auto rounded-lg border border-slate-700">
          <table className="w-full text-sm">
            <thead className="bg-slate-800">
              <tr>
                {['Step', 'Module', 'Description'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-slate-300 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {steps.map(([mod, desc], i) => (
                <tr key={i} className="hover:bg-slate-800/50">
                  <td className="px-4 py-2.5 text-slate-400 font-mono">{i + 1}</td>
                  <td className="px-4 py-2.5 text-indigo-400 font-medium whitespace-nowrap">{mod}</td>
                  <td className="px-4 py-2.5 text-slate-300">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-200 mb-3">Quick Start</h2>
        <ol className="space-y-2 text-slate-300">
          {[
            'Generate All Persona Data in the sidebar',
            'Train on Selected Persona — LSTM + KNN + RF all fitted on the same data',
            'Run Simulation — see quality timelines in Streaming tab',
            'Switch to Comparison for multi-seed statistical analysis',
            'Use Cross-User to run the personalisation proof experiment',
          ].map((s, i) => (
            <li key={i} className="flex gap-3">
              <span className="text-indigo-400 font-mono shrink-0">{i + 1}.</span>
              <span>{s}</span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-200 mb-3">The Five Personas</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {personas.map(([name, desc]) => (
            <div key={name} className="bg-slate-800 border border-slate-700 rounded-lg p-4">
              <div className="text-indigo-400 font-medium mb-1">{name}</div>
              <div className="text-sm text-slate-400">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-200 mb-2">Network Parameters</h2>
        <div className="flex flex-wrap gap-2">
          {['throughput_dl', 'throughput_ul', 'latency', 'packet_loss', 'jitter', 'signal_strength', 'mobility_speed'].map(f => (
            <code key={f} className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-indigo-300">{f}</code>
          ))}
        </div>
      </section>
    </div>
  )
}
