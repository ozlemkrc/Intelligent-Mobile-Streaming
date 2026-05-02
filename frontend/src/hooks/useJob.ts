import { useState, useEffect, useRef } from 'react'
import { api } from '../api'

type JobStatus = 'idle' | 'running' | 'done' | 'error'

export function useJob(onDone?: () => void) {
  const [status, setStatus] = useState<JobStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clear = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }

  useEffect(() => () => clear(), [])

  const start = (jobId: string) => {
    setStatus('running')
    setProgress(0)
    setError(null)

    intervalRef.current = setInterval(async () => {
      try {
        const job = await api.getJob(jobId)
        setProgress(job.progress)
        if (job.status === 'done') {
          clear()
          setStatus('done')
          onDone?.()
        } else if (job.status === 'error') {
          clear()
          setStatus('error')
          setError(job.error ?? 'Unknown error')
        }
      } catch {
        clear()
        setStatus('error')
        setError('Failed to poll job status')
      }
    }, 1000)
  }

  return { status, progress, error, start, isRunning: status === 'running' }
}
