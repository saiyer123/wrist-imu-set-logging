export type Segment = {
  id: number
  start_s: number
  end_s: number
  exercise: string
  reps: number
  tempo_s: number | null
  consistency: number
  confidence: number
  cls_confidence: number
  channel: string
  multiplier: number
  trace: number[]
  trace_hz: number
  peaks: number[]
  truth: { exercise: string; reps: number } | null
}

export type MissedSet = {
  start_s: number
  end_s: number
  exercise: string
  reps: number
}

export type Workout = {
  workout_id: string
  split: string
  duration_s: number
  plot_hz: number
  acc_mag: number[]
  gyr_mag: number[]
  segments: Segment[]
  missed_truth: MissedSet[]
  n_true_sets: number
}

/** What the system decided to do with a detected set, given the threshold. */
export type Disposition = 'auto' | 'review'

export type UserAction = 'confirmed' | 'corrected' | null
