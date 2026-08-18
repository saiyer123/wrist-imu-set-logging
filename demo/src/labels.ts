/** Display names and a stable colour per exercise. */
export const PRETTY: Record<string, string> = {
  squats: 'Squats',
  lunges: 'Lunges',
  bicep_curls: 'Bicep curls',
  situps: 'Sit-ups',
  pushups: 'Push-ups',
  tricep_extensions: 'Tricep extensions',
  dumbbell_rows: 'Dumbbell rows',
  jumping_jacks: 'Jumping jacks',
  dumbbell_shoulder_press: 'Shoulder press',
  lateral_shoulder_raises: 'Lateral raises',
}

const HUES: Record<string, number> = {
  squats: 12, lunges: 32, bicep_curls: 52, situps: 92, pushups: 152,
  tricep_extensions: 178, dumbbell_rows: 200, jumping_jacks: 224,
  dumbbell_shoulder_press: 268, lateral_shoulder_raises: 316,
}

export const colourFor = (ex: string, l = 62) => `hsl(${HUES[ex] ?? 0} 70% ${l}%)`
export const pretty = (ex: string) => PRETTY[ex] ?? ex
