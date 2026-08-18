import { useEffect, useRef, useState, type RefObject } from 'react'

/**
 * Canvas sizing that survives layout.
 *
 * Drawing straight from a useEffect is unreliable: the first pass can run while
 * the element still measures 0 wide, which silently produces a blank canvas that
 * never repaints because no React dependency changed. Observing the element's
 * real box and re-rendering on change fixes both that and window resizing.
 */
export function useCanvasSize<T extends HTMLElement>(): [RefObject<T>, { w: number; h: number }] {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize((prev) =>
        Math.abs(prev.w - width) < 0.5 && Math.abs(prev.h - height) < 0.5
          ? prev
          : { w: width, h: height },
      )
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  return [ref, size]
}

/** Prepare a device-pixel-ratio-correct 2D context, or null if not yet sized. */
export function prepare(cv: HTMLCanvasElement | null, w: number, h: number) {
  if (!cv || w <= 0 || h <= 0) return null
  const dpr = window.devicePixelRatio || 1
  cv.width = Math.round(w * dpr)
  cv.height = Math.round(h * dpr)
  const g = cv.getContext('2d')
  if (!g) return null
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, w, h)
  return g
}
