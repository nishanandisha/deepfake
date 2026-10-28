"use client";

import { useReducedMotion } from "framer-motion";

/**
 * Shared entrance motion.
 *
 * Two problems this solves, both from each component declaring its own
 * `initial/animate/transition` inline:
 *
 * 1. **The stagger outran the content.** The results page delayed its tenth
 *    panel by 0.45s on top of a 0.45s duration, so the page was still
 *    assembling itself almost a second after the data had arrived. Entrance
 *    motion should introduce content, not gate it. `STEP` is small and
 *    `MAX_DELAY` caps the chain however many panels get added later.
 *
 * 2. **`prefers-reduced-motion` was ignored.** The CSS media query in
 *    globals.css cannot help here: framer-motion sets `opacity: 0` as inline
 *    style and animates it in JS, so a reduced-motion user would have seen
 *    invisible panels rather than merely still ones.
 *
 * Note on how the reduced-motion case is expressed. `initial` is what
 * framer serialises into inline style during SSR, and `useReducedMotion()`
 * necessarily reports `false` on the server and the real preference on the
 * client. Branching `initial` on it therefore renders different HTML on each
 * side, React fails hydration and drops the whole tree -- which blanked the
 * page when this helper first landed. So `initial` and `animate` stay
 * constant and only `transition` varies: identical markup either way, and a
 * zero-duration transition means the content simply appears instead of
 * travelling.
 */
const STEP = 0.035;
const MAX_DELAY = 0.22;
const DURATION = 0.34;

// A gentle decelerating curve: quick to leave, slow to settle.
const EASE = [0.16, 1, 0.3, 1] as const;

export type EntranceProps = {
  initial: { opacity: number; y: number };
  animate: { opacity: number; y: number };
  transition: { duration: number; ease: typeof EASE; delay: number };
};

/**
 * Returns a function mapping a panel's index to entrance props.
 * Pass the index, not a hand-tuned delay, so the cap stays enforced.
 */
export function useEntrance(): (index?: number) => EntranceProps {
  const reduce = useReducedMotion();

  return (index = 0) => ({
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0 },
    transition: {
      duration: reduce ? 0 : DURATION,
      ease: EASE,
      delay: reduce ? 0 : Math.min(index * STEP, MAX_DELAY),
    },
  });
}
