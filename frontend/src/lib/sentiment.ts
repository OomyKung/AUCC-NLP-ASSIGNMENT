/**
 * When a per-message sentiment is too weak to present as a verdict.
 *
 * Measured on the Wisesight test split with the deployed blend: under 0.5
 * confidence the classifier is right 47.5% of the time, which is barely better
 * than always guessing the majority class. Showing "neutral 48%" in that band
 * tells a reader something untrue, so the interface says "ไม่ชัดเจน" instead.
 *
 * The number is served by the backend (`GET /api/sentiments`) rather than
 * written here, so it cannot drift away from the value the model was measured
 * at. The default matches the backend's and is only used before that first
 * response arrives.
 */
import { api } from '../services/api'

let threshold = 0.5

// Fetched once per page load. A failure leaves the default in place, which is
// the same number the backend currently serves.
void api
  .sentiments()
  .then((rows) => {
    const served = rows.find((row) => typeof row.unclear_below === 'number')
    if (served?.unclear_below != null) threshold = served.unclear_below
  })
  .catch(() => {
    /* keep the default */
  })

/** The confidence below which a prediction is not presented as a verdict. */
export function unclearBelow(): number {
  return threshold
}

/**
 * Whether this prediction is too weak to show as a label.
 *
 * A missing confidence counts as unclear: an unscored message is not a neutral
 * one, and the two should not look the same.
 */
export function isUnclear(confidence: number | null | undefined): boolean {
  return confidence == null || confidence < threshold
}
