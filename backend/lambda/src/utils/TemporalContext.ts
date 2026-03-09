/**
 * Temporal context for agent prompts. Provides the current date for temporal grounding
 * so models correctly judge whether years in user queries are in the past or future.
 *
 * Set CURRENT_DATE_OVERRIDE (YYYY-MM-DD) for tests or deterministic runs.
 */

export interface TemporalContext {
  date: string;   // "2026-03-08"
  year: number;   // 2026
  iso: string;    // full ISO string
}

/**
 * Parse override as UTC midnight. Date-only "YYYY-MM-DD" is interpreted as local
 * in some JS runtimes; appending "Z" forces UTC and avoids timezone shift.
 */
function parseOverrideAsUtc(override: string): Date {
  return /^\d{4}-\d{2}-\d{2}$/.test(override.trim())
    ? new Date(`${override.trim()}T00:00:00.000Z`)
    : new Date(override);
}

/**
 * Get the current temporal context. Uses CURRENT_DATE_OVERRIDE env var when set.
 * All fields use UTC for consistency (avoids date/year mismatch across timezones).
 * CURRENT_DATE_OVERRIDE (YYYY-MM-DD) is always interpreted as UTC midnight.
 */
export function getTemporalContext(): TemporalContext {
  const override = process.env.CURRENT_DATE_OVERRIDE;
  const d = override ? parseOverrideAsUtc(override) : new Date();

  const iso = d.toISOString();
  const date = iso.split('T')[0];
  const year = d.getUTCFullYear();

  return { date, year, iso };
}

/**
 * Format temporal context as a prompt block for injection into agent prompts.
 */
export function formatTemporalContextForPrompt(ctx?: TemporalContext): string {
  const { date } = ctx ?? getTemporalContext();
  return `CONTEXT:
- Current date: ${date} (YYYY-MM-DD)
- Use this date when judging whether years in the query are in the past or future.
- Macro-causal analysis uses historical data; years before today's date are valid for analysis.
`;
}
