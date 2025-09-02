// src/agents/schemas/modelInputs.schema.ts
import { z } from "zod";

export const YearMonth = z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/, "YYYY-MM");

export const XLearnerSchema = z.object({
  treatment_variables: z.array(z.string()).default([]),
  outcome_variables: z.array(z.string()).default([]),
  confounders: z.array(z.string()).default([]),
  time_periods: z
    .object({ start: YearMonth, end: YearMonth })
    .partial()
    .optional(),
});

export const RegimeClassifierSchema = z.object({
  market_indicators: z.array(z.string()).default([]),
  lookback_periods: z.number().int().positive().default(12),
  regime_count: z.number().int().positive().default(3),
});

export const UncertaintyEstimatorSchema = z.object({
  base_estimates: z.array(z.string()).default([]),
  confidence_level: z.number().gt(0).lte(1).default(0.95),
  bootstrap_samples: z.number().int().positive().max(20000).default(1000),
});

export const ModelInputsSchema = z.object({
  xLearner: XLearnerSchema.optional(),
  regimeClassifier: RegimeClassifierSchema.optional(),
  uncertaintyEstimator: UncertaintyEstimatorSchema.optional(),
});

export type ZodModelInputs = z.infer<typeof ModelInputsSchema>;
