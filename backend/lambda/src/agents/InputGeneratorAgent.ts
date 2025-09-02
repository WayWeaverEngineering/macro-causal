// src/agents/InputGeneratorAgent.ts
import { ChatOpenAI } from "@langchain/openai";
import {
  ZodModelInputs,
  YearMonth,
} from "../schemas/modelInputs.schema";
import { Ontology, emptyOntology } from "../ontology/types";
import { AgentState, ExecutionStep, ModelInputs } from "../models/AgentModels";

type AnalysisType =
  | "treatment_effect"
  | "regime_classification"
  | "uncertainty_estimation"
  | "other";

type PromptAnalysis = {
  analysisType: AnalysisType;
  entities?: string[];
};

type CleanResult = { inputs: ModelInputs; unknowns: string[]; repairs: string[] };

export class InputGeneratorAgent {
  private model: ChatOpenAI;
  private ontology: Ontology;

  constructor(model: ChatOpenAI, ontology?: Ontology) {
    this.model = model;
    this.ontology = ontology ?? emptyOntology;
  }

  async invoke(state: AgentState): Promise<Partial<AgentState>> {
    const step: ExecutionStep = {
      stepId: "input-generation",
      stepName: "Generating Model Inputs",
      description: "Converting natural language query to structured inputs for ML models",
      status: "in_progress",
      startTime: new Date(),
    };

    try {
      if (!state.promptAnalysis) throw new Error("No prompt analysis available");

      const { inputs, unknowns, repairs } = await this.generateModelInputs(
        state.userQuery,
        state.promptAnalysis as PromptAnalysis
      );

      step.status = "completed";
      step.endTime = new Date();
      step.metadata = {
        analysisType: (state.promptAnalysis as PromptAnalysis).analysisType,
        inputsGenerated: Object.keys(inputs).length,
        unknownsCount: unknowns.length,
        repairCount: repairs.length,
      };

      return {
        generatedInputs: inputs,
        currentStep: step,
        executionSteps: this.upsertExecutionStep(state.executionSteps, step),
      };
    } catch (error) {
      step.status = "failed";
      step.endTime = new Date();
      step.error = error instanceof Error ? error.message : "Unknown error";
      return {
        currentStep: step,
        executionSteps: this.upsertExecutionStep(state.executionSteps, step),
        error: step.error,
      };
    }
  }

  private async generateModelInputs(
    userQuery: string,
    promptAnalysis: PromptAnalysis
  ): Promise<CleanResult> {
    const allowed = {
      treatments: this.ontology.treatments.map((o) => o.id),
      outcomes: this.ontology.outcomes.map((o) => o.id),
      confounders: this.ontology.confounders.map((o) => o.id),
      indicators: this.ontology.indicators.map((o) => o.id),
      baseEstimates: this.ontology.baseEstimates.map((o) => o.id),
    };

    const systemMsg =
      "You translate macro-finance natural language queries into STRICT JSON model inputs. " +
      "Choose ONLY from allowed ids provided. If something is not available, add it to `unknowns` (array) and do NOT invent ids. " +
      "Use monthly year-month dates (YYYY-MM). Ensure end >= start. Never include text outside JSON.";

    const userMsg = `
USER QUERY:
${userQuery}

ANALYSIS TYPE: ${promptAnalysis.analysisType}

ALLOWED IDS:
- treatments: ${allowed.treatments.join(", ") || "(none)"}
- outcomes: ${allowed.outcomes.join(", ") || "(none)"}
- confounders: ${allowed.confounders.join(", ") || "(none)"}
- indicators: ${allowed.indicators.join(", ") || "(none)"}
- baseEstimates: ${allowed.baseEstimates.join(", ") || "(none)"}

RULES:
1) Propose inputs relevant to the analysis type:
   - treatment_effect -> xLearner
   - regime_classification -> regimeClassifier
   - uncertainty_estimation -> uncertaintyEstimator
   - other -> whichever are relevant; omit irrelevant.
2) Use only ALLOWED IDS. Unknown items go to "unknowns".
3) Prefer recent windows if unspecified: last 36 months (YYYY-MM).
4) For uncertainty, default confidence_level=0.95, bootstrap_samples ≤ 20000.
5) Omit empty sections rather than inventing.

Return strictly valid JSON with optional keys:
{
  "xLearner"?: {...},
  "regimeClassifier"?: {...},
  "uncertaintyEstimator"?: {...},
  "unknowns"?: string[]
}
`.trim();

    // Use the model directly for JSON parsing
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const response = await this.model.invoke([
          { role: "system", content: systemMsg },
          { role: "user", content: userMsg },
        ]);
        
        const content = response.content as string;
        
        // Extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error("No JSON found in response");
        }
        
        const raw = JSON.parse(jsonMatch[0]) as ZodModelInputs & { unknowns?: string[] };

        const cleaned = this.validateAndCleanInputs(
          raw,
          promptAnalysis.analysisType
        );
        return cleaned;
      } catch (e) {
        if (attempt === 3) break;
        await new Promise((r) => setTimeout(r, 200 * attempt));
      }
    }

    return this.generateFallbackInputs(promptAnalysis.analysisType);
  }

  private validateAndCleanInputs(
    inputs: ZodModelInputs & { unknowns?: string[] },
    analysisType: AnalysisType
  ): CleanResult {
    const repairs: string[] = [];
    const unknowns = inputs.unknowns ?? [];
    const cleaned: ModelInputs = {};

    const ymOk = (ym?: string) => (ym ? YearMonth.safeParse(ym).success : false);
    const now = new Date();
    const thisYM = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
    const past36YM = `${now.getFullYear() - 3}-${String(now.getMonth() + 1).padStart(2, "0")}`;

    const ensurePeriod = (tp?: { start?: string; end?: string }) => {
      let start = tp?.start && ymOk(tp.start) ? tp.start : past36YM;
      let end = tp?.end && ymOk(tp.end) ? tp.end : thisYM;
      if (!ymOk(tp?.start)) repairs.push(`Coerced invalid/missing start to ${start}`);
      if (!ymOk(tp?.end)) repairs.push(`Coerced invalid/missing end to ${end}`);
      if (start > end) {
        [start, end] = [end, start];
        repairs.push("Swapped start/end to maintain start<=end");
      }
      return { start, end };
    };

    if (inputs.xLearner && (analysisType === "treatment_effect" || analysisType === "other")) {
      cleaned.xLearner = {
        treatment_variables: inputs.xLearner.treatment_variables ?? [],
        outcome_variables: inputs.xLearner.outcome_variables ?? [],
        confounders: inputs.xLearner.confounders ?? [],
        time_periods: ensurePeriod(inputs.xLearner.time_periods ?? {}),
      };
    }

    if (inputs.regimeClassifier && (analysisType === "regime_classification" || analysisType === "other")) {
      const lp = inputs.regimeClassifier.lookback_periods ?? 12;
      const rc = inputs.regimeClassifier.regime_count ?? 3;
      cleaned.regimeClassifier = {
        market_indicators: inputs.regimeClassifier.market_indicators ?? [],
        lookback_periods: lp > 0 ? lp : (repairs.push("lookback_periods<=0; set 12"), 12),
        regime_count: rc > 0 ? rc : (repairs.push("regime_count<=0; set 12"), 3),
      };
    }

    if (inputs.uncertaintyEstimator && (analysisType === "uncertainty_estimation" || analysisType === "other")) {
      let cl = inputs.uncertaintyEstimator.confidence_level ?? 0.95;
      let bs = inputs.uncertaintyEstimator.bootstrap_samples ?? 1000;
      if (!(cl > 0 && cl <= 1)) { cl = 0.95; repairs.push("confidence_level out of bounds; set 0.95"); }
      if (!(bs > 0 && bs <= 20000)) { bs = 1000; repairs.push("bootstrap_samples out of bounds; set 1000"); }
      cleaned.uncertaintyEstimator = {
        base_estimates: inputs.uncertaintyEstimator.base_estimates ?? [],
        confidence_level: cl,
        bootstrap_samples: bs,
      };
    }

    // Filter out irrelevant sections by analysisType
    if (analysisType === "regime_classification") delete (cleaned as any).xLearner, delete (cleaned as any).uncertaintyEstimator;
    if (analysisType === "treatment_effect") delete (cleaned as any).regimeClassifier, delete (cleaned as any).uncertaintyEstimator;
    if (analysisType === "uncertainty_estimation") delete (cleaned as any).xLearner, delete (cleaned as any).regimeClassifier;

    return { inputs: cleaned, unknowns, repairs };
  }

  private generateFallbackInputs(analysisType: AnalysisType): CleanResult {
    const inputs: ModelInputs = {};
    const now = new Date();
    const end = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
    const start = `${now.getFullYear() - 3}-${String(now.getMonth() + 1).padStart(2, "0")}`;

    if (analysisType === "treatment_effect" || analysisType === "other") {
      inputs.xLearner = {
        treatment_variables: ["policy_intervention"],
        outcome_variables: ["gdp_growth", "inflation_rate"],
        confounders: ["interest_rate", "exchange_rate"],
        time_periods: { start, end },
      };
    }
    if (analysisType === "regime_classification" || analysisType === "other") {
      inputs.regimeClassifier = {
        market_indicators: ["volatility_index", "trend_indicator"],
        lookback_periods: 12,
        regime_count: 3,
      };
    }
    if (analysisType === "uncertainty_estimation" || analysisType === "other") {
      inputs.uncertaintyEstimator = {
        base_estimates: ["treatment_effect_estimate"],
        confidence_level: 0.95,
        bootstrap_samples: 1000,
      };
    }
    return { inputs, unknowns: [], repairs: ["fallback_used"] };
  }

  private upsertExecutionStep(currentSteps: ExecutionStep[], updatedStep: ExecutionStep): ExecutionStep[] {
    const idx = currentSteps.findIndex((s) => s.stepId === updatedStep.stepId);
    if (idx === -1) return [...currentSteps, updatedStep];
    const next = currentSteps.slice();
    next[idx] = updatedStep;
    return next;
  }
}
