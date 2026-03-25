import type { CodeExecutionRequest, CodeExecutionResult } from "@/types";

// ─── Code Execution Service ──────────────────────────────────────────
// Abstraction layer for executing Python analytics in isolated sandboxes.
// Supports Daytona for production and a local simulation for development.

const DAYTONA_API_KEY = process.env.DAYTONA_API_KEY || "";
const DAYTONA_API_URL = process.env.DAYTONA_API_URL || "https://api.daytona.io/v1";

interface CodeExecutor {
  name: string;
  execute(request: CodeExecutionRequest): Promise<CodeExecutionResult>;
  isAvailable(): Promise<boolean>;
}

// ─── Daytona Executor ────────────────────────────────────────────────

class DaytonaExecutor implements CodeExecutor {
  name = "daytona";

  async isAvailable(): Promise<boolean> {
    if (!DAYTONA_API_KEY) return false;
    try {
      const res = await fetch(`${DAYTONA_API_URL}/health`, {
        headers: { Authorization: `Bearer ${DAYTONA_API_KEY}` },
        signal: AbortSignal.timeout(3000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  async execute(request: CodeExecutionRequest): Promise<CodeExecutionResult> {
    const start = Date.now();
    try {
      const res = await fetch(`${DAYTONA_API_URL}/execute`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${DAYTONA_API_KEY}`,
        },
        body: JSON.stringify({
          code: request.code,
          language: request.language,
          timeout: request.timeout || 30000,
        }),
        signal: AbortSignal.timeout(request.timeout || 30000),
      });

      if (!res.ok) {
        return {
          success: false,
          output: "",
          error: `Daytona execution failed: ${res.status}`,
          executionTime: Date.now() - start,
        };
      }

      const data = await res.json();
      return {
        success: data.success,
        output: data.stdout || "",
        error: data.stderr || undefined,
        artifacts: data.artifacts || [],
        executionTime: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        output: "",
        error: `Daytona error: ${err instanceof Error ? err.message : "unknown"}`,
        executionTime: Date.now() - start,
      };
    }
  }
}

// ─── Template-Based Local Executor (no real code execution) ──────────
// Provides pre-built analytics templates for common financial operations
// without requiring an actual sandbox. Used when Daytona is unavailable.

class TemplateExecutor implements CodeExecutor {
  name = "template";

  async isAvailable(): Promise<boolean> {
    return true;
  }

  async execute(request: CodeExecutionRequest): Promise<CodeExecutionResult> {
    const start = Date.now();

    // Parse the template intent from code comments
    const templateMatch = request.code.match(/# template: (\w+)/);
    if (!templateMatch) {
      return {
        success: false,
        output: "",
        error: "Code execution requires Daytona. Set DAYTONA_API_KEY in .env, or use a template (# template: event_study).",
        executionTime: Date.now() - start,
      };
    }

    const template = templateMatch[1];
    const result = this.runTemplate(template, request.code);
    return { ...result, executionTime: Date.now() - start };
  }

  private runTemplate(template: string, code: string): Omit<CodeExecutionResult, "executionTime"> {
    switch (template) {
      case "event_study":
        return {
          success: true,
          output: "Event study template: Calculate abnormal returns around an event date.\nRequires: ticker, event_date, window_days\nOutput: Pre-event, event-day, and post-event returns with significance.",
          artifacts: [{
            type: "table",
            name: "event_study_results",
            data: JSON.stringify({
              headers: ["Period", "Return %", "Abnormal Return %", "Significant"],
              rows: [
                ["Pre-event (-5 to -1)", "-0.3%", "-0.5%", "No"],
                ["Event day (0)", "+2.1%", "+1.8%", "Yes"],
                ["Post-event (+1 to +5)", "+0.8%", "+0.4%", "No"],
              ],
            }),
          }],
        };

      case "peer_comparison":
        return {
          success: true,
          output: "Peer comparison template: Compare a ticker against its sector peers.\nRequires: ticker, peer_tickers, metrics\nOutput: Comparative metrics table.",
          artifacts: [{
            type: "table",
            name: "peer_comparison",
            data: JSON.stringify({
              headers: ["Ticker", "P/E", "Revenue Growth", "Margin", "1M Return"],
              rows: [
                ["TARGET", "25.3x", "+12%", "18.5%", "+3.2%"],
                ["PEER_1", "22.1x", "+8%", "15.2%", "+1.8%"],
                ["PEER_2", "28.7x", "+15%", "21.0%", "+4.5%"],
              ],
            }),
          }],
        };

      case "factor_sensitivity":
        return {
          success: true,
          output: "Factor sensitivity template: Regression-based factor exposure.\nRequires: ticker, factors (rates, oil, fx)\nOutput: Beta coefficients and R-squared.",
          artifacts: [{
            type: "table",
            name: "factor_sensitivity",
            data: JSON.stringify({
              headers: ["Factor", "Beta", "t-Stat", "p-Value"],
              rows: [
                ["Market (SPY)", "1.15", "8.3", "<0.001"],
                ["10Y Rate", "-0.42", "-3.1", "0.002"],
                ["Oil (WTI)", "0.08", "0.6", "0.55"],
              ],
            }),
          }],
        };

      default:
        return {
          success: false,
          output: "",
          error: `Unknown template: ${template}. Available: event_study, peer_comparison, factor_sensitivity`,
        };
    }
  }
}

// ─── Execution Orchestrator ───────────────────────────────────────────

class CodeExecutionService {
  private executors: CodeExecutor[] = [
    new DaytonaExecutor(),
    new TemplateExecutor(),
  ];

  async execute(request: CodeExecutionRequest): Promise<CodeExecutionResult> {
    for (const executor of this.executors) {
      if (await executor.isAvailable()) {
        return executor.execute(request);
      }
    }

    return {
      success: false,
      output: "",
      error: "No code execution backend available",
      executionTime: 0,
    };
  }

  async getAvailableExecutor(): Promise<string | null> {
    for (const executor of this.executors) {
      if (await executor.isAvailable()) return executor.name;
    }
    return null;
  }
}

export const codeExecutionService = new CodeExecutionService();
