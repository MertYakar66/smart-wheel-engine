import { NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

/**
 * API bridge to the smart-wheel-engine Python backend.
 *
 * Reads from the Parquet feature store and returns JSON data
 * for the terminal UI (vol edge, regime, options candidates).
 *
 * GET /api/engine?action=vol_edge&ticker=AAPL
 * GET /api/engine?action=regime&ticker=AAPL
 * GET /api/engine?action=candidates&limit=20
 * GET /api/engine?action=status
 */

const ENGINE_ROOT = path.resolve(process.cwd(), "..");

function buildPythonScript(action: string, params: Record<string, string>): string {
  const ticker = params.ticker || "AAPL";
  const limit = params.limit || "20";

  switch (action) {
    case "vol_edge":
      return `
import json, sys
sys.path.insert(0, '${ENGINE_ROOT}')
from data.feature_store import FeatureStore
store = FeatureStore('${ENGINE_ROOT}/data/features')
df = store.read_features('vol_edge', '${ticker}')
if df is not None and not df.empty:
    last = df.iloc[-1]
    result = {
        'ticker': '${ticker}',
        'iv_rv_spread': float(last.get('iv_rv_spread', 0)) if 'iv_rv_spread' in last.index else None,
        'iv_rv_ratio': float(last.get('iv_rv_ratio', 0)) if 'iv_rv_ratio' in last.index else None,
        'edge_score': float(last.get('edge_score', 0)) if 'edge_score' in last.index else None,
        'vrp_percentile': float(last.get('vrp_percentile', 0)) if 'vrp_percentile' in last.index else None,
        'vol_regime': int(last.get('vol_regime', 1)) if 'vol_regime' in last.index else None,
        'iv_rank': float(last.get('iv_rank', 50)) if 'iv_rank' in last.index else None,
        'rv_21d': float(last.get('rv_21d', 0)) if 'rv_21d' in last.index else None,
        'atm_iv': float(last.get('atm_iv', 0)) if 'atm_iv' in last.index else None,
        'date': str(last.get('date', '')) if 'date' in last.index else None,
    }
    print(json.dumps(result))
else:
    print(json.dumps({'error': 'No data', 'ticker': '${ticker}'}))
`;

    case "regime":
      return `
import json, sys
import numpy as np
sys.path.insert(0, '${ENGINE_ROOT}')
from data.feature_store import FeatureStore
store = FeatureStore('${ENGINE_ROOT}/data/features')
df = store.read_features('regime', '${ticker}')
if df is not None and not df.empty:
    last = df.iloc[-1]
    # Only include numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result = {}
    for c in numeric_cols:
        val = last[c]
        result[c] = float(val) if not np.isnan(val) else None
    result['ticker'] = '${ticker}'
    result['date'] = str(last.get('date', '')) if 'date' in last.index else None
    print(json.dumps(result))
else:
    print(json.dumps({'error': 'No data', 'ticker': '${ticker}'}))
`;

    case "candidates":
      return `
import json, sys
sys.path.insert(0, '${ENGINE_ROOT}')
from data.feature_store import FeatureStore
store = FeatureStore('${ENGINE_ROOT}/data/features')
candidates = []
features = store.list_features('vol_edge')
for cat, ticker in features:
    df = store.read_features('vol_edge', ticker)
    if df is not None and not df.empty:
        last = df.iloc[-1]
        edge = float(last.get('edge_score', 0)) if 'edge_score' in last.index and str(last.get('edge_score', 0)) != 'nan' else 0
        candidates.append({
            'ticker': ticker,
            'edge_score': edge,
            'iv_rv_spread': float(last.get('iv_rv_spread', 0)) if 'iv_rv_spread' in last.index and str(last.get('iv_rv_spread', 0)) != 'nan' else 0,
            'iv_rank': float(last.get('iv_rank', 50)) if 'iv_rank' in last.index and str(last.get('iv_rank', 50)) != 'nan' else 50,
            'vol_regime': int(last.get('vol_regime', 1)) if 'vol_regime' in last.index and str(last.get('vol_regime', 1)) != 'nan' else 1,
        })
candidates.sort(key=lambda x: x['edge_score'], reverse=True)
print(json.dumps(candidates[:${limit}]))
`;

    case "status":
      return `
import json, sys
sys.path.insert(0, '${ENGINE_ROOT}')
from data.feature_store import FeatureStore
store = FeatureStore('${ENGINE_ROOT}/data/features')
stats = store.get_storage_stats()
features = store.list_features()
categories = {}
for cat, ticker in features:
    categories[cat] = categories.get(cat, 0) + 1
print(json.dumps({
    'storage': stats,
    'categories': categories,
    'total_features': len(features),
}))
`;

    default:
      return `import json; print(json.dumps({'error': 'Unknown action: ${action}'}))`;
  }
}

function runPython(script: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", ["-c", script], {
      cwd: ENGINE_ROOT,
      env: { ...process.env, PYTHONPATH: ENGINE_ROOT },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(stderr || `Process exited with code ${code}`));
      }
    });

    proc.on("error", (err) => {
      reject(err);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      proc.kill();
      reject(new Error("Process timed out"));
    }, 30000);
  });
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get("action") || "status";
  const params: Record<string, string> = {};

  searchParams.forEach((value, key) => {
    if (key !== "action") params[key] = value;
  });

  try {
    const script = buildPythonScript(action, params);
    const stdout = await runPython(script);

    const data = JSON.parse(stdout.trim());
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("Engine API error:", message);
    return NextResponse.json(
      { error: "Engine unavailable", detail: message },
      { status: 503 }
    );
  }
}
