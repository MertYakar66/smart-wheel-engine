# TradingView Bridge

Visual cockpit for the Smart Wheel Engine. Ships three things:

| File | Purpose |
|---|---|
| `smart_wheel_signals.pine` | Pine v5 indicator that mirrors `engine/tv_signals.py` on the chart |
| `alert_payload_schema.json` | JSON Schema for the webhook body emitted by the indicator |
| `README.md` | This file — setup and operational notes |

The underlying architecture is documented in `TRADINGVIEW_INTEGRATION_REPORT.md`
at the repo root. This README is the hands-on checklist.

---

## 1. Install the indicator in TradingView

1. Open a chart on TradingView.
2. Pine Editor (bottom panel) → **New** → **Indicator**.
3. Paste the contents of `smart_wheel_signals.pine`.
4. Click **Save** then **Add to chart**.

You should immediately see:
- Bollinger bands and a yellow SMA-20 line
- Phase-shaded background (green = post-expansion, red = compression, …)
- On the most recent bar, a `PUT ZONE` / `CALL ZONE` / `STRANGLE ZONE` / `AVOID`
  label if the zone is active

The thresholds at the top of the Pine file **must stay in sync** with the
constants at the top of `engine/tv_signals.py`. The parity test under
`tests/test_tv_signals.py::test_pine_parity_constants` enforces this.

## 2. Point an alert at the engine webhook

TradingView supports outbound HTTP webhooks on paid plans. For a **private
local** deployment the workflow is:

1. Start the engine API locally: `python engine_api.py`
2. Expose it to the internet via any tunnel you trust — e.g.
   `cloudflared tunnel --url http://localhost:8787`, `ngrok http 8787`, or a
   Tailscale funnel. Copy the public URL.
3. In TradingView, right-click the indicator → **Add alert**.
4. **Condition** → Smart Wheel Signals → pick one of the four zones.
5. **Notifications** tab → enable **Webhook URL** and paste
   `https://<tunnel>/api/tv/webhook`.
6. Leave **Message** empty — the Pine script supplies a valid JSON body via
   `alertcondition(message=...)`.
7. Click **Create**.

### Optional shared secret

If you want to reject arbitrary POSTs to your tunneled URL, set
`TV_WEBHOOK_SECRET` in the environment before starting the engine, then edit
the `*Msg` templates in the Pine script to include a `"secret":"<value>"`
field. The engine's `_handle_tv_webhook` rejects payloads with missing or
wrong secrets (`HTTP 401`).

## 3. What happens when an alert fires

```
TradingView alert fires
        │
        ▼
POST /api/tv/webhook  (JSON payload matches alert_payload_schema.json)
        │
        ▼
engine_api.EngineAPIHandler._handle_tv_webhook
        │
        ├─► TVAlert.parse   (validate schema, strip unknown keys)
        │
        ▼
engine_api.EngineAPIHandler._enrich_alert
        │
        ├─► compute_tv_signal(ohlcv)     engine-side parity re-check
        ├─► WheelRunner.analyze_ticker   wheel score, events, IV rank
        │
        ▼
verdict ∈ {proceed, review, skip}
        │
        ├─► appended to in-memory ring buffer (/api/tv/alerts)
        └─► returned in the webhook response body
```

The enriched response contains (among other fields):

```json
{
  "ticker": "MU",
  "signal": "wheel_put_zone",
  "verdict": "proceed",
  "pine_agrees": true,
  "phase": "post_expansion",
  "wheel_score": 72.4,
  "iv_rank": 48.2,
  "preferred_dte": 31,
  "preferred_delta_range": [0.18, 0.22],
  "days_to_earnings": 21
}
```

## 4. Polling-only mode (no webhook, no tunnel)

If you do not want to expose the engine to the public internet you can skip
the webhook path entirely and use `GET /api/tv/signal?ticker=<T>` as a pull
API instead. The same `TVSignal` struct comes back. A shell cron running
every 15 minutes is enough for a daily workflow:

```bash
curl -s http://localhost:8787/api/tv/scan?limit=25 | jq .
```

This returns every ticker in the wheel-qualified universe whose Pine-parity
signal currently fires a zone flag.

## 5. Operational notes

- The alert log is in-memory. If you restart `engine_api.py` you lose the
  backlog; swap `_TV_ALERT_LOG` for a SQLite write if you want persistence.
- TradingView alert counts are plan-limited. Define at most 3-4 concurrent
  alerts per symbol and let the engine enrich them rather than wiring a
  separate alert per threshold.
- Pine cannot read options chains, so the indicator deliberately does not
  mention IV rank or earnings. Those checks live in the engine's enrichment
  step.
- The bridge is **private-use first**. Nothing about it assumes a customer
  base; all state lives on the local machine.

## 6. Known limitations

1. **Percentile lookback is approximate.** TradingView limits historical bars
   per plan. On a Free plan `pctlLookback` above ~250 will silently clamp.
2. **Drawing tools are opaque to Pine.** Hand-drawn support/resistance lines
   cannot feed the script.
3. **Data source drift.** Default US equity data on TradingView comes from
   Cboe; the engine uses Bloomberg historical. Small price disagreements are
   normal — the engine's internal parity test uses its own OHLCV so this
   drift never affects `compute_tv_signal` output.
