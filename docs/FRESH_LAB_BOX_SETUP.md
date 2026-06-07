# Fresh lab-box bring-up — pulling Bloomberg from a transient machine

Use this when you sit down at a **fresh or shared machine that has a Bloomberg
Terminal** and want to pull data into the repo — a university lab box, a
borrowed desktop, any managed workstation where **nothing of ours persists**
and the Claude Code session **starts with no memory**. This is the
Bloomberg-pull counterpart to `docs/LAPTOP_SETUP.md` (which is about *your own*
machine + Theta + regenerating derived data).

The whole point of this file: a fresh Claude terminal that has never seen this
project can be brought to "ready to pull" by following these steps in order —
no recall required. The steps below are exactly what worked on the lab box
(`IITS-I108-09`, user `mertmert`, 2026-06-02 → 06-05). Paths like
`C:\Users\mertmert\...` and `C:\Anaconda3\...` are the *worked example* — the
machine-specific bits are flagged; everything else is invariant.

---

## The golden rule: clone before you orient

A fresh box does **not** have the repo. If you tell the terminal to read
`CLAUDE.md` / a worklog *first*, it finds nothing and (correctly) stops. Order
is always:

1. **Clone**, 2. check out the working branch, 3. **then** read the orientation
docs. Do not skip ahead.

---

## Ordered bring-up

### STEP 0 — Clone, then orient

```powershell
git clone https://github.com/MertYakar66/smart-wheel-engine.git C:\Users\mertmert\smart-wheel-engine
cd C:\Users\mertmert\smart-wheel-engine
git checkout data/bloomberg-refresh-2026-06-02   # the live data-refresh branch
```

Only now read, in order: `CLAUDE.md` (auto-loaded), `AGENTS.md`, this file, and
the latest `docs/worklog/bloomberg-*` entry on the refresh branch for where the
pull campaign stands.

### STEP 1a — Gate on the Bloomberg Desktop API *before* you invest in setup

Lab terminals frequently allow the **GUI** but not the **Desktop API**. Cheap
preliminary check first, so you don't build a whole Python env on a box that
can't pull:

- `bbcomm.exe` / `wintrv.exe` running (Bloomberg Terminal logged in), and
- port **8194** listening (`Test-NetConnection 127.0.0.1 -Port 8194`).

If the Terminal is up and 8194 is open, proceed to env. The *decisive* data
test comes in STEP 1b once xbbg is installed.

### STEP 2 — Python environment (the box has no usable Python)

Managed boxes ship only the Microsoft-Store Python stub, which is useless here.
Bootstrap from Anaconda and build a venv **off the repo tree** so it never
pollutes the working copy:

```powershell
C:\Anaconda3\python.exe -m venv C:\Users\mertmert\bbg-venv      # Anaconda path is machine-specific
C:\Users\mertmert\bbg-venv\Scripts\Activate.ps1

# blpapi is NOT on PyPI — it installs only from Bloomberg's own index:
pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
pip install xbbg==1.2.6 pandas==3.0.3

$env:PYTHONUTF8 = "1"   # narwhals' repr uses box-drawing chars that crash Windows cp1252
```

### STEP 1b — Decisive API gate (run it now that xbbg is in)

```python
from xbbg import blp
print(blp.bdp("SPX Index", "PX_LAST"))   # must return a real number
print(blp.bdp("AAPL US Equity", "PX_LAST"))
```

Real entitled values (we saw **SPX 7610.67 / AAPL 315.12** on 06-02,
**SPX 7383.74 / AAPL 307.34** on 06-05) → the box can pull, continue. A
`#N/A`, a permission error, or a hang → **stop**: this box is GUI-only, no
amount of setup fixes it.

### STEP 3 — Git identity

```powershell
git config user.name  "Mert Yakar"
git config user.email "mertyakar.my@gmail.com"
```

### STEP 4 — GitHub push auth (browser OAuth — never a PAT)

Transcripts from these sessions get exported, so **never paste a personal
access token into chat**. Use Git Credential Manager's browser OAuth. Run it as
**one line, with no leading `!`** (the `!` is a Claude-Code in-session shell
prefix and is invalid in raw PowerShell; a wrapped line also breaks it):

```powershell
$env:GCM_GITHUB_AUTHMODES='browser'; git -C 'C:\Users\mertmert\smart-wheel-engine' push -u origin HEAD:data/bloomberg-refresh-2026-06-02
```

A browser window opens for the OAuth grant; once you authorize, GCM caches the
credential for the session. Confirm with a non-interactive **dry-run push**
before relying on it.

### STEP 5 — (only if a panel is too big for GitHub) rclone → Google Drive

GitHub's hard cap is **100 MB/file**. Deep-history panels that exceed that even
gzipped go to Google Drive, not git:

```powershell
rclone copy <file> gdrive:swe-deep-history/
```

rclone's Drive remote was set up on the desktop; a lab box needs its own
browser OAuth grant (revoke it at teardown — see below).

---

## Pinned versions (verified working)

| Component | Version | Note |
|---|---|---|
| Python | 3.13.9 | from Anaconda (`C:\Anaconda3\python.exe`) |
| blpapi | 3.26.4.2 | Bloomberg index-url only, **not** PyPI |
| xbbg | 1.2.6 | returns long-format `narwhals` frames |
| pandas | 3.0.3 | |
| Claude Code | 2.1.163 | `winget install Anthropic.ClaudeCode` |

---

## Gotchas (the part that saves the next fresh terminal)

- **Clone before orient** — see the golden rule above.
- **API-gate before env-investment** — STEP 1a/1b; don't build Python on a
  GUI-only box.
- **MS-Store Python stub is a dead end** — bootstrap from Anaconda.
- **`blpapi` is not on PyPI** — needs `--index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/`.
- **`PYTHONUTF8=1`** — without it, narwhals' `repr` crashes the Windows console (cp1252).
- **xbbg ≥ 1.2 changed shape** — it returns long-format narwhals
  (`ticker, date, field, value`), not the old wide pandas MultiIndex. Convert
  with `.to_native()` then `pivot_table(index=['date','ticker'], columns='field')`.
  The `INDX_MWEIGHT` member column is now **"Member Ticker and Exchange Code"**
  (title case).
- **One invalid field nulls the whole `bdh` batch** — only request fields you've
  validated; a single bad field returns NaN for the entire call.
- **No leading `!` in raw PowerShell**, and keep the auth command on one
  un-wrapped line (STEP 4).
- **venv off the repo tree** — keep `bbg-venv` outside the working copy.
- **Monoliths stay frozen < 100 MB.** The connector reads the committed
  monoliths on the refresh branch; do not let a pull balloon them. Deep history
  is written as separate `.gz` slices under `data/bloomberg/deep/` and pushed to
  the **buffer branch `deep-history/bloomberg-raw`** (never merged). Anything
  too big for GitHub even gzipped → Google Drive.
- **OHLCV rotation contract (load-bearing).** The committed `sp500_ohlcv.csv`
  stores columns *rotated*; the connector compensates on read. A correct pull
  must use `FIELD_MAP = {"PX_HIGH":"open","PX_LAST":"high","PX_LOW":"low",
  "PX_OPEN":"close","PX_VOLUME":"volume"}`. Gate every pulled chunk: each row
  must satisfy `open == max(o,h,l,c)` **and** `low == min(o,h,l,c)` — zero
  violations means the rotation held. (This caught a real inversion bug in an
  earlier pull; never skip the gate.)

---

## Security teardown — shared / managed box

The clone and venv vanish when the box is reclaimed, but **OAuth grants are
account-level and outlive the machine.** At session end, explicitly:

1. **Revoke the GitHub GCM browser-OAuth grant** and clear the credential
   (`git credential-manager logout` / sign out of GitHub in the box's browser).
2. **Revoke any rclone Google-Drive OAuth grant.**
3. Confirm no PAT or token was ever written to disk (browser OAuth avoids this
   by design).

Leaving a grant active on a shared box is the one durable footprint these
sessions can leave — don't.

---

## Then what — start pulling

Bring-up ends at "ready to pull." The actual backfill (pull → rotation/seam
gate → gzip → push to `deep-history/bloomberg-raw`, with the storage model and
exact commands) lives in the Bloomberg deep-history worklog on the refresh
branch (`docs/worklog/bloomberg-deep-history-2026-06-04.md`). Pull **newest
window first, walking backward** toward the floor (1994 for IV-bearing series —
Bloomberg's implied-vol hard floor; the engine is option/IV-centric, so
pre-1994 price-only history isn't comparable), committing each window as it
lands so a box reclaim can't lose it.

---

## Appendix — paste-ready cold-start prompt

Hand this to a brand-new Claude Code terminal on a fresh Bloomberg box. It
encodes the order above so the terminal needs no prior memory:

```text
You're on a fresh, possibly shared lab box with a Bloomberg Terminal but none of
our project files. Do NOT read project docs before cloning — they aren't here yet.

1. Clone, then check out the data branch:
   git clone https://github.com/MertYakar66/smart-wheel-engine.git C:\Users\<me>\smart-wheel-engine
   cd into it; git checkout data/bloomberg-refresh-2026-06-02
   Now read CLAUDE.md, AGENTS.md, docs/FRESH_LAB_BOX_SETUP.md, and the latest
   docs/worklog/bloomberg-* entry to orient.

2. Gate the Bloomberg Desktop API before building anything: confirm the Terminal
   is logged in and port 8194 is open. (Decisive bdp test comes after xbbg.)

3. Build a venv OFF the repo with Anaconda's python:
   C:\Anaconda3\python.exe -m venv C:\Users\<me>\bbg-venv ; activate it
   pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
   pip install xbbg==1.2.6 pandas==3.0.3 ; set PYTHONUTF8=1

4. Decisive gate: blp.bdp("SPX Index","PX_LAST") must return a real number. If
   not, stop and tell me — the box is GUI-only.

5. Set git identity (Mert Yakar / mertyakar.my@gmail.com). For push, use GCM
   browser OAuth on ONE line, no leading '!':
   $env:GCM_GITHUB_AUTHMODES='browser'; git -C '<repo>' push -u origin HEAD:data/bloomberg-refresh-2026-06-02
   Never paste a token in chat. Verify with a dry-run push, then tell me you're
   ready and wait — I'll confirm before you start the metered pull.

At session end (shared box): revoke the GitHub and any rclone OAuth grants.
```
