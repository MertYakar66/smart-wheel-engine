# Launch TradingView Desktop on Windows with Chrome DevTools Protocol enabled.
# Lets Claude Code attach via the tradingview MCP (port 9222).
# Counterpart to launch-tradingview-cdp.sh (macOS).

$Port = 9222

$candidatePaths = @(
    "$env:LOCALAPPDATA\TradingView\TradingView.exe",
    "$env:LOCALAPPDATA\Programs\TradingView\TradingView.exe",
    "$env:ProgramFiles\TradingView\TradingView.exe",
    "${env:ProgramFiles(x86)}\TradingView\TradingView.exe"
)

$tvExe = $candidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

# Fall back to Microsoft Store (MSIX) install. The WindowsApps folder is
# ACL-locked for listing, so probe by package name pattern. Get-AppxPackage
# returns InstallLocation even though Get-ChildItem on the parent fails.
if (-not $tvExe) {
    $pkg = Get-AppxPackage -Name "TradingView.Desktop" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pkg -and (Test-Path "$($pkg.InstallLocation)\TradingView.exe")) {
        $tvExe = "$($pkg.InstallLocation)\TradingView.exe"
    }
}

if (-not $tvExe) {
    Write-Host "ERROR: TradingView.exe not found in any standard location." -ForegroundColor Red
    Write-Host "Install TradingView Desktop from https://www.tradingview.com/desktop/ then re-run."
    Write-Host "Checked:"
    $candidatePaths | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

Write-Host "Closing existing TradingView instances (if any)..."
Get-Process -Name "TradingView" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "Launching TradingView Desktop with CDP on port $Port..."
Write-Host "  exe: $tvExe"
Start-Process -FilePath $tvExe -ArgumentList "--remote-debugging-port=$Port"

Write-Host ""
Write-Host "TradingView Desktop is starting with CDP on port $Port."
Write-Host "Verify with: curl http://localhost:$Port/json"
Write-Host ""
Write-Host "Next: restart Claude Code so the tradingview MCP tools load,"
Write-Host "then run tv_health_check to confirm CDP attach."
