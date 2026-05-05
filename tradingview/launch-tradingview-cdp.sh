#!/bin/bash
# Launch TradingView Desktop with Chrome DevTools Protocol enabled
# This allows Claude Code to connect and control your charts

PORT=9222

# Kill any existing TradingView instance
echo "Closing existing TradingView instance (if any)..."
pkill -x "TradingView" 2>/dev/null
sleep 2

# Launch with CDP
echo "Launching TradingView Desktop with CDP on port $PORT..."
open -a "TradingView" --args --remote-debugging-port=$PORT

echo ""
echo "TradingView Desktop is starting with CDP on port $PORT"
echo "You can verify the connection at: http://localhost:$PORT/json"
echo ""
echo "Next step: Open Claude Code and paste the one-shot setup prompt"
echo "from the tradingview-mcp-jackson repo."
