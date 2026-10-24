#!/usr/bin/env bash
# Pull the latest code from GitHub and restart the app.
# Run from inside the repo:  bash deploy/update.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
git pull
uv sync
sudo systemctl restart nexora
echo "Updated and restarted. Logs: sudo journalctl -u nexora -f"
