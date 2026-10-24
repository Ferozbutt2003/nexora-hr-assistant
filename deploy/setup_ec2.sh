#!/usr/bin/env bash
# One-time setup for an Ubuntu EC2 instance.
# Run from inside the cloned repo:  bash deploy/setup_ec2.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$APP_DIR/.env" ]; then
  echo "Missing $APP_DIR/.env - create it with your API keys first (see .env.example)."
  exit 1
fi

# 2 GB swap: extra headroom so PyTorch never runs the server out of memory.
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# uv manages Python and all packages.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$APP_DIR"
uv sync

# Install and start the background service (restarts on crash and on reboot).
sed "s|__APP_DIR__|$APP_DIR|g; s|__USER__|$USER|g; s|__HOME__|$HOME|g" \
  deploy/nexora.service | sudo tee /etc/systemd/system/nexora.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now nexora
sudo systemctl restart nexora

echo
echo "Done. Open http://<your-instance-public-ip>:8501"
echo "Logs: sudo journalctl -u nexora -f"
