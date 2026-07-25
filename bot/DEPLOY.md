# Deploying PolyglotBot

Target: Oracle Cloud Always Free (an always-free Ampere/AMD VM is enough —
the bot is a single async process). Any Linux box works the same way from
step 2 onward.

## 0. Prerequisites
- A Matrix account for the bot (create one on matrix.org or your homeserver;
  fetch an access token via any client's help section or a login API call).
- The winning model's API key (e.g. GROQ_API_KEY).
- A generated `bot/config.yaml`:
  `uv run python scripts/build_report.py --emit-bot-config`
  (or copy `bot/config.yaml.example` and edit).

## 1. Oracle Cloud Always Free VM
1. Create an Always Free VM (Ubuntu 24.04 image is fine).      # VERIFY shapes
2. Open ingress on port 9100 only if you want Prometheus to scrape /metrics
   from outside; otherwise leave the default deny.
3. SSH in.

## 2. Docker path (recommended)
```bash
git clone https://github.com/Blanq-one/langbench && cd langbench
cp .env.example .env   # fill MATRIX_* and the model API key
docker compose -f bot/docker-compose.yml up -d
docker compose -f bot/docker-compose.yml logs -f   # JSON logs
```

## 3. systemd path (no Docker)
```bash
sudo git clone https://github.com/Blanq-one/langbench /opt/langbench
cd /opt/langbench
uv sync --extra bot
cp .env.example .env   # fill in
sudo useradd -r polyglot && sudo chown -R polyglot: /opt/langbench
sudo cp bot/polyglotbot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now polyglotbot
journalctl -u polyglotbot -f
```

## 4. Use it
1. Invite the bot's Matrix user into an UNENCRYPTED room (v1 does not support
   E2E encryption — the invite must be to a room with encryption off).
2. `!help` for commands. `!lang de` to set the room language.
3. `!feedback on` to opt the room into writing feedback (default is off).

## 5. Operations
- Metrics: `curl localhost:9100/metrics` — requests seen, feedback served,
  provider latency histogram, error counts.
- Rate limits: the bot enforces the per-model RPM/RPD from bot/config.yaml
  and replies "I'm rate-limited right now" instead of erroring when the daily
  quota runs out.
- Privacy: the bot writes nothing to disk — no message content, no cache.
  Room state and the !level window live in memory and vanish on restart.
- Logs are JSON lines on stderr.

## Known v1 limits (by design)
Text only; one language per room; no E2E rooms; no per-user settings; the
!level window is in-memory only and lost on restart (privacy feature, not a
bug).
