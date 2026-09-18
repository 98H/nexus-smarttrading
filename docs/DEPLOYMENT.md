# Deployment & Operations Guide: SmartTrading

## 🚀 Live Access URLs
- **Public Preview URL:** [https://river-alternatives-isolated-parker.trycloudflare.com/preview/prod-smarttrading-b2c2d0/](https://river-alternatives-isolated-parker.trycloudflare.com/preview/prod-smarttrading-b2c2d0/)
- **Local Gateway Path:** [/preview/prod-smarttrading-b2c2d0/](/preview/prod-smarttrading-b2c2d0/)
- **Internal Port:** `8100`
- **Process PID:** `31477`
- **Runtime Engine:** `python_preview`
- **Health Status:** `HEALTHY (HTTP 200)`
- **Deployed Timestamp:** `2026-09-18T01:26:38.930457+00:00`

## 📋 Execution Command
```bash
/usr/local/lib/hermes-agent/venv/bin/python3 app.py --port 8100
```

## 🩺 Health Check Verification
```bash
curl -I http://127.0.0.1:8100/
```

## 📜 Live Deployment Logs
Logs are stored at `/root/nexus-agent-graph/workspaces/prod-smarttrading-b2c2d0/logs/deploy.log`.
