export HF_HUB_ENABLE_HF_TRANSFER=1
uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202 --dtype bfloat16
