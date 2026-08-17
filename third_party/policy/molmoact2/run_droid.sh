export HF_HUB_ENABLE_HF_TRANSFER=1
uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port 8101 --dtype bfloat16
