from types import SimpleNamespace

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

import scripts.run_open_loop_inference_lerobot as open_loop


def test_open_loop_parser_uses_inference_cuda_graph_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_open_loop_inference_lerobot.py",
            "--dataset",
            "dummy/repo",
            "--episode_idx",
            "0",
            "--checkpoint",
            "allenai/MolmoAct2",
            "--output_dir",
            "/tmp/open-loop",
            "--disable_inference_cuda_graph",
        ],
    )

    args = open_loop.parse_args()

    assert args.enable_inference_cuda_graph is False
    assert ("enable" + "_cuda_graph") not in vars(args)


def test_open_loop_molmoact2_config_uses_clean_cuda_graph_field(monkeypatch):
    captured = {}

    def fake_make_policy(cfg, ds_meta=None):
        captured["cfg"] = cfg
        captured["ds_meta"] = ds_meta
        return object()

    monkeypatch.setattr(open_loop, "make_policy", fake_make_policy)
    monkeypatch.setattr(open_loop, "make_pre_post_processors", lambda cfg, **_: (object(), object()))

    dataset = SimpleNamespace(meta=SimpleNamespace())
    args = SimpleNamespace(
        policy_type=None,
        hf_ckpt=True,
        inference_action_mode="continuous",
        discrete_action_tokenizer=None,
        norm_tag="libero",
        seq_len=None,
        num_steps=None,
        enable_depth_reasoning=False,
        verbose=False,
        enable_inference_cuda_graph=False,
    )

    cfg, policy, preprocessor, postprocessor = open_loop._load_policy_and_processors(
        "allenai/MolmoAct2",
        dataset,
        device="cpu",
        args=args,
    )

    assert cfg is captured["cfg"]
    assert isinstance(cfg, MolmoAct2Config)
    assert cfg.enable_inference_cuda_graph is False
    assert not hasattr(cfg, "enable" + "_cuda_graph")
    assert captured["ds_meta"] is dataset.meta
    assert policy is not None
    assert preprocessor is not None
    assert postprocessor is not None
