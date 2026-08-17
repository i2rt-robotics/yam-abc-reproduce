from lerobot.processor.pipeline import PolicyProcessorPipeline


def make_molmoact2_pre_post_processors(config=None, dataset_stats=None):
    """
    MolmoAct2 runs its own preprocessing internally; keep LeRobot-side processors no-op.
    """
    del config, dataset_stats
    identity = lambda x: x  # no conversion; MolmoAct2 handles its own data shapes
    return (
        PolicyProcessorPipeline(
            name="molmoact2_pre",
            steps=[],
            to_transition=identity,
            to_output=identity,
        ),
        PolicyProcessorPipeline(
            name="molmoact2_post",
            steps=[],
            to_transition=identity,
            to_output=identity,
        ),
    )
