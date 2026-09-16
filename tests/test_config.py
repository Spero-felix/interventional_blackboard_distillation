from pathlib import Path


def test_teacher_config_loads_as_explicit_compact_teacher_protocol():
    from ibd.config import AppConfig

    config = AppConfig.from_yaml(
        Path(__file__).parents[1] / "configs" / "deepseek_teacher.yaml"
    )

    assert (
        config.protocol_version
        == "qwen25-socialsim-seven-state-context-v1"
    )
    assert not hasattr(config, "candidate_seeds")
    assert config.schema_retries == 1
    assert config.default_model.model


def test_deepseek_teacher_uses_one_candidate_role():
    from ibd.config import AppConfig

    config = AppConfig.from_yaml(
        Path(__file__).parents[1] / "configs" / "deepseek_teacher.yaml"
    )

    candidate_roles = {role for role in config.roles if role.startswith("candidate")}
    assert candidate_roles == {"candidate"}
