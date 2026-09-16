from ibd.context_memory import apply_context_patch
from ibd.schemas import ContextPatch, ReplacePair, UserContext


def context_with(**updates: list[str]) -> UserContext:
    payload = UserContext.empty().model_dump()
    payload.update(updates)
    return UserContext.model_validate(payload)


def patch_with(
    *,
    add: dict[str, list[str]] | None = None,
    replace: dict[str, list[dict[str, str]]] | None = None,
    remove: dict[str, list[str]] | None = None,
) -> ContextPatch:
    payload = ContextPatch.empty().model_dump()
    for section, updates in (
        ("add", add),
        ("replace", replace),
        ("remove", remove),
    ):
        if updates is not None:
            payload[section].update(updates)
    return ContextPatch.model_validate(payload)


def test_reducer_applies_replace_remove_add_and_preserves_order() -> None:
    previous = context_with(
        active_concerns=["担心答辩", "担心失眠"],
        goals_and_priorities=["先把论文交上去"],
    )
    patch = patch_with(
        add={"active_concerns": ["担心导师失望", "担心导师失望"]},
        replace={
            "active_concerns": [
                ReplacePair(old="担心答辩", new="担心明天答辩卡住").model_dump()
            ]
        },
        remove={"active_concerns": ["担心失眠"]},
    )

    result = apply_context_patch(previous, patch)

    assert result.context.active_concerns == [
        "担心明天答辩卡住",
        "担心导师失望",
    ]
    assert result.context.goals_and_priorities == ["先把论文交上去"]
    assert result.errors == ()


def test_reducer_treats_duplicate_add_and_absent_remove_as_no_op() -> None:
    previous = context_with(active_concerns=["担心答辩"])
    patch = patch_with(
        add={"active_concerns": ["担心答辩"]},
        remove={"active_concerns": ["不存在的事实"]},
    )

    result = apply_context_patch(previous, patch)

    assert result.context.active_concerns == ["担心答辩"]
    assert result.errors == ()


def test_reducer_skips_replace_when_old_value_is_absent() -> None:
    previous = context_with(active_concerns=["担心答辩"])
    patch = patch_with(
        replace={
            "active_concerns": [
                {"old": "担心面试", "new": "担心明天面试"}
            ]
        }
    )

    result = apply_context_patch(previous, patch)

    assert result.context == previous
    assert result.errors == (
        "active_concerns: replace.old not found: 担心面试",
    )


def test_reducer_rejects_same_fact_replace_remove_and_add_remove_conflicts() -> None:
    previous = context_with(active_concerns=["担心答辩"])
    patch = patch_with(
        add={"active_concerns": ["不想听建议"]},
        replace={
            "active_concerns": [
                {"old": "担心答辩", "new": "担心明天答辩"}
            ]
        },
        remove={"active_concerns": ["担心答辩", "不想听建议"]},
    )

    result = apply_context_patch(previous, patch)

    assert result.context.active_concerns == ["担心答辩"]
    assert result.errors == (
        "active_concerns: replace/remove conflict: 担心答辩",
        "active_concerns: add/remove conflict: 不想听建议",
    )
