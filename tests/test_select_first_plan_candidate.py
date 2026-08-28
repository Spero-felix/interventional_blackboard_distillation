import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import ScriptedBackend
from ibd.teacher import TeacherRunner


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_first_plan_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_first_plan_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def make_trace(history, app_config, *, example_id="e-1", split="train"):
    return TeacherRunner(ScriptedBackend(), app_config).run(
        example_id,
        history,
        split=split,
    )


def make_full_dataset(history, app_config):
    templates = {
        split: make_trace(
            history,
            app_config,
            example_id=f"template-{split}",
            split=split,
        )
        for split in ("train", "dev", "diagnostic_holdout")
    }
    traces = []
    for split, count in (("train", 1500), ("dev", 200), ("diagnostic_holdout", 300)):
        traces.extend(
            templates[split].model_copy(update={"example_id": f"{split}-{index:04d}"})
            for index in range(count)
        )
    return traces


def write_traces(path, traces):
    path.write_text(
        "".join(
            json.dumps(trace.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for trace in traces
        ),
        encoding="utf-8",
    )


def test_select_s1_rebuilds_only_final_fields(history, app_config):
    trace = make_trace(history, app_config)
    converted = module.select_s1(trace)
    s1 = next(item for item in trace.candidates if item.strategy_id == "S1")

    assert converted.final_selection.selected_candidate_id == s1.candidate_id
    assert converted.final_selection.selected_strategy_id == "S1"
    assert converted.final_selection.response_goal == s1.response_goal
    assert converted.final_selection.response_act == s1.response_act
    assert converted.final_response == s1.response

    before = trace.model_dump(mode="json")
    after = converted.model_dump(mode="json")
    before.pop("final_selection")
    before.pop("final_response")
    after.pop("final_selection")
    after.pop("final_response")
    assert after == before


def test_select_s1_is_idempotent(history, app_config):
    first = module.select_s1(make_trace(history, app_config))
    second = module.select_s1(first)
    assert second.model_dump(mode="json") == first.model_dump(mode="json")


def test_select_s1_uses_strategy_id_when_candidates_are_reordered(
    history,
    app_config,
):
    trace = make_trace(history, app_config)
    reordered = trace.model_copy(update={"candidates": list(reversed(trace.candidates))})
    converted = module.select_s1(reordered)
    assert converted.final_selection.selected_strategy_id == "S1"
    assert converted.final_response == "候选回复-1"


@pytest.mark.parametrize("s1_count", [0, 2])
def test_select_s1_rejects_missing_or_duplicate_s1(history, app_config, s1_count):
    trace = make_trace(history, app_config)
    s1 = next(item for item in trace.candidates if item.strategy_id == "S1")
    non_s1 = [item for item in trace.candidates if item.strategy_id != "S1"]
    candidates = non_s1 if s1_count == 0 else [s1, s1, non_s1[0]]
    malformed = trace.model_construct(**{
        **trace.__dict__,
        "candidates": candidates,
    })

    with pytest.raises(ValueError, match=f"exactly one S1 candidate; found {s1_count}"):
        module.select_s1(malformed)


def test_convert_dataset_preserves_order_and_reports_exact_counts(
    history,
    app_config,
):
    traces = make_full_dataset(history, app_config)
    converted, summary = module.convert_dataset(traces)

    assert [item.example_id for item in converted] == [item.example_id for item in traces]
    assert summary.total == 2000
    assert summary.splits == {
        "train": 1500,
        "dev": 200,
        "diagnostic_holdout": 300,
    }
    assert summary.selected_before == {"S2": 2000}
    assert summary.selected_after == {"S1": 2000}


def test_convert_dataset_rejects_duplicate_ids(history, app_config):
    traces = make_full_dataset(history, app_config)
    traces[-1] = traces[-1].model_copy(update={"example_id": traces[0].example_id})
    with pytest.raises(ValueError, match=f"duplicate example_id {traces[0].example_id}"):
        module.convert_dataset(traces)


def test_convert_dataset_rejects_wrong_total(history, app_config):
    traces = make_full_dataset(history, app_config)[:-1]
    with pytest.raises(ValueError, match="expected 2000 traces; found 1999"):
        module.convert_dataset(traces)


def test_convert_dataset_rejects_wrong_split_counts(history, app_config):
    traces = make_full_dataset(history, app_config)
    traces[-1] = traces[-1].model_copy(update={"split": "train"})
    with pytest.raises(ValueError, match="unexpected split counts"):
        module.convert_dataset(traces)


def test_main_writes_complete_s1_dataset_and_summary(
    tmp_path,
    capsys,
    history,
    app_config,
):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "nested" / "converted.jsonl"
    write_traces(source, make_full_dataset(history, app_config))

    result = module.main(["--input", str(source), "--output", str(target)])

    assert result == 0
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2000
    assert {row["final_selection"]["selected_strategy_id"] for row in rows} == {"S1"}
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "output": str(target),
        "selected_after": {"S1": 2000},
        "selected_before": {"S2": 2000},
        "splits": {"dev": 200, "diagnostic_holdout": 300, "train": 1500},
        "total": 2000,
    }


def test_main_refuses_existing_output(tmp_path, history, app_config):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "converted.jsonl"
    write_traces(source, make_full_dataset(history, app_config))
    target.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        module.main(["--input", str(source), "--output", str(target)])
    assert target.read_text(encoding="utf-8") == "keep me"


def test_main_refuses_same_input_and_output(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="input and output must be different"):
        module.main(["--input", str(source), "--output", str(source)])
    assert source.read_text(encoding="utf-8") == "unchanged"


def test_main_validation_failure_leaves_no_output_or_temporary_file(
    tmp_path,
    history,
    app_config,
):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "nested" / "converted.jsonl"
    write_traces(source, make_full_dataset(history, app_config)[:-1])

    with pytest.raises(ValueError, match="expected 2000 traces"):
        module.main(["--input", str(source), "--output", str(target)])
    assert not target.exists()
    assert not target.parent.exists()
