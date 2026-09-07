from ibd.schemas import STATE_ANCHOR_FIELDS, StateBlackboard
from ibd.state_guides import (
    STATE_FIELD_GUIDES,
    render_state_label_guide,
)


def test_state_guides_cover_schema_fields_and_enum_values():
    assert tuple(STATE_FIELD_GUIDES) == STATE_ANCHOR_FIELDS
    properties = StateBlackboard.model_json_schema()["properties"]
    for field, guide in STATE_FIELD_GUIDES.items():
        assert tuple(properties[field]["enum"]) == tuple(guide.values)


def test_state_label_guide_contains_high_risk_boundaries():
    prompt = render_state_label_guide()
    assert "Absence of an advice request does not imply closed" in prompt
    assert "Wanting to act but feeling unable is not ambivalent" in prompt
    assert "terminal thanks without an unfinished request" in prompt
    assert "Permitted local response effects" in prompt
    assert "Do not use this field to change" in prompt
