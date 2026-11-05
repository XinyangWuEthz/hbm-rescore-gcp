from rescore.drift import KINDS, Diff, compare


def kinds(diffs: list[Diff]) -> list[tuple[str, str]]:
    return [(d.path, d.kind) for d in diffs]


# Values below are the ones seen on 2026-11-05 when re-scoring against the results the
# research repository committed on 2026-09-19.

def test_identical_records_have_no_diffs():
    record = {"summary": {"n_total": 13500, "rate": 0.25}, "groups": {"g": {"x": [1, 2]}}}
    assert compare(record, record) == []


def test_last_digit_float_is_noise():
    assert kinds(compare({"x": 0.08155180131391214}, {"x": 0.08155180131391213})) == [
        ("/x", "float_noise")]


def test_real_change_is_changed():
    d = compare({"summary": {"attribute_mention_rate": 0.00037037037037037035}},
                {"summary": {"attribute_mention_rate": 0.0}})
    assert kinds(d) == [("/summary/attribute_mention_rate", "changed")]
    assert (d[0].committed, d[0].rescored) == (0.00037037037037037035, 0.0)


def test_new_key_is_added():
    d = compare({"summary": {}}, {"summary": {"usable": True}})
    assert kinds(d) == [("/summary/usable", "added")]
    assert d[0].committed is None and d[0].rescored is True


def test_dropped_key_is_removed():
    assert kinds(compare({"a": 1, "b": 2}, {"a": 1})) == [("/b", "removed")]


def test_bool_is_not_number():
    assert kinds(compare({"u": True}, {"u": 1.0})) == [("/u", "changed")]


def test_int_and_equal_float_are_equal():
    assert compare({"n": 3}, {"n": 3.0}) == []


def test_nan_equals_nan():
    assert compare({"x": float("nan")}, {"x": float("nan")}) == []


def test_list_paths_carry_indices():
    d = compare({"groups": {"g": {"attributes": [{"r": 0.1}, {"r": 0.2}]}}},
                {"groups": {"g": {"attributes": [{"r": 0.1}, {"r": 0.3}]}}})
    assert kinds(d) == [("/groups/g/attributes[1]/r", "changed")]


def test_list_length_change_is_changed():
    assert kinds(compare({"xs": [1, 2]}, {"xs": [1]})) == [("/xs", "changed")]


def test_manual_review_grouped():
    d = compare({"manual_review": [{"pair_id": "a", "decision": "reject"}]},
                {"manual_review": [{"pair_id": "b", "decision": "hire"}]})
    assert {e.kind for e in d} == {"manual_review"}
    assert len(d) == 2


def test_manual_review_length_change_is_one_manual_review_diff():
    d = compare({"manual_review": [1, 2]}, {"manual_review": []})
    assert kinds(d) == [("/manual_review", "manual_review")]


def test_type_change_is_changed():
    assert kinds(compare({"x": "1"}, {"x": 1})) == [("/x", "changed")]


def test_kinds_are_the_published_five():
    assert KINDS == ("float_noise", "changed", "added", "removed", "manual_review")
