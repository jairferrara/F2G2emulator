"""
Tests for common/progress.py.
"""

from common.progress import stage


def test_stage_prints_start_and_done_with_elapsed_time(capsys):
    with stage("Doing work"):
        pass

    captured = capsys.readouterr().out
    lines = captured.splitlines()

    assert lines[0] == "Doing work..."
    assert lines[1].startswith("Doing work: done in")
    assert lines[1].endswith("s")


def test_stage_runs_the_wrapped_block_between_the_two_prints(capsys):
    events = []

    with stage("X"):
        events.append("inside")

    captured = capsys.readouterr().out
    lines = captured.splitlines()

    assert events == ["inside"]
    assert lines[0] == "X..."
    assert lines[1].startswith("X: done in")
