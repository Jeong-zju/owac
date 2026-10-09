import json

from owac.evaluation.baseline_protocol import write_json
from owac.runtime.baseline_dashboard import load_payload, render


def test_dashboard_does_not_report_zero_success_before_any_test(tmp_path):
    (tmp_path / "progress").mkdir()
    write_json(tmp_path / "protocol.json", {"training": {"minimum_steps": 10000}})
    write_json(tmp_path / "status.json", {"status": "running", "stage": "train"})
    write_json(tmp_path / "progress/training.json", {"completed_steps": 1000})
    (tmp_path / "progress/curve.jsonl").write_text(
        '{"step": 1001, "train_flow_mse": 0.1}\n{"step": 1002, "train_flow_mse": '
    )
    payload = load_payload(tmp_path)
    assert payload["success_rate"] is None
    assert payload["episodes"] == []
    assert payload["step"] == 1001


def test_regeneration_preserves_cloud_check_time_and_deduplicates_history(tmp_path):
    (tmp_path / "progress").mkdir()
    checked = "2026-10-09T08:00:00+00:00"
    write_json(tmp_path / "progress/check.json", {"checked_at_utc": checked})
    write_json(tmp_path / "status.json", {"status": "failed", "error": "</script><script>bad"})
    output = tmp_path / "view.html"
    first = render(tmp_path, output)
    second = render(tmp_path, output)
    assert first["checked_at"] == second["checked_at"] == checked
    assert first["generated_at"] != checked
    history = [
        json.loads(line) for line in (tmp_path / "monitor-history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 1
    assert "</script><script>bad" not in output.read_text()
