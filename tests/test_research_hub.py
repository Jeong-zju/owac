"""Behavioral checks for source fidelity, freshness and a read-only local server."""

import hashlib
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from owac.runtime.research_hub import (
    build_snapshot,
    make_handler,
    read_document,
    render_page,
    stage_section,
    write_snapshot,
)


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / "repo"
    root = tmp_path / "worktrees/n02"
    repo.mkdir()
    root.mkdir(parents=True)
    (repo / "AGENTS.md").write_text("# Shared instructions\n")
    content = "# P0\r\n\r\n中文 exact text — `x`\r\n</script><script>bad()</script>\r\n"
    files = {
        "README.md": "# N02\n",
        "AGENTS.md": "# Instructions\n",
        "docs/status.md": "# Status\n\nPending independent audit.\n",
        "docs/prompts/n02/P0.md": content,
        "docs/prompts/n02/CONTRACT.md": "# Contract\n\nOne stage only.\n",
        "docs/prompts/DeepSeek_N04_implementation.md": "# Other route copy\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    registry = {
        "reviewed_at": "2026-10-14",
        "shared_documents": ["AGENTS.md"],
        "routes": [
            {
                "id": "N02",
                "checkout": "worktrees/n02",
                "branch": "lzh/n02",
                "version": "v1.1",
                "documents": [],
                "summary_sources": {
                    "docs/status.md": hashlib.sha256(files["docs/status.md"].encode()).hexdigest()
                },
                "stages": [
                    {
                        "id": "P0",
                        "title": "Foundation",
                        "prompt_kind": "original",
                        "prompt_sources": [
                            "docs/prompts/n02/CONTRACT.md",
                            "docs/prompts/n02/P0.md",
                        ],
                    }
                ],
            }
        ],
    }
    registry_path = repo / "hub.json"
    registry_path.write_text(json.dumps(registry))
    return repo, root, registry_path, files


def test_source_bytes_and_packet_are_exact(workspace):
    repo, root, registry, files = workspace
    snapshot = build_snapshot(repo, registry)
    route = snapshot["routes"][0]
    source = next(d for d in route["documents"] if d["relative"].endswith("P0.md"))
    assert source["content"] == files["docs/prompts/n02/P0.md"]
    assert source["sha256"] == hashlib.sha256(source["content"].encode()).hexdigest()
    packet = route["stages"][0]["prompt"]["text"]
    assert str(root) in packet
    assert files["docs/prompts/n02/CONTRACT.md"] in packet
    assert source["content"] in packet
    assert not any("N04" in d["relative"] for d in route["documents"])


def test_refresh_picks_up_new_source_and_flags_old_summary(workspace):
    repo, root, registry, _ = workspace
    assert not build_snapshot(repo, registry)["routes"][0]["summary_stale"]
    (root / "docs/status.md").write_text("# Status\n\nNew result.\n")
    (root / "docs/prompts/n02/P0.md").write_text("# P0 changed\n")
    route = build_snapshot(repo, registry)["routes"][0]
    assert route["summary_stale"]
    assert "# P0 changed\n" in route["stages"][0]["prompt"]["text"]
    assert any("摘要来源已变化" in warning for warning in route["warnings"])


def test_complete_packet_keeps_its_own_scope_without_legacy_wrapper(workspace):
    repo, _, registry, files = workspace
    settings = json.loads(registry.read_text())
    stage = settings["routes"][0]["stages"][0]
    stage["prompt_kind"] = "complete"
    stage["prompt_sources"] = ["docs/prompts/n02/P0.md"]
    registry.write_text(json.dumps(settings))
    packet = build_snapshot(repo, registry)["routes"][0]["stages"][0]["prompt"]
    assert packet["available"]
    assert packet["text"] == files["docs/prompts/n02/P0.md"]
    assert "不越过阶段门槛" not in packet["text"]


def test_missing_prompt_fails_without_falling_back(workspace):
    repo, root, registry, _ = workspace
    (root / "docs/prompts/n02/P0.md").unlink()
    prompt = build_snapshot(repo, registry)["routes"][0]["stages"][0]["prompt"]
    assert prompt["available"] is False
    assert "P0.md" in prompt["reason"]


def test_path_and_symlink_cannot_escape_checkout(workspace, tmp_path):
    _, root, _, _ = workspace
    (tmp_path / "outside.md").write_text("private")
    (root / "link.md").symlink_to(tmp_path / "outside.md")
    assert not read_document(root, "../../outside.md", "N02")["available"]
    assert not read_document(root, "link.md", "N02")["available"]


def test_plan_stage_boundaries_and_missing_stage():
    text = "# Plan\n\n### S0：基础\n\nfirst\n\n### S1：测量\n\nsecond\n\n## 9. Budget\n"
    assert stage_section(text, "S0") == "### S0：基础\n\nfirst\n\n"
    assert stage_section(text, "S1") == "### S1：测量\n\nsecond\n\n"
    with pytest.raises(ValueError, match="S8"):
        stage_section(text, "S8")


def test_offline_page_embeds_safe_complete_source(workspace, tmp_path):
    repo, _, registry, files = workspace
    output = tmp_path / "output"
    snapshot = write_snapshot(repo, output, registry)
    page = (output / "index.html").read_text()
    assert page == render_page(snapshot)
    assert "</script><script>bad()" not in page
    payload = page.split('<script type="application/json" id="hub-data">', 1)[1].split(
        "</script>", 1
    )[0]
    data = json.loads(payload)
    assert data["live"] is False
    assert files["docs/prompts/n02/P0.md"] in data["routes"][0]["stages"][0]["prompt"]["text"]
    inventory = json.loads((output / "inventory.json").read_text())
    assert "content" not in inventory["routes"][0]["documents"][0]


def test_server_refreshes_only_allowlisted_endpoints(workspace):
    repo, root, registry, _ = workspace
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(repo, registry))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/") as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
        (root / "docs/prompts/n02/P0.md").write_text("# Live update\n")
        with urlopen(url + "/api/snapshot") as response:
            snapshot = json.load(response)
        assert snapshot["live"] is True
        assert "Live update" in snapshot["routes"][0]["stages"][0]["prompt"]["text"]
        for forbidden in ("/AGENTS.md", "/../../outside.md", "/api/execute"):
            with pytest.raises(HTTPError) as error:
                urlopen(url + forbidden)
            assert error.value.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_actual_registry_packets_resolve_and_do_not_mix_routes():
    repo = Path(__file__).resolve().parents[1]
    snapshot = build_snapshot(repo)
    for route in snapshot["routes"]:
        if not route["exists"]:
            continue  # Portable clones need not contain this machine's sibling worktrees.
        for stage in route["stages"]:
            if stage.get("prompt_kind", "none") == "none":
                continue
            assert stage["prompt"]["available"], (route["id"], stage["id"])
            assert route["checkout"] in stage["prompt"]["text"]
            assert all(route["id"].lower() in path.lower() for path in stage["prompt"]["sources"])


def test_refresh_runs_make_in_repo_and_reports_failures(workspace):
    repo, _, registry, _ = workspace
    makefile = repo / "Makefile"
    makefile.write_text("research-hub:\n\t@pwd > actual-cwd.txt\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(repo, registry))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/rebuild"

    def post(headers):
        return urlopen(Request(url, method="POST", headers=headers))

    try:
        for headers in ({}, {"X-OWAC-Hub": "1", "Origin": "https://unrelated.example"}):
            with pytest.raises(HTTPError) as error:
                post(headers)
            assert error.value.code == 403
        assert not (repo / "actual-cwd.txt").exists()
        with post({"X-OWAC-Hub": "1"}) as response:
            data = json.load(response)
        assert (repo / "actual-cwd.txt").read_text().strip() == str(repo)
        receipt = data["last_refresh"]
        assert receipt["command"] == ["make", "research-hub"]
        assert receipt["cwd"] == str(repo)
        assert receipt["exit_code"] == 0
        assert data["live"] is True
        assert json.loads((repo / "outputs/research-hub/last-refresh.json").read_text()) == receipt
        makefile.write_text("research-hub:\n\t@exit 1\n")
        with pytest.raises(HTTPError) as error:
            post({"X-OWAC-Hub": "1"})
        assert error.value.code == 500
        assert "失败" in json.load(error.value)["error"]
        assert (repo / "outputs/research-hub/last-refresh.log").read_text()
        # A failed build releases the lock, so the next corrected request can succeed.
        makefile.write_text("research-hub:\n\t@pwd > actual-cwd.txt\n")
        with post({"X-OWAC-Hub": "1"}) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
