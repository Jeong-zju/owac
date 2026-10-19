"""Local research navigation across the OWAC checkouts.

The registry records editorial choices; documents and Git metadata are read afresh.
Refresh runs the fixed ``make research-hub`` target in the main repo only.
No training, worktree mutation, external service, or third-party dependency is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

MAX_DOCUMENT_BYTES = 1_000_000
TIMEZONE = ZoneInfo("Asia/Shanghai")


def timestamp(value: float | None = None) -> str:
    return (
        datetime.fromtimestamp(value, TIMEZONE).isoformat(timespec="seconds")
        if value
        else (datetime.now(TIMEZONE).isoformat(timespec="seconds"))
    )


def confined_file(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks escaping a registered checkout."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"文件超出 checkout：{relative}")
    return candidate


def read_document(root: Path, relative: str, route_id: str) -> dict:
    result = {"id": f"{route_id}:{relative}", "route_id": route_id, "relative": relative}
    try:
        path = confined_file(root, relative)
        if path.suffix.lower() != ".md":
            raise ValueError("仅收录 Markdown 原文")
        result["path"] = str(path)
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError("文档超过单文件读取上限")
        raw = path.read_bytes()
        content = raw.decode("utf-8")
        headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
        result.update(
            title=headings[0] if headings else path.stem,
            content=content,
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
            modified_at=timestamp(path.stat().st_mtime),
            available=True,
            category=document_category(relative),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        result.update(title=relative, available=False, error=str(exc), content="")
    return result


def document_category(relative: str) -> str:
    if "/prompts/" in relative:
        return "提示词"
    if "/plans/" in relative or "full_development" in relative:
        return "技术方案"
    if "/experiments/" in relative:
        return "实验与验收"
    if "/reviews/" in relative:
        return "研究与复查"
    if "/candidates/" in relative:
        return "候选说明"
    return "项目说明"


def git_state(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=8,
        ).stdout.strip()

    try:
        # -uno avoids scanning large untracked experiment trees. Report this scope explicitly.
        changes = run("status", "--porcelain", "--untracked-files=no")
        return {
            "available": True,
            "branch": run("branch", "--show-current") or "detached HEAD",
            "head": run("rev-parse", "HEAD"),
            "tracked_changes": len(changes.splitlines()) if changes else 0,
            "change_scope": "已跟踪文件；未跟踪文档另行标记",
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def discover_documents(root: Path, route: dict) -> list[str]:
    """Select route-owned prose only, not the other prompts copied into its tree."""
    own_id = route["id"].lower()
    paths = {"README.md", "AGENTS.md", "docs/status.md"}
    paths.update(route.get("documents", []))
    for folder in (
        "docs",
        "research/candidates",
        "research/experiments",
        "research/reviews",
    ):
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.md"):
            relative = path.relative_to(root).as_posix()
            if "/results/" not in relative and own_id in relative.lower():
                paths.add(relative)
    return sorted(paths)


def stage_section(content: str, stage_id: str) -> str:
    """Extract one plan stage, preserving its exact bytes between headings."""
    match = re.search(rf"^### {re.escape(stage_id)}[：:].*$", content, re.MULTILINE)
    if not match:
        raise ValueError(f"方案中找不到阶段 {stage_id}")
    end = re.search(r"^#{1,3} ", content[match.end() :], re.MULTILINE)
    return content[match.start() : match.end() + end.start() if end else len(content)]


def prompt_packet(route: dict, stage: dict, docs: dict[str, dict]) -> dict:
    """Bind an explicit stage to its actual checkout and exact source documents."""
    kind = stage.get("prompt_kind", "none")
    if kind == "none":
        return {"available": False, "kind": kind, "reason": "此节点没有独立阶段提示词"}
    sources = stage.get("prompt_sources", [])
    missing = [p for p in sources if not docs.get(p, {}).get("available")]
    if not sources or missing:
        return {
            "available": False,
            "kind": kind,
            "reason": "提示词来源缺失：" + "、".join(missing or ["未配置来源"]),
        }
    if kind == "complete":
        return {
            "available": True,
            "kind": kind,
            "sources": sources,
            "text": "\n\n".join(docs[path]["content"] for path in sources),
        }
    introduction = (
        f"请进入 {route['checkout']}。\n"
        f"目标分支：{route['branch']}。\n"
        f"路线：{route['id']} / {route['version']}；本次节点：{stage['id']} {stage['title']}。\n"
        "先完整阅读该 checkout 的 AGENTS.md、README.md、docs/status.md，"
        "并核对当前未提交修改、前序交付与独立验收。\n"
        f"{stage.get('instruction', '')}\n"
        "本次只处理所选节点；前置验收未满足时，先报告缺项，不越过阶段门槛。\n"
    )
    if kind == "plan":
        introduction += "以下是完整方案原文。仅执行所选阶段，其余阶段用于理解接口和约束。\n"
    elif kind == "resume":
        introduction += (
            "这是继续研究前的状态核对：先对照下面的当前记录，列出尚未解决的问题、"
            "证据缺口和最小下一步，供确定下一次开发范围；不重新执行历史原型提示词。\n"
        )
    chunks = [introduction]
    for relative in sources:
        doc = docs[relative]
        chunks.append(f"\n---\n来源：{doc['path']}\nSHA256：{doc['sha256']}\n\n{doc['content']}")
    return {"available": True, "kind": kind, "sources": sources, "text": "".join(chunks)}


def build_snapshot(repo: Path, registry_path: Path | None = None) -> dict:
    repo = repo.resolve()
    registry_path = registry_path or repo / "configs/research/hub.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    workspace = repo.parent
    snapshot = {
        "schema_version": 1,
        "generated_at": timestamp(),
        "repo": str(repo),
        "registry": str(registry_path),
        "reviewed_at": registry["reviewed_at"],
        "routes": [],
        "shared_documents": [],
        "warnings": [],
    }
    for entry in registry["routes"]:
        route = dict(entry)
        root = (workspace / route["checkout"]).resolve()
        if not root.is_relative_to((workspace / "worktrees").resolve()):
            raise ValueError(f"checkout 不在 worktrees 中：{root}")
        route["checkout"] = str(root)
        route["git"] = git_state(root)
        route["exists"] = root.is_dir()
        route["warnings"] = []
        documents = [
            read_document(root, path, route["id"]) for path in discover_documents(root, route)
        ]
        docs = {doc["relative"]: doc for doc in documents}
        try:
            tracked = set(
                subprocess.run(
                    ["git", "-C", str(root), "ls-files", "-z"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=8,
                ).stdout.split("\0")
            )
        except (OSError, subprocess.SubprocessError):
            tracked = set()
        for doc in documents:
            doc["tracked"] = doc["relative"] in tracked
        route["documents"] = documents
        route["summary_stale"] = any(
            docs.get(path, {}).get("sha256") != expected
            for path, expected in route.get("summary_sources", {}).items()
        )
        if route["summary_stale"]:
            route["warnings"].append("摘要来源已变化，请复核当前节点；原文已读取最新内容。")
        if not route["exists"]:
            route["warnings"].append("工作树不存在；不可复制阶段交接包。")
        if route["git"].get("branch") != route["branch"]:
            route["warnings"].append("实际分支与登记分支不一致，请核对工作树。")
        for stage in route["stages"]:
            stage["prompt"] = prompt_packet(route, stage, docs)
            if not route["exists"]:
                stage["prompt"]["available"] = False
            if stage.get("section_source"):
                content = docs.get(stage["section_source"], {}).get("content", "")
                try:
                    stage["excerpt"] = stage_section(content, stage["id"])
                except ValueError as exc:
                    stage["excerpt"] = ""
                    stage["prompt"] = {"available": False, "reason": str(exc)}
                    route["warnings"].append(str(exc))
        route["latest_status"] = (
            docs.get("docs/status.md", {}).get("content", "").split("\n\n")[1:2]
        )
        snapshot["routes"].append(route)
    snapshot["shared_documents"] = [
        read_document(repo, path, "shared") for path in registry["shared_documents"]
    ]
    return snapshot


def render_page(snapshot: dict, *, live: bool = False) -> str:
    template = Path(__file__).with_name("research_hub_template.html").read_text(encoding="utf-8")
    # Prevent source Markdown from closing the inert JSON script element.
    payload = json.dumps({**snapshot, "live": live}, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("__HUB_DATA__", payload)


def write_snapshot(repo: Path, output: Path, registry_path: Path | None = None) -> dict:
    snapshot = build_snapshot(repo, registry_path)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "index.html"
    pending = destination.with_suffix(".tmp")
    pending.write_text(render_page(snapshot), encoding="utf-8")
    pending.replace(destination)
    (output / "inventory.json").write_text(
        json.dumps(
            {
                "generated_at": snapshot["generated_at"],
                "routes": [
                    {
                        "id": route["id"],
                        "checkout": route["checkout"],
                        "git": route["git"],
                        "summary_stale": route["summary_stale"],
                        "warnings": route["warnings"],
                        "documents": [
                            {k: v for k, v in doc.items() if k != "content"}
                            for doc in route["documents"]
                        ],
                    }
                    for route in snapshot["routes"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot


def make_handler(repo: Path, registry: Path | None = None) -> type[BaseHTTPRequestHandler]:
    repo = repo.resolve()
    rebuild_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/rebuild":
                self.send_error(404)
                return
            # A fixed endpoint, not a general command runner. Cross-origin pages cannot
            # supply this header without CORS permission, which this server never grants.
            origin = self.headers.get("Origin")
            if self.headers.get("X-OWAC-Hub") != "1" or (
                origin and origin != f"http://{self.headers.get('Host')}"
            ):
                self.send_json(403, {"error": "请从统筹台本地服务页面刷新"})
                return
            if not rebuild_lock.acquire(blocking=False):
                self.send_json(409, {"error": "页面正在生成，请稍后刷新"})
                return
            started_at = timestamp()
            started = time.monotonic()
            try:
                result = subprocess.run(
                    ["make", "research-hub"],
                    cwd=repo,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                receipt = {
                    "command": ["make", "research-hub"],
                    "cwd": str(repo),
                    "started_at": started_at,
                    "completed_at": timestamp(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "exit_code": result.returncode,
                }
                output = repo / "outputs/research-hub"
                output.mkdir(parents=True, exist_ok=True)
                (output / "last-refresh.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                (output / "last-refresh.log").write_text(
                    result.stdout + result.stderr, encoding="utf-8"
                )
                if result.returncode:
                    self.send_json(500, {"error": "make research-hub 失败，详见 last-refresh.log"})
                    return
                self.send_json(
                    200,
                    {
                        **build_snapshot(repo, registry),
                        "live": True,
                        "last_refresh": receipt,
                    },
                )
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                self.send_json(500, {"error": f"生成失败：{exc}"})
            finally:
                rebuild_lock.release()

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path not in {"/", "/index.html", "/api/snapshot"}:
                self.send_error(404)
                return
            try:
                snapshot = build_snapshot(repo, registry)
                if path == "/api/snapshot":
                    body = json.dumps({**snapshot, "live": True}, ensure_ascii=False).encode()
                    mime = "application/json; charset=utf-8"
                else:
                    body = render_page(snapshot, live=True).encode()
                    mime = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError) as exc:
                self.send_error(500, explain=str(exc))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "serve"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / "outputs/research-hub"
    snapshot = write_snapshot(repo, output, args.registry)
    print(f"本地快照：{output / 'index.html'}", flush=True)
    print(
        f"已读取 {len(snapshot['routes'])} 条路线；生成时间 {snapshot['generated_at']}", flush=True
    )
    if args.command == "serve":
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(repo, args.registry))
        print(f"Local: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
