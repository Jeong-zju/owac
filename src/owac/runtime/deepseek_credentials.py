"""Launch one DeepSeek consumer with only its role-specific private credential."""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Mapping
from pathlib import Path

ROLE_VARIABLES = {
    "coding": "DEEPSEEK_CODING_API_KEY",
    "closed-loop": "DEEPSEEK_CLOSED_LOOP_API_KEY",
}


def environment_for_role(
    role: str,
    *,
    credential_dir: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read the selected private file; never fall back to another credential."""
    if role not in ROLE_VARIABLES:
        raise ValueError(f"未知 DeepSeek 凭据角色：{role}")
    variable = ROLE_VARIABLES[role]
    directory = credential_dir or Path.home() / ".config/owac/deepseek"
    path = directory / f"{role}.env"
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError(f"凭据必须是当前用户所有的私有普通文件（权限 600）：{path}")
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != 1 or not entries[0].startswith(variable + "="):
        raise ValueError(f"凭据文件必须只定义 {variable}：{path}")
    key = entries[0].split("=", 1)[1]
    if not key or any(character.isspace() for character in key):
        raise ValueError(f"{variable} 缺失或格式无效：{path}")
    result = dict(os.environ if environment is None else environment)
    for name in (*ROLE_VARIABLES.values(), "DEEPSEEK_API_KEY"):
        result.pop(name, None)
    result[variable] = key
    # Compatibility alias is scoped to this child, never exported globally.
    result["DEEPSEEK_API_KEY"] = key
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLE_VARIABLES)
    parser.add_argument("--check", action="store_true", help="仅检查配置，不调用模型")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not args.check and not command:
        parser.error("必须提供 --check 或 -- 后的启动命令")
    try:
        environment = environment_for_role(args.role)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"DeepSeek 凭据配置错误：{exc}\n")
    if args.check:
        print(f"{args.role}: {ROLE_VARIABLES[args.role]} 已配置（未调用 API）")
        return
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
