"""Credential routing must not inherit the caller's other DeepSeek identity."""

import sys
from pathlib import Path

import pytest

from owac.runtime.deepseek_credentials import ROLE_VARIABLES, environment_for_role, main


@pytest.fixture
def credentials(tmp_path):
    directory = tmp_path / ".config/owac/deepseek"
    directory.mkdir(parents=True, mode=0o700)
    for role, variable in ROLE_VARIABLES.items():
        path = directory / f"{role}.env"
        path.write_text(f"{variable}=test-{role}-secret\n")
        path.chmod(0o600)
    return directory


@pytest.mark.parametrize("role", ROLE_VARIABLES)
def test_role_replaces_generic_key_and_removes_other_identity(credentials, role):
    parent = {
        "PATH": "/usr/bin",
        "DEEPSEEK_API_KEY": "stale-generic-secret",
        "DEEPSEEK_CODING_API_KEY": "inherited-coding-secret",
        "DEEPSEEK_CLOSED_LOOP_API_KEY": "inherited-loop-secret",
    }
    child = environment_for_role(role, credential_dir=credentials, environment=parent)
    chosen = ROLE_VARIABLES[role]
    assert child[chosen] == child["DEEPSEEK_API_KEY"] == f"test-{role}-secret"
    assert child["PATH"] == parent["PATH"]
    assert parent["DEEPSEEK_API_KEY"] == "stale-generic-secret"
    assert not (set(ROLE_VARIABLES.values()) - {chosen}) & child.keys()


def test_missing_role_does_not_fall_back_to_parent_or_other_file(credentials):
    (credentials / "closed-loop.env").unlink()
    with pytest.raises(FileNotFoundError):
        environment_for_role(
            "closed-loop",
            credential_dir=credentials,
            environment={"DEEPSEEK_API_KEY": "inherited-secret"},
        )


@pytest.mark.parametrize("problem", ["public", "wrong-role", "multiple", "symlink"])
def test_reject_unsafe_or_ambiguous_role_files(credentials, problem):
    path = credentials / "coding.env"
    if problem == "public":
        path.chmod(0o644)
    elif problem == "wrong-role":
        path.write_text("DEEPSEEK_CLOSED_LOOP_API_KEY=wrong-secret\n")
    elif problem == "multiple":
        path.write_text("DEEPSEEK_CODING_API_KEY=first\nDEEPSEEK_API_KEY=second\n")
    else:
        path.unlink()
        path.symlink_to(credentials / "closed-loop.env")
    with pytest.raises(ValueError):
        environment_for_role("coding", credential_dir=credentials, environment={})


def test_check_never_launches_or_prints_secret(credentials, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: credentials.parents[2])
    monkeypatch.setattr(sys, "argv", ["credentials", "--role", "closed-loop", "--check"])
    main()
    output = capsys.readouterr().out
    assert "DEEPSEEK_CLOSED_LOOP_API_KEY" in output
    assert "test-closed-loop-secret" not in output


def test_cli_executes_only_with_chosen_environment(credentials, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: credentials.parents[2])
    monkeypatch.setattr(
        sys, "argv", ["credentials", "--role", "coding", "--", "dsh", "--profile", "web"]
    )
    launched = []

    def capture_launch(executable, command, environment):
        launched.append((executable, command, environment))

    monkeypatch.setattr(
        "owac.runtime.deepseek_credentials.os.execvpe",
        capture_launch,
    )
    main()
    executable, command, environment = launched[0]
    assert executable == "dsh" and command == ["dsh", "--profile", "web"]
    assert environment["DEEPSEEK_API_KEY"] == "test-coding-secret"
    assert "DEEPSEEK_CLOSED_LOOP_API_KEY" not in environment
