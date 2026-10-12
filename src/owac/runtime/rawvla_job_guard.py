"""Forward supervisor termination to owned simulation process groups."""

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def terminate_children(signum, frame):
    listing = subprocess.check_output(["ps", "-eo", "pid,ppid,pgid", "--no-headers"], text=True)
    groups = {
        int(group)
        for _, parent, group in (line.split() for line in listing.splitlines())
        if int(parent) == os.getpid() and int(group) != os.getpgrp()
    }
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while groups and time.monotonic() < deadline:
        for group in list(groups):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                groups.remove(group)
        time.sleep(0.1)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("libero", "robotwin", "paper"))
    args, remaining = parser.parse_known_args()
    signal.signal(signal.SIGTERM, terminate_children)
    signal.signal(signal.SIGINT, terminate_children)
    from owac.runtime import rawvla_full_eval

    original = rawvla_full_eval.run_process

    def preserve_debug(command, cwd, environment, log, timeout):
        command = list(command)
        if "--debug-dump-dir" in command:
            index = command.index("--debug-dump-dir") + 1
            directory = Path(command[index]) / f"launch-{uuid.uuid4().hex[:12]}"
            command[index] = str(directory)
        return original(command, cwd, environment, log, timeout)

    rawvla_full_eval.run_process = preserve_debug
    if args.mode == "paper":
        from owac.runtime.rawvla_paper_eval import main as run

        sys.argv = [sys.argv[0], *remaining]
    else:
        from owac.runtime.rawvla_full_eval import main as run

        sys.argv = [sys.argv[0], args.mode, *remaining]
    run()


if __name__ == "__main__":
    main()
