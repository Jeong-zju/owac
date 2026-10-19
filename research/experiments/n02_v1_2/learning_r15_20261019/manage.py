"""Manage only this bounded sequence's foreground jobs through supervisor."""

import argparse
import configparser
import json
import shlex
from pathlib import Path
from xmlrpc.client import ServerProxy

from supervisor.xmlrpc import SupervisorTransport


def main(args):
    root = Path("/workspace/owac/worktrees/n02-measurement-completion")
    evidence = root / "research/experiments/n02_v1_2/learning_r15_20261019"
    evidence.mkdir(parents=True, exist_ok=True)
    if not args.name.startswith("owac-n02-r15-"):
        raise ValueError("only r15 jobs are managed here")
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read("/etc/supervisor/supervisord.conf")
    server = ServerProxy(
        "http://localhost",
        transport=SupervisorTransport(None, None, cfg.get("supervisorctl", "serverurl")),
    )
    if args.command:
        path = Path("/etc/supervisor/conf.d") / (args.name + ".conf")
        if path.exists():
            raise FileExistsError("preserve earlier attempt: " + str(path))
        command = args.command[1:] if args.command[0] == "--" else args.command
        text = (
            f"[program:{args.name}]\ncommand={shlex.join(command)}\ndirectory={root}\n"
            f'environment=CUDA_VISIBLE_DEVICES="{args.gpu}",PYTHONUNBUFFERED="1",'
            f'OMP_NUM_THREADS="4",PYTHONPATH="{root}/src:/workspace/owac/rawvla/source/starVLA"\n'
            "autostart=false\nautorestart=false\nstartsecs=1\nstopasgroup=true\n"
            "killasgroup=true\nstopwaitsecs=30\nredirect_stderr=true\n"
            f"stdout_logfile={evidence}/{args.name}.log\nstdout_logfile_maxbytes=0\n"
        )
        path.write_text(text)
        added, _, _ = server.supervisor.reloadConfig()[0]
        if args.name not in added:
            raise RuntimeError("new supervisor group was not added")
        server.supervisor.addProcessGroup(args.name)
        server.supervisor.startProcess(args.name, True)
    info = server.supervisor.getProcessInfo(args.name)
    print(json.dumps({k: info[k] for k in ("name", "statename", "pid", "exitstatus")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    main(parser.parse_args())
