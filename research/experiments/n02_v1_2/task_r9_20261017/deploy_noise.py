"""Fixed old D1 weights, two GPUs, no optimization; separate from new adaptation."""

import subprocess
from pathlib import Path

root = Path("/workspace/owac/worktrees/n02-measurement-completion")
py = root / "outputs/n02-full/runtime/py310-cu128/bin/python"
parts = []
for arm, gpu, step in [("narrow", 0, 50), ("wide", 1, 25)]:
    name = f"owac-n02-r9-noise-{arm}"
    command = (
        f"{py} research/experiments/n02_v1_2/task_r9_20261017/noise_probe.py "
        f"--config configs/encoders/n02_v12/raw_fixedfront_r8_{arm}_robotwin.json "
        f"--checkpoint outputs/n02-v12/P2/raw-backend-r8-fixedfront-{arm}-s1103-d1/"
        f"checkpoint_{step:07d}.pt "
        "--captures outputs/n02-v12/task-r9-noise-capture-a1 "
        f"--output outputs/n02-v12/task-r9-noise-probes/d1-{arm}-a1"
    )
    parts.append(f'''[program:{name}]
directory={root}
command={command}
environment=PYTHONPATH="{root}/src:/workspace/owac/rawvla/source/starVLA",CUDA_VISIBLE_DEVICES="{gpu}",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1"
autostart=false
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={root}/outputs/n02-v12/{name}.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=1
''')
p = Path("/etc/supervisor/conf.d/owac-n02-r9-noise.conf")
if p.exists():
    raise FileExistsError(p)
p.write_text("\n".join(parts))
subprocess.run(["supervisorctl", "reread"], check=True)
subprocess.run(["supervisorctl", "update"], check=True)
for arm in ["narrow", "wide"]:
    subprocess.run(["supervisorctl", "start", f"owac-n02-r9-noise-{arm}"], check=True)
