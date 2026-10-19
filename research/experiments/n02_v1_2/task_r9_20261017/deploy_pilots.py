"""Install only this experiment's bounded supervisor jobs; never alter infrastructure."""

import subprocess
from pathlib import Path

root = Path("/workspace/owac/worktrees/n02-measurement-completion")
py = root / "outputs/n02-full/runtime/py310-cu128/bin/python"
services = []
for arm, gpu in [("control", 0), ("task", 1)]:
    name = f"owac-n02-r9-pilot-{arm}"
    command = (
        f"{py} research/experiments/n02_v1_2/task_r9_20261017/pilot.py "
        f"--config configs/encoders/n02_v12/raw_task_r9_{arm}_robotwin.json "
        f"--output outputs/n02-v12/P2/raw-task-r9-{arm}-pilot-a1 --batch 8"
    )
    services.append((name, gpu, command))
services.append(
    (
        "owac-n02-r9-sensor-capture",
        0,
        f"{py} research/experiments/n02_v1_2/task_r9_20261017/capture.py",
    )
)
parts = []
for name, gpu, command in services:
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
p = Path("/etc/supervisor/conf.d/owac-n02-r9-audits.conf")
if p.exists():
    raise FileExistsError(p)
p.write_text("\n".join(parts))
subprocess.run(["supervisorctl", "reread"], check=True)
subprocess.run(["supervisorctl", "update"], check=True)
for name, _, _ in services:
    subprocess.run(["supervisorctl", "start", name], check=True)
