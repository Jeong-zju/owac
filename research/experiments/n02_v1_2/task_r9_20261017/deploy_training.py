"""Launch only the declared P2 paired budget after pilots and sensor probes complete."""

import json
import subprocess
from pathlib import Path

root = Path("/workspace/owac/worktrees/n02-measurement-completion")
py = root / "outputs/n02-full/runtime/py310-cu128/bin/python"
for arm in ("control", "task"):
    receipt = json.loads(
        (root / f"outputs/n02-v12/P2/raw-task-r9-{arm}-pilot-a1/pilot.json").read_text()
    )
    assert receipt["save_reload_action_max_abs"] == 0 and receipt["frozen_parameters_unchanged"]
for arm in ("narrow", "wide"):
    receipt = json.loads(
        (root / f"outputs/n02-v12/task-r9-noise-probes/d1-{arm}-a1/noise_probe.json").read_text()
    )
    assert len(receipt["rows"]) == 27 and receipt["optimizer_updates"] == 0
parts = []
for arm, gpu, port in [("control", 0, 18520), ("task", 1, 18521)]:
    name = f"owac-n02-r9-adapt-{arm}"
    command = (
        f"{py} research/experiments/n02_v1_2/task_r9_20261017/run.py "
        f"--config configs/encoders/n02_v12/raw_task_r9_{arm}_robotwin.json "
        f"--output outputs/n02-v12/P2/raw-task-r9-{arm}-s1103-a1 "
        f"--case outputs/n02-v12/representation-r8-rollouts/task-r9-{arm}-a1 --port {port}"
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
p = Path("/etc/supervisor/conf.d/owac-n02-r9-training.conf")
if p.exists():
    raise FileExistsError(p)
p.write_text("\n".join(parts))
subprocess.run(["supervisorctl", "reread"], check=True)
subprocess.run(["supervisorctl", "update"], check=True)
active = dict(
    title="任务表征与适配对照",
    phase="task_adaptation",
    phase_detail="同样512次样本曝光：一组学习RAW深层和融合读出并适配动作，另一组只适配动作。正常光开场不退化才进入真实抓持验收。",
    runs=[
        dict(
            run_id=f"raw-task-r9-{arm}-s1103-a1",
            stage="P2",
            label="任务表征＋动作适配" if arm == "task" else "冻结视觉对照",
            effective_batch_size=8,
            target=64,
        )
        for arm in ("task", "control")
    ],
)
(root / "outputs/n02-v12/active_training.json").write_text(
    json.dumps(active, ensure_ascii=False, indent=2) + "\n"
)
subprocess.run(["supervisorctl", "restart", "owac-n02-v12-monitor"], check=True)
for arm in ("control", "task"):
    subprocess.run(["supervisorctl", "start", f"owac-n02-r9-adapt-{arm}"], check=True)
