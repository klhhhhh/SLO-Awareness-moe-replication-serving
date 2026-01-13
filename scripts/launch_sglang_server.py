# bench_sglang/launch_server.py
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx


def wait_ready(url: str, timeout_s: int = 180) -> None:
    """
    Poll an HTTP endpoint until it responds 200.
    Use /v1/models if you run OpenAI-compatible server.
    """
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"Server not ready after {timeout_s}s. Last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--backend", default="sglang", choices=["sglang"])
    ap.add_argument(
        "--health-url",
        default="",  # if empty, auto use /v1/models
        help="Health check URL. Default: http://127.0.0.1:<port>/v1/models",
    )
    ap.add_argument(
        "--launch-mode",
        default="module",
        choices=["module", "cli"],
        help="module: python -m <module>; cli: <executable>",
    )
    ap.add_argument(
        "--entry",
        default="sglang.launch_server",
        help="When launch-mode=module: python -m ENTRY ...; when cli: ENTRY is executable",
    )
    ap.add_argument("--extra-args", default="", help="Extra args appended verbatim")

    args = ap.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.logdir) / f"sglang_{args.port}.log"

    # ======== 构造启动命令（你只需要在这里对齐你版本的参数名）========
    if args.launch_mode == "module":
        cmd = [
            "python",
            "-m",
            args.entry,  # e.g. sglang.launch_server
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        # 常见并行参数（如果你版本里不是这个名字，改一下即可）
        cmd += ["--tensor-parallel-size", str(args.tp)]
    else:
        # cli 方式：ENTRY 就是可执行文件名，如 sglang_server
        cmd = [
            args.entry,
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--tensor-parallel-size",
            str(args.tp),
        ]

    if args.extra_args.strip():
        cmd += args.extra_args.strip().split()

    # ======== 启动子进程（独立进程组，便于 kill）========
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # create a new process group
            env=os.environ.copy(),
        )

    base = f"http://127.0.0.1:{args.port}"
    health_url = args.health_url.strip() or f"{base}/v1/models"  # OpenAI-compatible 常用
    wait_ready(health_url, timeout_s=180)

    # 输出给上层 harness 使用
    print(f"SERVER_PID={proc.pid}")
    print(f"BASE_URL={base}")
    print(f"HEALTH_URL={health_url}")
    print(f"LOG={log_path}")


if __name__ == "__main__":
    main()
