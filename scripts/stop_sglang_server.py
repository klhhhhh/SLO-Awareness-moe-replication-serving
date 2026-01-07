# bench_sglang/stop_server.py
import argparse
import os
import signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--grace", type=int, default=10, help="seconds before SIGKILL")
    args = ap.parse_args()

    try:
        pgid = os.getpgid(args.pid)
    except ProcessLookupError:
        print("Process not found.")
        return

    os.killpg(pgid, signal.SIGTERM)
    print(f"Sent SIGTERM to process group {pgid}")

    # optional: wait a bit and SIGKILL if still alive
    import time
    t0 = time.time()
    while time.time() - t0 < args.grace:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            print("Process exited.")
            return
        time.sleep(1)

    os.killpg(pgid, signal.SIGKILL)
    print(f"Sent SIGKILL to process group {pgid}")


if __name__ == "__main__":
    main()
