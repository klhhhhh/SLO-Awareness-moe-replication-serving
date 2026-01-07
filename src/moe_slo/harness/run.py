# src/moe_slo/harness/run.py
from __future__ import annotations

import argparse, asyncio, json, os, time
import httpx

from moe_slo.adapters.sglang_client import SGLangClient, RequestRecord
from moe_slo.harness.workload import PoissonWorkload, BurstWorkload
from moe_slo.harness.metrics import summarize, window_series


async def replay(schedule_s, backend: SGLangClient, payload: dict, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    records = []

    async with httpx.AsyncClient() as client:
        t0 = time.time()

        async def launch(i: int, delay_s: float):
            # 等到预定发请求时间
            await asyncio.sleep(max(0.0, t0 + delay_s - time.time()))
            async with sem:
                rec = await backend.generate_stream(req_id=str(i), payload=payload, client=client)
                records.append(rec)

        tasks = [asyncio.create_task(launch(i, t)) for i, t in enumerate(schedule_s)]
        if tasks:
            await asyncio.gather(*tasks)

    # 保持按 req_id 排序可读
    records.sort(key=lambda r: int(r.req_id))
    return records


def save_jsonl(path: str, records: list[RequestRecord]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({
                "req_id": r.req_id,
                "send_ts": r.send_ts,
                "first_ts": r.first_ts,
                "last_ts": r.last_ts,
                "ttft_s": r.ttft_s,
                "e2e_s": r.e2e_s,
            }) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--mode", choices=["poisson", "burst"], default="poisson")

    # poisson
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=60)

    # burst
    ap.add_argument("--qps-on", type=float, default=8.0)
    ap.add_argument("--qps-off", type=float, default=1.0)
    ap.add_argument("--on-s", type=float, default=15.0)
    ap.add_argument("--off-s", type=float, default=60.0)
    ap.add_argument("--cycles", type=int, default=5)

    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--slo", type=float, default=2.0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    backend = SGLangClient(base_url=args.base_url, endpoint=args.endpoint)

    # 你可以换成自己的 prompt / messages
    payload = {
        "messages": [{"role": "user", "content": "Explain MoE in one paragraph."}],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
    }

    if args.mode == "poisson":
        wl = PoissonWorkload(qps=args.qps, duration_s=args.duration, seed=0)
        schedule_s = wl.schedule()
        tag = f"poisson_qps{args.qps}_dur{int(args.duration)}"
    else:
        wl = BurstWorkload(
            qps_on=args.qps_on, qps_off=args.qps_off,
            on_s=args.on_s, off_s=args.off_s,
            cycles=args.cycles, seed=0
        )
        schedule_s = wl.schedule()
        tag = f"burst_on{args.qps_on}x{int(args.on_s)}_off{args.qps_off}x{int(args.off_s)}_cy{args.cycles}"

    records = asyncio.run(replay(schedule_s, backend, payload, concurrency=args.concurrency))

    raw_path = os.path.join(args.outdir, f"{tag}.jsonl")
    save_jsonl(raw_path, records)

    summ = summarize(records, slo_s=args.slo)
    summ_path = os.path.join(args.outdir, f"{tag}.summary.json")
    with open(summ_path, "w") as f:
        json.dump(summ.to_dict(), f, indent=2)

    # burst 时输出时间序列
    if args.mode == "burst":
        series = window_series(records, slo_s=args.slo, window_s=1.0)
        series_path = os.path.join(args.outdir, f"{tag}.series.json")
        with open(series_path, "w") as f:
            json.dump([s.__dict__ for s in series], f, indent=2)

    print(json.dumps(summ.to_dict(), indent=2))
    print(f"raw={raw_path}")
    print(f"summary={summ_path}")


if __name__ == "__main__":
    main()
