"""Run the EvoMDT evaluation. Finished cases are skipped on re-run, and failures are recorded, never dropped.

Outputs go to artifacts/paper_eval/ (git-ignored):
  runs/<system>/<case_id>.json          one result per case
  runs/<system>/failures.json           cases that failed every attempt
  evolution_state.json                  state frozen after the development set

Systems:
  evomdt, evomdt_*    five agents, retrieval, consensus, using the frozen evolved state
  evomdt_baseline     five agents, retrieval, consensus, with no evolved state (the first-iteration design)
  single:<nim-model>  one prompt to one model, with no agents and no retrieval (the paper's comparison arm)

Steps:
  evolve   run the development cases in small batches, applying feedback after each batch
  run      run the listed case files for the chosen system
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from evomdt.config import AppConfig, LLMConfig, load_env_file  # noqa: E402
from evomdt.evolution import apply_feedback, initial_state  # noqa: E402
from evomdt.io import iter_dataset_tasks  # noqa: E402
from evomdt.llm import OpenAIChatProvider, parse_json_content  # noqa: E402
from evomdt.models import CaseDossier, EvolutionState  # noqa: E402
from evomdt.pipeline import EvoMDTSystem  # noqa: E402
from evomdt.prompts import case_prompt_payload  # noqa: E402

DATA = ROOT / "data" / "paper_eval"
OUT = ROOT / "artifacts" / "paper_eval"
STATE_PATH = OUT / "evolution_state.json"
LOGGER = logging.getLogger("run_paper_eval")

SINGLE_PROMPT = """You are an expert physician. Answer the task below.

{task}

Return JSON only with the field final_answer.
{instructions}"""


class RequestPacer:
    """Spaces request starts, and adapts: each 429 widens the gap, and each success narrows it again.

    Hosted limits here change with load and are not published, so a fixed rate either wastes capacity or
    sets off a retry storm in which retries themselves keep the limit tripped.
    """

    # Requests queue on these slots, so the cap must stay small or a burst of 429s becomes a long backlog.
    MAX_INTERVAL = 5.0

    def __init__(self, per_minute: float):
        self.floor = 60.0 / per_minute
        self.interval = self.floor
        self.next_slot = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if delay:
            await asyncio.sleep(delay)

    def record(self, status: int) -> None:
        if status == 429:
            self.interval = min(self.MAX_INTERVAL, self.interval * 1.2)
        elif status < 400:
            self.interval = max(self.floor, self.interval * 0.85)


def build_client(args: argparse.Namespace, statuses: Counter[int]) -> AsyncOpenAI:
    pacer = RequestPacer(args.rpm)

    async def before(_: httpx.Request) -> None:
        await pacer.wait()

    async def after(response: httpx.Response) -> None:
        statuses[response.status_code] += 1
        pacer.record(response.status_code)

    return AsyncOpenAI(
        api_key=os.environ["NIM_API_KEY"],
        base_url=os.environ["NIM_BASE_URL"],
        timeout=args.timeout,
        max_retries=2,
        http_client=httpx.AsyncClient(
            timeout=args.timeout,
            limits=httpx.Limits(max_connections=64),
            event_hooks={"request": [before], "response": [after]},
        ),
    )


def load_cases(names: list[str], per_file: int | None = None) -> list[CaseDossier]:
    seen: dict[str, CaseDossier] = {}
    for name in names:
        for case in list(iter_dataset_tasks(DATA / f"{name}.jsonl"))[:per_file]:
            seen.setdefault(case.case_id, case)
    return list(seen.values())


def build_system(args: argparse.Namespace, client: AsyncOpenAI) -> tuple[EvoMDTSystem, AppConfig]:
    config = AppConfig.load(ROOT / "config.toml")
    config.runtime.artifacts_dir = str(OUT / "unused_artifacts")
    entry = config.llm_for(config.runtime.default_llm)
    provider = OpenAIChatProvider(
        LLMConfig(api_key="from-env", base_url=entry.base_url, model_name=args.backbone or entry.model_name, temperature=entry.temperature, seed=entry.seed),
        client=client,
    )
    return EvoMDTSystem(config, provider_factory=lambda _key: provider), config


async def run_single(case: CaseDossier, model: str, client: AsyncOpenAI) -> dict:
    if case.task_type == "mcq":
        task = case.question
        instructions = "final_answer must be only the letter of the correct option."
    elif case.task_type == "generation":
        task = case.question
        instructions = "final_answer must directly answer in plain clinical prose, at the length a physician would give."
    else:
        task = json.dumps(case_prompt_payload(case), ensure_ascii=False, indent=1)
        instructions = (
            "final_answer must be one paragraph giving the recommended initial management: diagnosis and stage, "
            "primary treatment and key follow-up."
        )
    started = time.perf_counter()
    reply = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SINGLE_PROMPT.format(task=task, instructions=instructions)}],
        response_format={"type": "json_object"},
        temperature=0,
        seed=7,
    )
    content = reply.choices[0].message.content or ""
    answer = parse_json_content(content).get("final_answer", "")
    if not str(answer).strip():
        raise ValueError("empty final_answer")
    return {
        "case_id": case.case_id,
        "final_answer": str(answer).strip(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "total_tokens": reply.usage.total_tokens if reply.usage else None,
    }


def load_state(path: Path) -> EvolutionState:
    return EvolutionState.model_validate_json(path.read_text(encoding="utf-8"))


async def run_all(args: argparse.Namespace) -> int:
    statuses: Counter[int] = Counter()
    client = build_client(args, statuses)
    cases = load_cases(args.files, args.per_file)
    if args.limit:
        cases = cases[: args.limit]
    folder = OUT / "runs" / re.sub(r"[^a-z0-9._-]+", "_", args.system.lower())
    folder.mkdir(parents=True, exist_ok=True)
    pending = [case for case in cases if not (folder / f"{case.case_id}.json").exists()]
    print(f"{args.system}: {len(cases)} selected, {len(pending)} to run", flush=True)

    system = state = None
    if not args.system.startswith("single:"):
        system, config = build_system(args, client)
        state = initial_state(config) if args.system == "evomdt_baseline" else load_state(STATE_PATH)

    gate = asyncio.Semaphore(args.concurrency)
    failures: dict[str, str] = {}
    done = 0
    started = time.perf_counter()

    async def one(case: CaseDossier) -> None:
        nonlocal done
        async with gate:
            error = ""
            for attempt in range(1, args.attempts + 1):
                try:
                    if system is None:
                        payload = await run_single(case, args.system.split(":", 1)[1], client)
                    else:
                        result = await system.run_case(case, persist=False, state=state)
                        payload = json.loads(result.model_dump_json())
                    payload["attempts"] = attempt
                    (folder / f"{case.case_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
                    error = ""
                    break
                except Exception as exc:
                    cause = getattr(exc, "last_attempt", None)
                    inner = cause.exception() if cause is not None else exc
                    error = f"{type(inner).__name__}: {str(inner)[:300]}"
                    LOGGER.warning("%s attempt %d failed: %s", case.case_id, attempt, error)
                    await asyncio.sleep(30 * attempt)
            if error:
                failures[case.case_id] = error
            done += 1
            if done % 10 == 0 or done == len(pending) or error:
                rate = done / max(time.perf_counter() - started, 1) * 60
                print(
                    f"[{done}/{len(pending)}] {rate:.1f} cases/min, failures={len(failures)}, http={dict(statuses)}"
                    + (f" | {case.case_id}: {error[:120]}" if error else ""),
                    flush=True,
                )

    await asyncio.gather(*(one(case) for case in pending))
    existing = {}
    failure_path = folder / "failures.json"
    if failure_path.exists():
        existing = json.loads(failure_path.read_text(encoding="utf-8"))
    existing = {key: value for key, value in existing.items() if not (folder / f"{key}.json").exists()}
    existing.update(failures)
    failure_path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    print(f"done: wall={time.perf_counter() - started:.0f}s failures={len(failures)} http={dict(statuses)}", flush=True)
    return 0  # Case failures are recorded in failures.json and retried on re-run. Only a crash stops the chain.


async def evolve(args: argparse.Namespace) -> int:
    statuses: Counter[int] = Counter()
    client = build_client(args, statuses)
    system, config = build_system(args, client)
    cases = load_cases(args.files)
    if args.limit:
        cases = cases[: args.limit]
    state = initial_state(config)
    folder = OUT / "runs" / "evolution_dev"
    folder.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(case: CaseDossier):
        async with gate:
            for attempt in range(args.attempts):
                try:
                    return await system.run_case(case, persist=False, state=state)
                except Exception as exc:
                    LOGGER.warning("%s evolve attempt %d: %s", case.case_id, attempt, exc)
                    await asyncio.sleep(15 * (attempt + 1))
            return None

    for start in range(0, len(cases), args.batch):
        batch = cases[start : start + args.batch]
        results = await asyncio.gather(*(one(case) for case in batch))
        for result in results:
            if result is None or result.evaluation is None:
                continue
            state = apply_feedback(config, state, result.evaluation)
            (folder / f"{result.case.case_id}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        STATE_PATH.write_text(state.model_dump_json(indent=1), encoding="utf-8")
        print(f"[{start + len(batch)}/{len(cases)}] weights={state.role_weights} top_k={state.retrieval_top_k} http={dict(statuses)}", flush=True)
    print(json.dumps({"role_weights": state.role_weights, "retrieval_top_k": state.retrieval_top_k, "prompt_refinements": state.prompt_refinements}, indent=1), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["run", "evolve"])
    parser.add_argument("--system", default="evomdt")
    parser.add_argument("--files", nargs="+", required=True, help="Case file names under data/paper_eval, without .jsonl.")
    parser.add_argument("--backbone", help="Override the backbone model for EvoMDT runs.")
    parser.add_argument("--concurrency", type=int, default=4, help="Cases at once.")
    parser.add_argument("--rpm", type=float, default=38.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-file", type=int, help="Use only the first N cases of each file, so a smaller run stays balanced.")
    parser.add_argument("--batch", type=int, default=8, help="Evolve only: feedback is applied after each batch.")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env_file(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=OUT / "run.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(evolve(args) if args.step == "evolve" else run_all(args))


if __name__ == "__main__":
    sys.exit(main())
