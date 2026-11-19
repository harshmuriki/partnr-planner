#!/usr/bin/env python3
"""Queue isolated PARTNR runs inside one Slurm GPU allocation (Python 3.9+)."""

import argparse
import glob
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python"


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def datasets(patterns):
    found = {}
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise ValueError(f"No matches: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path.is_dir():
                path = path / "dataset.json"
                if not path.exists():
                    path = path.with_suffix(".json.gz")
            if not path.is_file() or not path.name.endswith((".json", ".json.gz")):
                raise ValueError(f"Expected a dataset JSON or episode folder: {path}")
            raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            data = json.loads(raw)
            if len(data.get("episodes", [])) != 1:
                raise ValueError(f"Each file must contain exactly one episode: {path}")
            # A glob can match both JSON and gzip. Reject ambiguous contents.
            key = str(path).removesuffix(".gz")
            if key in found and found[key][1] != raw:
                raise ValueError(f"JSON and gzip differ: {key}")
            found[key] = (path, raw, str(data["episodes"][0]["episode_id"]))
    return list(found.values())


def workspace(destination):
    destination.mkdir(parents=True)
    for name in ("habitat_llm", "scripts", "dataset_generation"):
        shutil.copytree(REPO / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "outputs", "logs"))
    for name in ("data", "third_party"):
        (destination / name).symlink_to(REPO / name, target_is_directory=True)
    # Credentials are inherited in memory, never copied into experiment output.


def command(python, planner, output, episode, overrides, probe=False, save_video=False):
    args = [python, "-u", "-m", "scripts.parallel_probe" if probe else
            "habitat_llm.examples.planner_demo", f"--config-name={planner}"]
    args += overrides
    # These controls belong to the scheduler and cannot be overridden by a job.
    args += [f"habitat.dataset.data_path={episode}",
             "+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/",
             f"paths.results_dir={output}", f"evaluation.output_dir={output}",
             f"hydra.run.dir={output}/hydra", "hydra.job.chdir=False",
             f"evaluation.save_video={save_video}", "evaluation.generate_prediviz=False",
             "num_runs_per_episode=1", "num_proc=1", "mode=cli", "device=cpu"]
    return args


def recover_partial_logs(output, status):
    # Load only the stdlib recovery module; the scheduler does not need Habitat.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "episode_progress_recovery", REPO / "habitat_llm/utils/episode_progress.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recover_run(output, status)


def result_metrics(output):
    logs = sorted(p for p in output.glob("dataset/planner-log/*.json")
                  if not p.name.startswith("checkpoint-"))
    if not logs:
        return None
    data = json.loads(logs[0].read_text())
    steps = data.get("steps", [])
    final = steps[-1] if steps else {}
    stats = final.get("stats", {})
    partial = data.get("complete") is False
    if partial:
        # A blocking skill may have advanced further than the last planner return.
        checkpoint_stats = data.get("latest_simulator_checkpoint", {}).get("stats", {})
        if checkpoint_stats:
            stats = checkpoint_stats
    if "task_state_success" not in stats:
        return None
    result = {"task_percent_complete": stats.get("task_percent_complete"),
              "task_state_success": stats["task_state_success"],
              "task_explanation": stats.get("task_explanation"),
              "cost_metrics": final.get("cost_metrics", {})}
    if final.get("termination"):
        result["termination"] = final["termination"]
    if partial:
        result.update(metrics_are_partial=True, episode_log_status=data.get("status"),
                      recorded_actions=len(data.get("actions", [])))
    cost_files = sorted(output.glob("dataset/llm_costs/*.json"))
    if cost_files:
        report = json.loads(cost_files[0].read_text())
        result["model_costs"] = report["summary"]
        result["model_cost_report"] = str(cost_files[0])
    return result


def stop_process(process):
    # Kill the group even if the root has exited: solvers may still be alive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def gpu_sample():
    try:
        data = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=5)
        return data.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", required=True,
                        help="Single-episode files, folders, or quoted globs")
    parser.add_argument("--runs-per-episode", type=int, default=1)
    parser.add_argument("--run-ids", nargs="+", help="Run only these exact repetition IDs from the generated queue")
    parser.add_argument("--workers", default="6", help="Concurrency, or auto for CPU capacity")
    parser.add_argument("--cpus-per-worker", type=int, default=2)
    parser.add_argument("--planner", choices=["vlm-tamp", "rail", "react", "tru-pomdp"], default="vlm-tamp")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=600, help="Wall seconds per run (default: 600, including exploration and API calls)")
    parser.add_argument("--job-id", help="Dispatch this coordinator into an existing Slurm allocation")
    parser.add_argument("--dry-run", action="store_true", help="Read-only preview, no API or GPU calls")
    parser.add_argument("--probe", action="store_true", help="Concurrent environment/PDDL check, no API calls")
    parser.add_argument("--save-video", action="store_true", help="Save a video for each episode")
    args = parser.parse_args(argv)
    if args.cpus_per_worker < 1 or args.runs_per_episode < 1 or args.timeout <= 0:
        parser.error("CPU count, repetition count, and timeout must be positive")
    try:
        entries = datasets(args.episodes)
        workers = None if args.workers == "auto" else int(args.workers)
        if workers is not None and workers < 1:
            raise ValueError("workers must be positive")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.job_id:
        # One Slurm step owns the coordinator and all child processes.
        info = subprocess.check_output(["scontrol", "show", "job", args.job_id, "-o"], text=True)
        fields = dict(part.split("=", 1) for part in info.split() if "=" in part)
        if fields.get("JobState") != "RUNNING" or int(fields.get("NumNodes", "0")) != 1:
            parser.error("Use a running single-node allocation")
        capacity = int(fields["NumCPUs"]) // args.cpus_per_worker
    else:
        capacity = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0)))) // args.cpus_per_worker
    workers = min(workers or capacity, len(entries) * args.runs_per_episode)
    if workers < 1 or workers > capacity:
        parser.error(f"CPU capacity is {capacity} workers at {args.cpus_per_worker} CPUs each")
    output = (args.output_dir or REPO / "results" /
              ("parallel_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if output.exists():
        parser.error(f"Use a fresh output directory; already exists: {output}")
    if args.job_id and not args.dry_run:
        forwarded = list(argv if argv is not None else sys.argv[1:])
        for i, value in enumerate(forwarded):
            if value == "--job-id":
                del forwarded[i:i + 2]
                break
            if value.startswith("--job-id="):
                del forwarded[i]
                break
        forwarded += ["--workers", str(workers), "--output-dir", str(output)]
        return subprocess.call(["srun", "--jobid=" + args.job_id, "--overlap",
                                "--ntasks=1", f"--cpus-per-task={workers * args.cpus_per_worker}",
                                args.python, str(Path(__file__).resolve()), *forwarded])
    planner = "baselines/single_agent_" + {
        "vlm-tamp": "vlm_tamp_pddl",
        "rail": "rail_planner",
        "react": "zero_shot_react_summary",
        "tru-pomdp": "tru_pomdp",
    }[args.planner]
    overrides = {
        "vlm-tamp": ["evaluation.agents.agent_0.planner.plan_config.vlm.max_completion_tokens=8192"],
        "rail": ["evaluation.agents.agent_0.planner.plan_config.vlm.max_completion_tokens=8192"],
        "react": ["evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=api"],
        "tru-pomdp": [],
    }[args.planner]
    overrides += args.override
    jobs = []
    for path, raw, episode_id in entries:
        token = hashlib.sha256(str(path).removesuffix(".gz").encode()).hexdigest()[:8]
        for run in range(1, args.runs_per_episode + 1):
            run_id = f"{path.parent.name}_{token}_r{run:03d}"
            root = output / "runs" / run_id
            jobs.append({"run_id": run_id, "episode_id": episode_id, "source": str(path),
                         "output": str(root), "status": "pending"})
    if args.run_ids:
        selected = set(args.run_ids)
        unknown = selected - {job["run_id"] for job in jobs}
        if unknown:
            parser.error("Unknown run IDs: " + ", ".join(sorted(unknown)))
        jobs = [job for job in jobs if job["run_id"] in selected]
        workers = min(workers, len(jobs))
    print(f"{len(jobs)} runs, {workers} workers, {args.cpus_per_worker} CPUs/worker", flush=True)
    print(f"Output: {output}", flush=True)
    if args.dry_run:
        for job in jobs:
            root = Path(job["output"])
            print(shlex.join(command(args.python, planner, root, root / "input/dataset.json.gz", overrides, args.probe, args.save_video)))
        return 0
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        parser.error("Run inside a Slurm step or supply --job-id; no GPU work on login nodes")
    if not gpu_sample():
        parser.error("No visible NVIDIA GPU")
    # Load a key without importing any planner or sending a request.
    if not args.probe:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is missing (environment or repository .env)")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"planner": planner, "workers": workers,
               "cpus_per_worker": args.cpus_per_worker, "probe": args.probe,
               "timeout_seconds": args.timeout, "overrides": overrides, "save_video": args.save_video,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "jobs": jobs})
    workspaces = [output / "workers" / str(i) for i in range(workers)]
    for path in workspaces:
        workspace(path)
    for job in jobs:
        root = Path(job["output"])
        (root / "input").mkdir(parents=True)
        source = Path(job["source"])
        raw = next(raw for path, raw, _ in entries if str(path) == str(source))
        (root / "input/dataset.json.gz").write_bytes(gzip.compress(raw, mtime=0))
        for name in ("initial_robot_memory.json", "scene_info.json", "spec.md", "review.json"):
            if (source.parent / name).is_file():
                shutil.copy2(source.parent / name, root / "input" / name)
    active = {}
    pending = list(jobs)
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_sample = 0
    try:
        while pending or active:
            if interrupted:
                break
            for slot in range(workers):
                if slot in active or not pending:
                    continue
                job = pending.pop(0)
                root = Path(job["output"])
                env = os.environ.copy()
                env.update({"OMP_NUM_THREADS": str(args.cpus_per_worker),
                            "MKL_NUM_THREADS": str(args.cpus_per_worker),
                            "OPENBLAS_NUM_THREADS": str(args.cpus_per_worker),
                            "HYDRA_FULL_ERROR": "1", "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTHONPATH": str(workspaces[slot]),
                            "PATH": str(Path(args.python).parent) + os.pathsep + env.get("PATH", ""),
                            "VLM_API_PROMPT_LOG_PATH": str(root / "vlm_api_prompts.txt")})
                if args.probe:
                    # First wave must coexist before any worker releases its GPU resources.
                    env.update({"PARTNR_PROBE_DIR": str(output / "probe_barrier"),
                                "PARTNR_PROBE_WORKERS": str(workers), "PARTNR_PROBE_SLOT": str(slot)})
                cmd = command(args.python, planner, root, root / "input/dataset.json.gz", overrides, args.probe, args.save_video)
                job.update(status="running", worker=slot, command=cmd, started=time.time())
                write_json(root / "run.json", job)
                # Capture the process from its first byte, including imports,
                # native simulator stderr, and failures before episode setup.
                log = (root / "log.txt").open("w", encoding="utf-8")
                (root / "console.log").symlink_to("log.txt")
                log.write(f"Simulator command started: {datetime.now().isoformat()}\n"
                          f"Working directory: {workspaces[slot]}\n"
                          f"Command: {shlex.join(cmd)}\n\n")
                log.flush()
                try:
                    process = subprocess.Popen(cmd, cwd=workspaces[slot], env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    log.write(f"Failed to start simulator command: {exc}\n")
                    log.close()
                    job.update(status="failed", error=str(exc), exit_code=None)
                    write_json(root / "run.json", job)
                    print(f"Failed to start {job['run_id']}: {exc}", flush=True)
                    continue
                active[slot] = (job, process, log)
                print("Started " + job["run_id"], flush=True)
            now = time.time()
            if now - last_sample > 10:
                with (output / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps({"time": now, "active": len(active), "gpu": gpu_sample()}) + "\n")
                last_sample = now
            for slot, (job, process, log) in list(active.items()):
                timed_out = now - job["started"] > args.timeout
                if process.poll() is None and not timed_out:
                    continue
                if timed_out:
                    stop_process(process)
                log.close()
                root = Path(job["output"])
                if timed_out or process.returncode != 0:
                    try:
                        job["recovered_logs"] = recover_partial_logs(
                            root, "timeout" if timed_out else "interrupted"
                        )
                    except (OSError, ValueError, KeyError) as exc:
                        job["log_recovery_error"] = str(exc)
                metrics = None
                try:
                    metrics = result_metrics(root) if not args.probe else json.loads((root / "probe.json").read_text())
                except (OSError, ValueError, KeyError):
                    pass
                console = (root / "console.log").read_text(errors="replace")
                caught_error = "An error occurred inside of this method:" in console or "Traceback (most recent call last):" in console
                job.update(exit_code=process.returncode, elapsed_seconds=now - job["started"], metrics=metrics,
                           status="timeout" if timed_out else "completed" if process.returncode == 0 and metrics is not None and not caught_error else "failed")
                write_json(root / "run.json", job)
                print(f"{job['status']}: {job['run_id']} task_success={metrics.get('task_state_success') if metrics else None}", flush=True)
                del active[slot]
            write_json(output / "summary.json", {"jobs": jobs})
            if active:
                time.sleep(1)
    finally:
        for job, process, log in active.values():
            stop_process(process)
            log.close()
            job.update(status="cancelled", exit_code=process.returncode)
            try:
                job["recovered_logs"] = recover_partial_logs(Path(job["output"]), "cancelled")
            except (OSError, ValueError, KeyError) as exc:
                job["log_recovery_error"] = str(exc)
            write_json(Path(job["output"]) / "run.json", job)
        for job in pending:
            job["status"] = "cancelled"
        write_json(output / "summary.json", {"jobs": jobs})
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 130 if interrupted else int(any(job["status"] != "completed" for job in jobs))


if __name__ == "__main__":
    sys.exit(main())
