#!/usr/bin/env python3
"""
Wrapper script to run multiple tasks N times each.

This script calls planner_demo.py via subprocess for each task N times,
organizing outputs by run ID (e.g., 1.1.1, 1.1.2, 1.1.3).

Usage:
    python scripts/run_tasks_wrapper.py --config baseline_evaluation_v1/configs/baseline_runs.yaml
    
    # With dry-run to test configuration:
    python scripts/run_tasks_wrapper.py --config baseline_evaluation_v1/configs/baseline_runs_minimal.yaml --dry-run
"""

import yaml
import subprocess
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import threading


def _tee_stream(pipe, log_path, stream):
    """Read from pipe and write to both log file and stream (runs in thread)."""
    with open(log_path, 'w') as log:
        for line in iter(pipe.readline, ''):
            log.write(line)
            log.flush()
            stream.write(line)
            stream.flush()
    pipe.close()


def _copy_html_trace_file(output_dir, task_name, run_num, base_output_dir):
    """
    Copy HTML trace file to outputs folder with renamed format: Task_1A_1.html
    
    Args:
        output_dir: Directory where the run output is stored (e.g., base_output_dir/Task_1A/Task_1A_1)
        task_name: Name of the task (e.g., "Task_1A")
        run_num: Run number (e.g., 1)
        base_output_dir: Base output directory for the experiment
    """
    try:
        # Find HTML files in the traces directory
        # Structure based on planner_demo.py: output_dir/{task_name}/traces/0/*.html
        # where output_dir is set to evaluation.output_dir which is the same as paths.results_dir
        # and dataset_file is extracted from episode file name (e.g., "Task_1A")
        
        # Try multiple possible paths
        possible_paths = [
            Path(output_dir) / task_name / "traces" / "0",  # output_dir/Task_1A/traces/0
            Path(output_dir) / "traces" / "0",  # output_dir/traces/0
            Path(output_dir).parent / task_name / "traces" / "0",  # base/Task_1A/traces/0
        ]
        
        traces_dir = None
        for path in possible_paths:
            if path.exists():
                traces_dir = path
                break
        
        if traces_dir is None:
            # Try searching recursively for traces/0 directory
            output_path = Path(output_dir)
            for traces_candidate in output_path.rglob("traces/0"):
                if traces_candidate.exists():
                    traces_dir = traces_candidate
                    break
        
        if traces_dir is None or not traces_dir.exists():
            print(f"  ⚠ Traces directory not found. Searched in: {output_dir}", file=sys.stderr)
            return
        
        # Find HTML files
        html_files = list(traces_dir.glob("*.html"))
        
        if not html_files:
            print(f"  ⚠ No HTML files found in: {traces_dir}", file=sys.stderr)
            return
        
        # Use the first HTML file found (or the most recent one)
        html_file = html_files[0]
        if len(html_files) > 1:
            # If multiple HTML files, use the most recently modified one
            html_file = max(html_files, key=lambda p: p.stat().st_mtime)
        
        # Create outputs directory: outputs_{output_base_dir name}
        # e.g., base_output_dir=results/baseline_evaluation_v3_gpt5 -> results/outputs_baseline_evaluation_v3_gpt5
        outputs_dir = Path(base_output_dir).parent / ("outputs_" + Path(base_output_dir).name)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create new filename: Task_1A_1.html
        new_filename = f"{task_name}_{run_num}.html"
        dest_path = outputs_dir / new_filename
        
        # Copy the file
        shutil.copy2(html_file, dest_path)
        print(f"  ✓ Copied HTML trace: {new_filename} -> {dest_path}")
        
    except Exception as e:
        print(f"  ⚠ Failed to copy HTML trace file: {e}", file=sys.stderr)


def run_task(task_config, run_num, base_output_dir, planner_config, global_overrides=None, dry_run=False):
    """
    Run planner_demo.py for one task instance.
    
    Args:
        task_config: Dict with task_id, episode_file, runtime_objects
        run_num: Which run (1 to N)
        base_output_dir: Base results directory
        planner_config: Name of planner config
        dry_run: If True, print command without executing
    
    Returns:
        Dict with success status and run_id
    """
    task_id = task_config["task_id"]

    # Task folder (use episode_file's parent if task_folder not set)
    task_folder_path = Path(
        task_config.get("task_folder") or Path(task_config["episode_file"]).parent
    )
    task_name = task_folder_path.name

    # Create run ID using task name + run number (e.g., "Task_1A/Task_1A_1")
    run_id = f"{task_name}_{run_num}"

    # Create output directory: base_output_dir/Task_1A/Task_1A_1
    output_dir = Path(base_output_dir) / task_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract episode_id from the episode JSON file
    episode_id = None
    episode_file_path = Path(task_config['episode_file'])
    if episode_file_path.exists():
        try:
            import gzip
            # Check if it's a .gz file
            with gzip.open(episode_file_path, 'rt') as f:
                episode_data = json.load(f)

            # Extract episode_id from first episode
            if episode_data and 'episodes' in episode_data and len(episode_data['episodes']) > 0:
                episode_id = episode_data['episodes'][0].get('episode_id')
        except Exception as e:
            print(f"  ⚠ Could not extract episode_id: {e}")

    print(f"\n{'='*60}")
    print(f"Running: Task {task_id}, Run {run_num} → {run_id}")
    print(f"Episode: {task_config['episode_file']}")
    if episode_id:
        print(f"Episode ID: {episode_id}")
    print(f"Runtime Config: {task_config.get('runtime_config', 'None')}")

    # Load and display runtime config details
    if task_config.get('runtime_config'):
        import yaml
        runtime_config_path = Path(task_config['runtime_config'])
        if runtime_config_path.exists():
            try:
                with open(runtime_config_path, 'r') as f:
                    runtime_config = yaml.safe_load(f)

                # Display robot location
                if runtime_config and 'runtime_robot_location' in runtime_config:
                    robot_loc = runtime_config['runtime_robot_location']
                    if 'room' in robot_loc:
                        print(f"  🤖 Robot Location: {robot_loc['room']}")

                # Display runtime objects
                if runtime_config and 'runtime_objects' in runtime_config:
                    runtime_objs = runtime_config['runtime_objects']
                    if runtime_objs.get('enabled', False) and 'objects' in runtime_objs:
                        print(f"  🍾 Runtime Objects: {len(runtime_objs['objects'])} object(s)")
                        for idx, obj in enumerate(runtime_objs['objects'], 1):
                            obj_class = obj.get('class', 'unknown')
                            furniture = obj.get('furniture_name')
                            position = obj.get('position')
                            if furniture:
                                print(f"     {idx}. {obj_class} on {furniture}")
                            elif position:
                                print(f"     {idx}. {obj_class} at position {position}")
                            else:
                                print(f"     {idx}. {obj_class}")
                    else:
                        print(f"  🍾 Runtime Objects: None (disabled)")
            except Exception as e:
                print(f"  ⚠ Could not parse runtime config: {e}")

    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Pass task folder so planner_demo finds .json.gz/.json and .yaml inside it
    data_path = task_folder_path.resolve()

    # Build command
    cmd = [
        "python", "-m", "habitat_llm.examples.planner_demo",
        f"--config-name={planner_config}",
        f"habitat.dataset.data_path={data_path}",
        f"paths.results_dir={output_dir}",
        f"evaluation.output_dir={output_dir}",  # Set output_dir as well
    ]

    # Add global Hydra overrides (applied to all tasks)
    if global_overrides:
        cmd.extend(global_overrides)

    # Runtime config is discovered from the task folder by planner_demo (no override needed)

    # For single-episode files, use index 0 (not the episode_id)
    # episode_indices expects array indices, not episode IDs
    if episode_id:
        cmd.append(f"+episode_indices=[0]")

    # Add any additional task-specific hydra overrides
    if task_config.get("hydra_overrides"):
        for override in task_config["hydra_overrides"]:
            cmd.append(override)

    print(f"Command: {' '.join(cmd)}\n")

    if dry_run:
        print(f"[DRY RUN] Would execute command above")
        return {"success": True, "run_id": run_id, "dry_run": True}

    # Run command
    timeout_seconds = 60*60  # 60 minutes timeout
    try:
        start_time = datetime.now()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent.parent,
        )
        t_stdout = threading.Thread(
            target=_tee_stream,
            args=(proc.stdout, output_dir / "stdout.log", sys.stdout),
        )
        t_stderr = threading.Thread(
            target=_tee_stream,
            args=(proc.stderr, output_dir / "stderr.log", sys.stderr),
        )
        t_stdout.daemon = True
        t_stderr.daemon = True
        t_stdout.start()
        t_stderr.start()
        
        # Wait for process with timeout
        try:
            result = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"\n⚠ Timeout after {timeout_seconds}s ({timeout_seconds/60:.1f} minutes) - killing process for {run_id}", file=sys.stderr)
            proc.kill()  # Kill the process
            proc.wait()  # Wait for it to actually terminate
            result = -1  # Set error code
            raise subprocess.TimeoutExpired(cmd, timeout_seconds)
        
        # Wait for output threads to finish
        t_stdout.join(timeout=5)  # Give threads 5 seconds to finish after process ends
        t_stderr.join(timeout=5)
        
        if result != 0:
            raise subprocess.CalledProcessError(result, cmd)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save run metadata
        metadata = {
            "run_id": run_id,
            "task_id": task_config["task_id"],
            "run_num": run_num,
            "episode_file": task_config["episode_file"],
            "runtime_config": task_config.get("runtime_config"),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "command": " ".join(cmd),
            "success": True
        }
        with open(output_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Completed: {run_id} (duration: {duration:.1f}s)")
        
        # Copy HTML trace file to outputs folder with renamed format
        _copy_html_trace_file(output_dir, task_name, run_num, base_output_dir)
        
        return {"success": True, "run_id": run_id, "duration": duration}

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        is_timeout = isinstance(e, subprocess.TimeoutExpired)
        error_msg = f"Timeout after {timeout_seconds}s" if is_timeout else f"Error: {e}"
        print(f"✗ Failed: {run_id}", file=sys.stderr)
        print(error_msg, file=sys.stderr)

        # stdout/stderr already in stdout.log and stderr.log; create error summary
        with open(output_dir / "error.log", "w") as f:
            f.write(f"Command failed:\n{' '.join(cmd)}\n\n")
            if is_timeout:
                f.write(f"Timeout: Process exceeded {timeout_seconds}s ({timeout_seconds/60:.1f} minutes) and was killed\n\n")
            else:
                f.write(f"Exit code: {e.returncode if hasattr(e, 'returncode') else 'N/A'}\n\n")
            if hasattr(e, 'stdout') and e.stdout:
                f.write(f"Stdout:\n{e.stdout}\n\n")
            if hasattr(e, 'stderr') and e.stderr:
                f.write(f"Stderr:\n{e.stderr}\n")

        # Save error metadata
        is_timeout = isinstance(e, subprocess.TimeoutExpired)
        metadata = {
            "run_id": run_id,
            "task_id": task_config["task_id"],
            "run_num": run_num,
            "episode_file": task_config["episode_file"],
            "runtime_config": task_config.get("runtime_config"),
            "error": str(e),
            "exit_code": e.returncode if hasattr(e, 'returncode') else -1,
            "timeout": is_timeout,
            "timeout_seconds": timeout_seconds if is_timeout else None,
            "command": " ".join(cmd),
            "success": False
        }
        with open(output_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Try to copy HTML file even if there was an error (it might have been generated before timeout/failure)
        _copy_html_trace_file(output_dir, task_name, run_num, base_output_dir)

        return {"success": False, "run_id": run_id, "error": str(e), "timeout": is_timeout}


def save_task_summary(task_id, task_results, base_output_dir, task_name):
    """Save summary for all N runs of a task"""
    # Save summary in the task folder
    summary_file = Path(base_output_dir) / task_name / f"task_summary.json"

    success_count = sum(1 for r in task_results["runs"] if r.get("success", False))
    num_runs = len(task_results["runs"])

    summary = {
        "task_id": task_id,
        "task_name": task_name,
        "task_folder": task_results.get("task_folder"),
        "episode_file": task_results.get("episode_file"),
        "runtime_config": task_results.get("runtime_config"),
        "num_runs": num_runs,
        "success_count": success_count,
        "success_rate": success_count / num_runs if num_runs > 0 else 0,
        "runs": task_results["runs"]
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Task summary saved: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple PARTNR tasks N times each with organized logging"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to tasks config YAML file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        help="Only run specific task IDs (e.g., 1.1 1.2 2.1)"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_output_dir = config["output_base_dir"]
    num_runs = config["num_runs_per_task"]
    planner_config = config["planner_config"]
    tasks_raw = config["tasks"]
    tasks_root = config.get("tasks_root", "")
    global_overrides = config.get("global_hydra_overrides", [])

    # Convert simple task list to full task configs
    tasks = []
    for task in tasks_raw:
        if isinstance(task, str):
            # New simple format: just task folder name
            task_folder = Path(tasks_root) / task if tasks_root else Path(task)

            # Find episode file: {task_folder}/{folder_name}.json.gz
            episode_file = task_folder / f"{task}.json.gz"

            # Find runtime config: first .yaml file in task_folder
            runtime_config = None
            if task_folder.exists():
                yaml_files = list(task_folder.glob("*.yaml"))
                if yaml_files:
                    runtime_config = str(yaml_files[0])

            # Extract task_id from yaml filename (e.g., "T1-A.yaml" -> "1-A") or use folder name
            task_id = task
            if runtime_config:
                yaml_name = Path(runtime_config).stem  # e.g., "T1-A"
                if yaml_name.startswith('T'):
                    task_id = yaml_name[1:]  # Remove 'T' prefix

            tasks.append({
                "task_id": task_id,
                "task_folder": str(task_folder),
                "episode_file": str(episode_file),
                "runtime_config": runtime_config
            })
        else:
            # Old format: dict with task_id, task_folder, etc.
            tasks.append(task)

    # Filter tasks if specified
    if args.task_ids:
        tasks = [t for t in tasks if t["task_id"] in args.task_ids or Path(t.get("task_folder", "")).name in args.task_ids]
        if not tasks:
            print(f"Error: No tasks found matching IDs: {args.task_ids}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Baseline Evaluation Runner")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Planner: {planner_config}")
    print(f"Tasks: {len(tasks)}")
    print(f"Runs per task: {num_runs}")
    print(f"Total runs: {len(tasks) * num_runs}")
    print(f"Output directory: {base_output_dir}")
    if args.dry_run:
        print(f"Mode: DRY RUN (no execution)")
    print(f"{'='*60}\n")

    # Create base output directory
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    # Save experiment config
    experiment_config = {
        "config_file": str(config_path),
        "timestamp": datetime.now().isoformat(),
        "num_runs_per_task": num_runs,
        "planner_config": planner_config,
        "total_tasks": len(tasks),
        "total_runs": len(tasks) * num_runs,
        "task_ids": [t["task_id"] for t in tasks]
    }
    with open(Path(base_output_dir) / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    # Track all results
    all_results = []

    # Loop over tasks
    for task_idx, task_config in enumerate(tasks, 1):
        task_id = task_config["task_id"]

        # Task name from task folder (or episode_file parent for old config format)
        task_name = Path(
            task_config.get("task_folder") or Path(task_config["episode_file"]).parent
        ).name

        print(f"\n{'#'*60}")
        print(f"Task {task_idx}/{len(tasks)}: {task_id} ({task_name})")
        print(f"Episode file: {task_config['episode_file']}")
        print(f"Runtime config: {task_config.get('runtime_config', 'None')}")
        print(f"{'#'*60}")

        task_results = {
            "task_id": task_id,
            "task_name": task_name,
            "task_folder": task_config.get("task_folder"),
            "episode_file": task_config["episode_file"],
            "runtime_config": task_config.get("runtime_config"),
            "runs": []
        }

        # Run N times
        for run_num in range(1, num_runs + 1):
            result = run_task(
                task_config,
                run_num,
                base_output_dir,
                planner_config,
                global_overrides=global_overrides,
                dry_run=args.dry_run
            )
            task_results["runs"].append(result)

        # Save task summary after all N runs complete
        if not args.dry_run:
            task_summary = save_task_summary(task_id, task_results, base_output_dir, task_name)

            print(f"\n  Task {task_id} Complete:")
            print(f"    Success: {task_summary['success_count']}/{task_summary['num_runs']} ({task_summary['success_rate']:.1%})")
        else:
            print(f"\n  [DRY RUN] Task {task_id}: Would run {num_runs} times")

        all_results.append(task_results)

    # Save final experiment summary
    summary_file = Path(base_output_dir) / "experiment_summary.json"
    experiment_summary = {
        "config": experiment_config,
        "tasks": all_results,
        "overall_stats": {
            "total_runs": sum(len(t["runs"]) for t in all_results),
            "total_successes": sum(
                sum(1 for r in t["runs"] if r.get("success", False))
                for t in all_results
            )
        }
    }
    if not args.dry_run:
        experiment_summary["overall_stats"]["overall_success_rate"] = (
            experiment_summary["overall_stats"]["total_successes"]
            / experiment_summary["overall_stats"]["total_runs"]
        )

    with open(summary_file, "w") as f:
        json.dump(experiment_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All runs complete!")
    print(f"Summary saved to: {summary_file}")
    if not args.dry_run:
        print(f"Overall success rate: {experiment_summary['overall_stats']['overall_success_rate']:.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
