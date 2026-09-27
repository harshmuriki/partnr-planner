# Server setup and parallel runs

This workflow transfers the current local project, installs its recorded Habitat
environment, and runs independent evaluations on a Linux server. It does not
require a web server. Commands using `USER@SERVER` and `/srv/partnr-planner` are
placeholders: replace them with your login and a writable server directory.

## 1. Transfer code and data

From the local repository root:

```bash
rsync -av --progress \
  --exclude='.git' --exclude='.env' \
  --exclude='results/' --exclude='outputs/' --exclude='logs/' \
  --exclude='temp/' --exclude='visualizations/' --exclude='__pycache__/' \
  ./ USER@SERVER:/srv/partnr-planner/
```

This copies local modifications and ignored episode/config files, including
`baseline_evaluation_v1/` and `*.json.gz`; a fresh Git clone alone omits those.
It also transfers the populated third-party source directories, including
`third_party/pddlstream`, which is not registered in `.gitmodules`.
The copy excludes previous results and credentials. Provision API credentials on
the server separately. `rsync -a` preserves symlinks: replace any links pointing
outside this project with valid server paths or copy their targets separately.

Alternatively clone a committed version with submodules, then copy your custom
code, episodes, configs, and assets. Download assets using [INSTALLATION.md](../INSTALLATION.md)
if you do not transfer `data/`. Required assets include HSSD scenes and metadata,
OVMM objects, robot/humanoid assets, and any skill checkpoints used by your planner.

## 2. Install the environment

SSH to the server and change to the project directory. Follow
[INSTALLATION.md](../INSTALLATION.md) for the complete recorded installation.
Core commands are:

```bash
conda create -n habitat-llm python=3.9.2 cmake=3.14.0 -y
conda activate habitat-llm
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
python -m pip install -e ./third_party/habitat-lab/habitat-lab
python -m pip install -e ./third_party/habitat-lab/habitat-baselines
python -m pip install -e ./third_party/transformers-CFG
python -m pip install -r requirements.txt
python -m pip install -e .
```

These are the repository's recorded pins, not a newly verified compatibility
matrix. Adjust the CUDA package for the server's driver/hardware. Even configs
with `device=cpu` can require GPU rendering through headless Habitat/EGL.
Use `nvidia-smi` to inspect the available GPUs.

For VLM-TAMP PDDL, build the transferred Fast Downward source on the server
with a C++ toolchain and make available:

```bash
(cd third_party/pddlstream && python downward/build.py)
```

Check the model/provider configured in your chosen planner and supply its API key
in the worker environment (for OpenAI, `OPENAI_API_KEY`). Do not put keys in YAML
configs or tracked files.

## 3. Preview and smoke-test one task

From the server project root:

```bash
conda activate habitat-llm
python scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs_minimal.yaml \
  --output-dir results/server_smoke --dry-run

python scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs_minimal.yaml \
  --output-dir results/server_smoke
```

The minimal config runs `Task_3_Loc` with ReAct. Then smoke-test the planner you
actually intend to evaluate; the full local `baseline_runs.yaml` selects
`baselines/single_agent_vlm_tamp_pddl`. Add your intended task folders to its
`tasks` list and choose `num_runs_per_task`. Paths in these configs are relative
to the repository root. See [wrapper configuration](WRAPPER_SCRIPT_USAGE.md).
A dry run is read-only and prints commands; only a real run checks assets,
rendering, API access, and planner execution.

## Parallel workers

Use separate working copies so PDDLStream's relative `temp/` files and other
working-directory artifacts cannot collide. Distinct output directories also
keep experiment summaries and copied traces separate.

After setup and the smoke test, create two copies on the server. This excludes
large datasets, which are shared through a symlink:

```bash
mkdir -p /srv/partnr-worker-1 /srv/partnr-worker-2
rsync -a --exclude='.git' --exclude='.env' --exclude='data/' \
  --exclude='results/' --exclude='outputs/' --exclude='logs/' \
  --exclude='temp/' --exclude='visualizations/' \
  /srv/partnr-planner/ /srv/partnr-worker-1/
rsync -a --exclude='.git' --exclude='.env' --exclude='data/' \
  --exclude='results/' --exclude='outputs/' --exclude='logs/' \
  --exclude='temp/' --exclude='visualizations/' \
  /srv/partnr-planner/ /srv/partnr-worker-2/
ln -s /srv/partnr-planner/data /srv/partnr-worker-1/data
ln -s /srv/partnr-planner/data /srv/partnr-worker-2/data
```

Use new worker directories for these commands. Keep shared assets stable during
runs. Both copies can use the same Conda environment; launch from each worker's
root so its local `habitat_llm` source is used. Editable third-party dependencies
still refer to the installed source paths, which must remain present.

For a concrete two-worker example, put `Task_3_Loc` and `Task_4_Cnt` in **both**
workers' `baseline_runs.yaml` task lists, with one repetition initially. Then:

```bash
# Worker 1
cd /srv/partnr-worker-1
conda activate habitat-llm
mkdir -p logs
nohup python -u scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs.yaml \
  --task-ids Task_3_Loc --output-dir results/worker_1 \
  > logs/worker_1.log 2>&1 &
echo $! > logs/worker_1.pid

# Worker 2
cd /srv/partnr-worker-2
mkdir -p logs
nohup python -u scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs.yaml \
  --task-ids Task_4_Cnt --output-dir results/worker_2 \
  > logs/worker_2.log 2>&1 &
echo $! > logs/worker_2.pid
```

Start with two workers and measure RAM, GPU memory, CPU use, and API rate limits
before increasing concurrency. On a multi-GPU server you can prefix each `nohup`
command with `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`, respectively;
verify Habitat's EGL device selection during the smoke test. A single GPU may
support multiple workers if its memory permits. Separate copies do not isolate
GPU resources or API quotas.

If the server requires Slurm, submit each worker command through your site's
scheduler with the appropriate resources and worker working directory, rather
than starting background processes on a login node. Partition/account/resource
values depend on the cluster.

## Monitor and retrieve results

```bash
tail -f /srv/partnr-worker-1/logs/worker_1.log
nvidia-smi
```

Each worker writes its own experiment summary and per-run logs. The wrapper's
success count means the subprocess completed; use the planner's stats/traces to
assess task success. Real reruns replace matching run directories; use new output
paths to preserve earlier experiments. Each subprocess has a 30-minute timeout.

From your local machine, copy results into distinct destinations:

```bash
rsync -av USER@SERVER:/srv/partnr-worker-1/results/ results/server_worker_1/
rsync -av USER@SERVER:/srv/partnr-worker-2/results/ results/server_worker_2/
```

No evaluations are launched by following only the transfer and installation
sections; the smoke-test and worker launch commands execute the planner and can
make API requests.
