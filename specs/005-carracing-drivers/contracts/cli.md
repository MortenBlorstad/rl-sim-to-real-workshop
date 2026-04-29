# Contract: CarRacing CLI Surfaces

The two CarRacing drivers expose CLI flags that mirror the Pendulum drivers, plus two SB3-only HuggingFace flags. This contract is the single source of truth for argument names, defaults, validation, and error messages.

## `workshop-1/3-car-racing/train.py` (custom PPO)

### Flags

| Flag           | Type   | Default        | Notes                                                        |
|----------------|--------|----------------|--------------------------------------------------------------|
| `--timesteps`  | int    | `200_000`      | Total environment steps to train.                            |
| `--run-name`   | string | auto-timestamp | Used as the run-directory name. Auto-generated if omitted.   |
| `--no-eval`    | flag   | false          | Skip the post-training evaluation video.                     |
| `--force`      | flag   | false          | Overwrite an existing run directory (otherwise fails fast).  |

No `--seed` flag (matches Pendulum, per feature 003 Q2). The seed is configured in source via `hyperparameters["random_state"]`.

No HuggingFace flags on `train.py` — the custom PPO does not load from HF in this feature.

### Vector env construction (in source, not user-overridable)

```python
NUM_ENVS = 4
env = gym.make_vec(
    "CarRacing-v3",
    num_envs=NUM_ENVS,
    vectorization_mode="sync",
    wrappers=[GrayscaleObservation, lambda e: ResizeObservation(e, (84, 84)), lambda e: FrameStackObservation(e, 4)],
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
```

(`ResizeObservation` and `FrameStackObservation` need extra args, so the lambdas adapt them to the `wrappers=[Callable[[Env], Env], …]` signature `make_vec` expects.)

## `workshop-1/3-car-racing/train_sb3.py` (SB3 escape hatch)

### Flags

| Flag             | Type   | Default        | Notes                                                                                          |
|------------------|--------|----------------|------------------------------------------------------------------------------------------------|
| `--timesteps`    | int    | `200_000`      | Total environment steps to train (or fine-tune, when `--hf-repo` is set).                       |
| `--seed`         | int    | `42`           | RNG seed (SB3 driver keeps a `--seed` flag, matching `train_sb3.py` for Pendulum).             |
| `--run-name`     | string | auto-timestamp | Used as the run-directory name.                                                                |
| `--no-eval`      | flag   | false          | Skip the post-training evaluation video.                                                       |
| `--force`        | flag   | false          | Overwrite an existing run directory.                                                            |
| `--hf-repo`      | string | unset          | HuggingFace repo identifier (e.g. `sb3/ppo-CarRacing-v0`). When set, fine-tune from this checkpoint. |
| `--hf-filename`  | string | auto-derived   | Filename within the HF repo. Auto-derived as `<basename(repo_id)>.zip` when omitted.            |

### Auto-derived `--hf-filename` rule

```python
def _default_hf_filename(repo_id: str) -> str:
    return f"{repo_id.split('/', 1)[-1]}.zip"
```

Examples:
- `sb3/ppo-CarRacing-v0` → `ppo-CarRacing-v0.zip`
- `username/my-model` → `my-model.zip`
- `single-segment` → `single-segment.zip`

### Mutual-exclusion rules

- `--hf-filename` MAY appear without `--hf-repo` is **NOT** allowed; the driver raises `argparse` error: `--hf-filename requires --hf-repo`.
- All other flags compose freely; `--hf-repo X --timesteps 50000` means "fine-tune for 50k steps from the HF checkpoint".

### `--hf-repo` semantics (per spec clarification Q1, locked in)

When `--hf-repo` is set:

1. Resolve filename: `args.hf_filename` if provided, else `_default_hf_filename(args.hf_repo)`.
2. Download via `huggingface_hub.hf_hub_download(repo_id=args.hf_repo, filename=filename)`. Cache hit returns instantly; cache miss downloads and caches.
3. Load via `stable_baselines3.PPO.load(local_path, env=env, device=device)`. The loaded model carries the original training hyperparameters from the HF checkpoint; the local driver does NOT override them.
4. Fine-tune via `model.learn(total_timesteps=args.timesteps, callback=Sb3JsonlCallback(runlog), progress_bar=True)`.
5. Evaluate (unless `--no-eval`) via the same code path as a from-scratch run.

`meta.json` records `hf_repo_id` and `hf_filename`; both keys are present on every SB3 run, set to `null` for from-scratch.

### Error messages (contract-mandated wording shape)

#### `RunDirectoryExistsError` (both drivers)

```
Error: run directory '<path>' already exists.
Pick a different --run-name or pass --force to overwrite.
```

(Existing wording from `RunLogger`; this feature does not change it.)

#### Missing `swig` system dependency (both drivers, when `gymnasium[box2d]` import fails)

```
Error: CarRacing requires the Box2D physics engine, which depends on the
'swig' system tool. Install it with:
  - macOS:   brew install swig
  - Linux:   sudo apt-get install swig
  - Windows: choco install swig
Then re-run `uv sync --group workshop1`.
See workshop-1/3-car-racing/README.md for details.
```

#### HuggingFace download failure (`train_sb3.py` only)

The driver catches the union {`LocalEntryNotFoundError`, `RepositoryNotFoundError`, `EntryNotFoundError`, `requests.ConnectionError`, `OSError`} and raises a project-level `HuggingFaceLoadError(RuntimeError)`:

Offline + cache miss:
```
Error: could not download '<filename>' from HuggingFace repo '<repo_id>'.
You appear to be offline and the file is not in the local cache (~/.cache/huggingface/).
Either connect to the internet, or use a local pretrained artefact instead:
  uv run python workshop-1/3-car-racing/train_sb3.py [...] (without --hf-repo)
  ... and load from pretrained/ as documented in the README.
```

Repo not found:
```
Error: HuggingFace repo '<repo_id>' does not exist (or is private and you are
not authenticated). Check the spelling, or pick a public CarRacing repo such
as 'sb3/ppo-CarRacing-v0'.
```

File not found in repo:
```
Error: HuggingFace repo '<repo_id>' exists but does not contain a file named
'<filename>'. Use --hf-filename to specify the correct filename, or check the
repo's file list at https://huggingface.co/<repo_id>/tree/main.
```

#### Architecture mismatch between HF checkpoint and local config (`train_sb3.py` only, per spec edge case)

When `PPO.load(local_path, env=env)` raises a state-dict mismatch (typically `RuntimeError: Error(s) in loading state_dict for PPO`):

```
Error: HuggingFace checkpoint '<repo_id>/<filename>' was trained with a
network architecture or hyperparameter set incompatible with the local
SB3 PPO defaults. Underlying error:
  <SB3 error message>
Pick a different repo (e.g. 'sb3/ppo-CarRacing-v0' uses the standard
CnnPolicy defaults), or override locally to match the checkpoint.
```

## Examples

### Custom PPO from-scratch smoke

```bash
uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name smoke --force
```

### SB3 from-scratch smoke

```bash
uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke --force
```

### SB3 fine-tune from HuggingFace

```bash
uv run python workshop-1/3-car-racing/train_sb3.py \
    --hf-repo sb3/ppo-CarRacing-v0 \
    --timesteps 10000 \
    --run-name finetune \
    --force
```

### Override HF filename for non-conventional repos

```bash
uv run python workshop-1/3-car-racing/train_sb3.py \
    --hf-repo my-user/my-model \
    --hf-filename custom_name.zip \
    --timesteps 10000 --run-name finetune --force
```

### CPU-only

```bash
RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name cpu --force
```

(Same env-var override applies to both drivers, per feature 004's policy.)
