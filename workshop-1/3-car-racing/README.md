# Stage 3 — CarRacing-v3

Tren en bilkjøre-agent som ser pikselbilder fra spillet og lærer å holde seg på banen.

## Forutsetninger

CarRacing bruker Box2D-fysikkmotoren. Den krever at `swig` er installert på systemet:

```bash
brew install swig            # macOS
sudo apt-get install swig    # Linux
choco install swig           # Windows
```

Deretter:

```bash
uv sync --group workshop1
```

## Trening fra null med custom PPO

For deg som har fullført PPO-TODO-ene i stage 1. Bruker din `PPOAgent` med en CNN-arkitektur (auto-detektert fra observasjonsformen `(4, 84, 84)`):

```bash
uv run python workshop-1/3-car-racing/train.py --timesteps 200000 --run-name my-run
```

Skriver til `runs/car-racing/my-run/{meta.json, metrics.jsonl, model.pt, eval.mp4}`.

## Trening fra null med Stable-Baselines3

Den konstitusjonelle nødløsningen — bruker SB3 sin `PPO("CnnPolicy", ...)`:

```bash
uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 200000 --run-name my-run-sb3
```

Skriver `model.zip` istedenfor `model.pt`, men ellers samme katalogstruktur som over.

## Fortsett trening fra HuggingFace-modell

Ikke nok tid til å trene fra null? Last ned en ferdig-trent CarRacing-modell fra HuggingFace Hub og finetune fra den vektene:

```bash
uv run python workshop-1/3-car-racing/train_sb3.py \
    --hf-repo <bruker>/<repo-id> \
    --timesteps 10000 \
    --run-name finetune --force
```

Driveren laster ned modellen, initialiserer SB3 sin PPO med disse vektene, og fortsetter trening i `--timesteps` steg før evaluering. Andre kjøring med samme `--hf-repo` bruker den lokale cachen (`~/.cache/huggingface/`) — ingen nedlasting, < 1 sekund.

`meta.json` får feltene `hf_repo_id` og `hf_filename` så du kan se i ettertid hvilken modell finetuneringen startet fra.

**Merk:** ikke alle HuggingFace-repoer fungerer. Modellen må være lagret med SB3 ≥ 2.0 (gymnasium, ikke gamle `gym`). Hvis du ser feilen `ModuleNotFoundError: No module named 'gym'`, prøv et nyere repo eller fall tilbake til trening fra null.

## Override av enhet

Som i stage 2 — sett miljøvariabelen `RL_WORKSHOP_DEVICE` for å velge maskinvare:

```bash
RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/3-car-racing/train.py ...
RL_WORKSHOP_DEVICE=mps uv run python workshop-1/3-car-racing/train.py ...   # default på Apple Silicon
RL_WORKSHOP_DEVICE=cuda uv run python workshop-1/3-car-racing/train.py ...
```

På CarRacing-CNN er MPS faktisk raskere enn CPU (omtrent 2× på en M-serie Mac), så standardvalget `auto` er bra.

## Hyperparametere

`train.py` bruker en CNN-tilpasset variant av Pendulum sine hyperparametere:

| Param            | Verdi      | Notat                                              |
|------------------|------------|----------------------------------------------------|
| `rollout_size`   | 2048       | total over alle env (4 env × 512 per env)           |
| `n_epochs`       | 4          | Færre enn Pendulum siden CNN-oppdateringer er tyngre |
| `batch_size`     | 64         | minibatch                                          |
| `lr`             | 3e-4       | standard PPO                                       |
| `gamma`          | 0.99       | lengre horisonter for kjøring                      |
| `log_std_init`   | -0.5       | strammere utforskning (gass/brems er asymmetrisk)   |

`train_sb3.py` bruker SB3 sine `PPO("CnnPolicy")`-defaults.
