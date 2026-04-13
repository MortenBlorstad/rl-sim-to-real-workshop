# Workshop 2 — From Sim to Car (3 hours)

## Agenda

| Time | Block | What you do |
|------|-------|-------------|
| 0:00–0:20 | Intro: Sim-to-Real | Follow along with the lecture |
| 0:20–0:40 | Simulator setup | Run `0-simulator/first_drive.py` |
| 0:40–1:30 | Train agent in sim | Work on `1-train-sim/` |
| 1:30–1:45 | Break | |
| 1:45–2:00 | Intro: Real car | PiRacer Pro demo |
| 2:00–2:45 | Sim-to-Real transfer | Deploy model on car, test on track |
| 2:45–3:00 | Wrap-up + race | |

## Before you start

### 1. Download the DonkeyCar simulator

Go to [gym-donkeycar releases](https://github.com/tawnkramer/gym-donkeycar/releases) and download the right version for your OS:
- **Linux:** `DonkeySimLinux.zip`
- **Mac:** `DonkeySimMac.zip`
- **Windows:** `DonkeySimWin.zip`

Unpack it and note the path to the `donkey_sim` file.

**Linux:** Make it executable:
```bash
chmod +x DonkeySimLinux/donkey_sim.x86_64
```

### 2. Install dependencies

```bash
uv sync --group workshop2
```

### 3. Set the simulator path

Either edit `config.py` or set an environment variable:
```bash
export DONKEY_SIM_PATH="/path/to/donkey_sim"
```

## Step 0: First drive in the simulator

```bash
uv run python 0-simulator/first_drive.py
```

You should see the simulator start and the car drive straight ahead. Press Ctrl+C to stop.

## Step 1: Train an agent in the simulator

Pick an approach:

**Option A — End-to-end CNN (simpler):**
```bash
uv run python 1-train-sim/train_cnn.py
```

**Option B — VAE + PPO (better for sim-to-real):**
```bash
uv run python 1-train-sim/collect_data.py        # collect images
uv run python 1-train-sim/train_vae.py            # train the VAE
uv run python 1-train-sim/train_vae_ppo.py        # train PPO on the latent space
```

See `1-train-sim/config.py` for all hyperparameters.

## Step 2: Deploy on the real car

Once you have a trained model:

```bash
# Export the model
uv run python 2-real-car/export_model.py --model <path-to-model>

# Copy it to the Pi
bash 2-real-car/deploy.sh <pi-ip-address>
```

Follow the workshop leader's instructions for driving on the track.

## Stuck?

Pretrained models live in `pretrained/` — use them to keep moving.
