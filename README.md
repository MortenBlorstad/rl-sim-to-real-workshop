# Reinforcement Learning Workshop: Sim-to-Real

Learn reinforcement learning from the ground up — from theory and simulators to a self-driving mini car.

**Workshop 1:** Intro to RL, implement PPO, train an agent in Gymnasium (MountainCar, CarRacing)
**Workshop 2:** DonkeyCar simulator, train an agent, deploy on a real PiRacer Pro

## Prerequisites

- Python experience (functions, classes, numpy)
- Basic understanding of neural networks (what a forward pass, loss, and backprop are)
- No prior RL experience required

## Quick Start

### 1. Install uv

**Mac / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify it works:
```bash
uv --version
```

### Install swig

**macOS**

```bash
brew install swig
```

**Linux**
```
sudo apt update
sudo apt install swig
```

**Windows**
```powershell
winget install SWIG.SWIG
```

```
swig -version
```

> If the command is not found after installation, restart your terminal.

On Windows, reopen the terminal before running `swig -version`. If it is not recognized, add SWIG to PATH. [Here](https://www.youtube.com/watch?v=9umV9jD6n80) is how to add to PATH with the GUI.


### 2. Clone the repo

```bash
git clone https://github.com/<your-username>/rl-sim-to-real-workshop.git
cd rl-sim-to-real-workshop
```

### 3. Install dependencies

**Workshop 1:**
```bash
uv sync --group workshop1
```

**Workshop 2:**
```bash
uv sync --group workshop2
```

### 4. Verify setup

```bash
uv run python workshop-1/0-warmup/cartpole_random.py
```

You should see a CartPole window with a pole falling left and right. If it works, you are ready.

## Repo structure

```
├── workshop-1/                  # Day 1: RL theory + PPO + Gymnasium
│   ├── 0-warmup/                # Verify that the env works
│   ├── 1-ppo/                   # Implement PPO via TODOs
│   ├── 2-cartpole/              # Apply PPO to CartPole
│   └── 3-car-racing/            # Apply PPO to CarRacing
│
├── workshop-2/                  # Day 2: DonkeyCar sim + real car
│   ├── 0-simulator/             # First drive in the DonkeyCar sim
│   ├── 1-train-sim/             # Train an agent in the simulator
│   └── 2-real-car/              # Deploy the model on a PiRacer Pro
│
├── pi-setup/                    # Raspberry Pi + PiRacer setup
├── pretrained/                  # Pretrained models (safety net)
├── notebooks/                   # Google Colab backup
└── slides/                      # Presentations
```

## Resources

- [Gymnasium docs](https://gymnasium.farama.org/)
- [Spinning Up in Deep RL — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [DonkeyCar docs](https://docs.donkeycar.com)
- [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar)
- [Learning to Drive in 5 Minutes (VAE+PPO)](https://github.com/araffin/learning-to-drive-in-5-minutes)
