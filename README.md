# Traffic Signal Optimization Using Deep Reinforcement Learning

A complete implementation of a Deep Q-Network (DQN) agent that learns to control traffic lights at a 4-way intersection in SUMO (Simulation of Urban Mobility). The project demonstrates real reinforcement learning with step-by-step agent-environment interaction and compares the learned policy against a fixed-time baseline controller.

## 🎯 Project Goals

This project trains a DQN agent to:

- **Minimize average vehicle waiting time** at the intersection
- **Reduce queue lengths** on all approaches
- **Increase throughput** (vehicles completed per hour)
- **Learn adaptive control** that responds to traffic conditions

The agent learns through trial and error, receiving negative rewards for long waiting times and queues, gradually discovering effective traffic light control strategies.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Complete RL Implementation**: Full DQN with experience replay and target networks
- **SUMO Integration**: Realistic traffic simulation with TraCI control
- **Modular Architecture**: Clean, well-documented Python code
- **Configurable Traffic Patterns**: Balanced, NS-heavy, or EW-heavy traffic
- **Comprehensive Metrics**: Waiting time, queue length, throughput tracking
- **Visualization**: Training curves and comparison plots
- **Baseline Comparison**: Fixed-time controller for performance evaluation

## 📁 Project Structure

```
trafficRL/
├── traffic_rl/                 # Main package
│   ├── env/                    # Environment components
│   │   ├── sumo_env.py        # SUMO RL environment wrapper
│   │   └── route_generator.py # Traffic route generation
│   ├── dqn/                    # DQN implementation
│   │   ├── agent.py           # DQN agent with epsilon-greedy
│   │   ├── network.py         # Neural network architecture
│   │   └── replay_buffer.py   # Experience replay buffer
│   ├── config/                 # Configuration files
│   │   └── config.yaml        # Hyperparameters and settings
│   ├── utils/                  # Utility modules
│   │   ├── metrics.py         # Metrics computation
│   │   ├── plotting.py        # Visualization functions
│   │   └── logging_utils.py   # Logging utilities
│   └── sumo/                   # SUMO network files
│       ├── network.net.xml    # 4-way intersection network
│       ├── routes.rou.xml     # Vehicle routes (generated)
│       └── simulation.sumocfg # SUMO configuration
├── scripts/                    # Executable scripts
│   ├── train_dqn.py           # Train DQN agent
│   ├── evaluate_dqn.py        # Evaluate trained agent
│   ├── run_fixed_time_baseline.py  # Run baseline controller
│   └── compare_results.py     # Compare DQN vs baseline
├── logs/                       # Training logs (created at runtime)
├── models/                     # Saved models (created at runtime)
├── results/                    # Plots and metrics (created at runtime)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Requirements

### Software Dependencies

- **Python 3.10+**
- **SUMO 1.18+** (Simulation of Urban Mobility)
  - Download from: [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)
  - **Important**: Add SUMO to your system PATH

### Python Packages

All Python dependencies are listed in `requirements.txt`:

- PyTorch (deep learning)
- NumPy, Pandas (data processing)
- Matplotlib, Seaborn (visualization)
- TraCI, SUMOlib (SUMO interface)
- PyYAML, tqdm (utilities)

## 📦 Installation

### Step 1: Install SUMO

#### Windows
1. Download SUMO installer from [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)
2. Run installer and note installation directory
3. Add SUMO `bin` directory to PATH:
   - Example: `C:\Program Files (x86)\Eclipse\Sumo\bin`
4. Verify installation: `sumo --version`

#### Linux
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

#### macOS
```bash
brew install sumo
```

### Step 2: Install Python Dependencies

```bash
# Clone or navigate to project directory
cd trafficRL

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Check SUMO
sumo --version

# Check Python packages
python -c "import torch; import traci; print('All packages installed!')"
```

## 🚀 Quick Start

### 1. Train the DQN Agent

```bash
python scripts/train_dqn.py
```

This will:
- Generate traffic routes
- Train the DQN agent for 200 episodes (configurable)
- Save the best model to `models/dqn_best.pth`
- Log metrics to `logs/training_log.csv`
- Generate training curves in `results/`

**Training time**: ~30-60 minutes on a modern CPU (depends on configuration)

### 2. Evaluate the Trained Agent

```bash
python scripts/evaluate_dqn.py
```

This runs 10 evaluation episodes with the trained model using a greedy policy (no exploration).

### 3. Run the Fixed-Time Baseline

```bash
python scripts/run_fixed_time_baseline.py
```

This runs the same scenarios with a simple fixed-time controller (30s NS green, 30s EW green).

### 4. Compare Results

```bash
python scripts/compare_results.py
```

This generates comparison plots and prints improvement percentages.

## 📖 Usage

### Training with Custom Settings

```bash
# Train with SUMO GUI (to visualize)
python scripts/train_dqn.py --gui

# Train with specific random seed
python scripts/train_dqn.py --seed 42

# Train with custom config
python scripts/train_dqn.py --config my_config.yaml
```

### Evaluation Options

```bash
# Evaluate with GUI
python scripts/evaluate_dqn.py --gui

# Evaluate specific model checkpoint
python scripts/evaluate_dqn.py --model-path models/dqn_ep100.pth

# Evaluate with different seed
python scripts/evaluate_dqn.py --seed 123
```

### Modifying Traffic Patterns

Edit `traffic_rl/config/config.yaml`:

```yaml
traffic:
  distribution: "balanced"  # Options: balanced, ns_heavy, ew_heavy
  num_vehicles_per_episode: 1000  # Total vehicles
  episode_duration: 3600  # Seconds
```

### Adjusting DQN Hyperparameters

Edit `traffic_rl/config/config.yaml`:

```yaml
dqn:
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  batch_size: 64
  max_episodes: 200
```

## 🧠 How It Works

### State Representation

The agent observes a 9-dimensional state vector:

- **Queue lengths** (4 values): Number of stopped vehicles on each approach (N, S, E, W)
- **Waiting times** (4 values): Average waiting time per approach
- **Current phase** (1 value): Current traffic light phase (NS or EW green)

All values are normalized for stable learning.

### Action Space

The agent can choose between 2 actions:

- **Action 0**: Set North-South green, East-West red
- **Action 1**: Set East-West green, North-South red

The environment automatically handles yellow phases and enforces minimum green times.

### Reward Function

The agent receives a reward at each step:

```
reward = -0.5 × (total waiting time) 
         -0.3 × (total queue length)
         -0.2 × (phase change penalty)
```

**Goal**: Maximize cumulative reward (minimize waiting time and queues)

### DQN Algorithm

1. **Experience Replay**: Store transitions in a buffer, sample random batches for training
2. **Target Network**: Use a separate target network for stable Q-value estimation
3. **Epsilon-Greedy Exploration**: Start with random actions (ε=1.0), gradually become greedy (ε=0.05)
4. **Huber Loss**: Robust loss function for Q-value updates

### Training Process

```
For each episode:
    1. Reset environment with new traffic
    2. For each step:
        a. Select action (epsilon-greedy)
        b. Execute action in SUMO
        c. Observe reward and next state
        d. Store transition in replay buffer
        e. Sample batch and train network
        f. Update target network periodically
    3. Log metrics and save best model
```

## ⚙️ Configuration

All settings are in `traffic_rl/config/config.yaml`:

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sumo.gui` | Use SUMO GUI | `false` |
| `traffic.num_vehicles_per_episode` | Vehicles per episode | `1000` |
| `traffic.distribution` | Traffic pattern | `balanced` |
| `dqn.learning_rate` | Learning rate | `0.0001` |
| `dqn.gamma` | Discount factor | `0.99` |
| `dqn.epsilon_decay_steps` | Exploration decay | `50000` |
| `dqn.max_episodes` | Training episodes | `200` |
| `reward.waiting_time_weight` | Waiting time penalty | `-0.5` |

## 📊 Results

After training and evaluation, you'll find:

### Logs
- `logs/training_log.csv`: Episode-by-episode training metrics
- `logs/evaluation_log.csv`: Evaluation results
- `logs/training.log`: Detailed training logs

### Models
- `models/dqn_best.pth`: Best performing model
- `models/dqn_ep*.pth`: Periodic checkpoints

### Plots
- `results/training_curves.png`: Reward and loss over time
- `results/dqn_vs_baseline_comparison.png`: Performance comparison

### Expected Performance

With default settings, the DQN agent typically achieves:

- **20-40% reduction** in average waiting time vs fixed-time
- **15-30% reduction** in average queue length
- **10-20% increase** in throughput

*Note: Results vary based on traffic patterns and hyperparameters*

## 🔍 Troubleshooting

### SUMO Not Found

**Error**: `sumo: command not found` or `TraCI could not connect`

**Solution**:
- Verify SUMO is installed: `sumo --version`
- Check SUMO is in PATH
- On Windows, restart terminal after adding to PATH

### TraCI Connection Error

**Error**: `TraCI could not connect to SUMO`

**Solution**:
- Close any running SUMO instances
- Check if port 8813 is available
- Try running with `--gui` flag to see SUMO errors

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce `batch_size` in config.yaml
- Use CPU: Set `device: "cpu"` in code
- Reduce `buffer_size`

### Slow Training

**Issue**: Training takes too long

**Solutions**:
- Reduce `max_episodes` (try 50-100 for testing)
- Reduce `num_vehicles_per_episode`
- Reduce `episode_duration`
- Use GPU if available

### Poor Performance

**Issue**: Agent doesn't learn well

**Solutions**:
- Train longer (increase `max_episodes`)
- Adjust reward weights in config
- Try different `learning_rate` (0.0001 - 0.001)
- Ensure sufficient exploration (`epsilon_decay_steps`)

## 📚 Additional Information

### Understanding the Metrics

- **Average Waiting Time**: Mean time vehicles spend waiting (lower is better)
- **Average Queue Length**: Mean number of stopped vehicles (lower is better)
- **Throughput**: Vehicles completed per hour (higher is better)
- **Phase Changes**: Number of traffic light switches (too many = flickering)

### Extending the Project

Ideas for enhancements:

1. **Multi-intersection**: Extend to multiple coordinated intersections
2. **Advanced algorithms**: Try Double DQN, Dueling DQN, or PPO
3. **Real traffic data**: Use real-world traffic patterns
4. **Pedestrian crossings**: Add pedestrian phases
5. **Emergency vehicles**: Priority handling for emergency vehicles

### Citation

If you use this project for research, please cite:

```
@software{traffic_rl_dqn,
  title = {Traffic Signal Optimization Using Deep Reinforcement Learning},
  author = {Traffic RL Team},
  year = {2026},
  url = {https://github.com/yourusername/trafficRL}
}
```

## 📄 License

This project is provided for educational purposes. SUMO is licensed under EPL-2.0.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Learning! 🚦🤖**
