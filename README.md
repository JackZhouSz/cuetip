# CueTip 🎱

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-green" alt="Status: Active">
  <img src="https://img.shields.io/badge/Language-Python-blue" alt="Language: Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT">
</p>

## Overview

CueTip is a library for natural language interaction and planning with a pool simulation. It enables language models to interact with [PoolTool](https://github.com/ekiefl/pooltool), a fully 3D highly-accurate physics simulation of pool/billiards.

See our paper [here](https://arxiv.org/abs/2501.18291) for more details and come see us at SIGGRAPH 2025!

### Key Features

- **Natural Language Processing**: Language models interpret the results of shots through the emission of *natural language events*
- **Planning & Optimization**: Models describe target shots through a list of similar NL events, then black box optimization tunes shot parameters so that the outcome matches the target
- **Physics Simulation**: Integration with PoolTool for accurate physics-based shot outcomes
- **Neural Surrogate**: Fast neural surrogate model of SOTA pool agents for efficient optimization
- **Expert Grounding**: Optimisation utilises expert knowledge to enable grounded explanations of shots

## Installation

### Prerequisites

- Python 3.10+
- Language model API key (e.g. OpenAI) or local LM

### Setup

1. Install `uv` package manager:

```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment:

```bash
uv venv
```

3. Install required packages:

```bash
uv sync --all-extras --frozen
```

4. **Note**: PoolTool may require a font that is not installed by default. You can install it by running:

```bash
wget -O HackNerdFontMono-Regular.ttf https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/Hack/Regular/HackNerdFontMono-Regular.ttf
mv HackNerdFontMono-Regular.ttf .venv/lib/python3.10/site-packages/pooltool/ani/fonts/
```

## Usage

### Training

Before running the examples, train the neural surrogate model:

```bash
uv run train_neural_surrogate.py
```

This model helps optimize shot parameters efficiently during simulation.


### Verify Installation

To verify your installation is working correctly:

```bash
uv run test_shot_optimisation.py
```

this will run a simple shot optimisation example to ensure everything is working correctly.

### Example Agents

#### Function Agent

Run the function agent example which utilizes the trained neural surrogate to optimize shots:

```bash
uv run example_function_agent.py
```

#### Language Model Agent

Run the language model agent example where an LLM plans potential shots and the neural surrogate selects the best outcome:

```bash
uv run example_llm_agent.py
```

> **Note**: The LLM agent requires an OpenAI API key or a compatible local LLM setup in your environment variables. The DSPy library is used for LM inference and supports many backends.

## Docker Support

For containerized usage with GPU support:

```bash
# Build the Docker image
docker build -t cuetip:main .

# Run with NVIDIA GPU support
docker run --gpus all -it cuetip:main bash
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  <i>Built with ♥ for pool players and AI enthusiasts</i>
</p>