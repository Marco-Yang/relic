# ReLIC: Reinforcement Learning for Interlimb Coordination

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)


This repository contains the official implementation for the paper "Versatile Loco-Manipulation through Flexible Interlimb Coordination", accepted to CoRL 2025.

[**Project Website**](https://relic-locoman.rai-inst.com/) | [**arXiv Paper**](https://arxiv.org/abs/2506.07876)

---
*Our ReLIC policy enables a quadruped robot to walk with three legs and manipulate with the arm and one leg.*

## Overview

**Reinforcement Learning for Interlimb Coordination (ReLIC)** is an approach that enables versatile loco-manipulation through flexible interlimb coordination. The core of our method is a single, adaptive controller that learns to seamlessly coordinate all of a robot's limbs. It intelligently bridges the execution of precise manipulation motions with the generation of stable, dynamic gaits, allowing the robot to interact with its environment in a versatile and robust manner.

## Install
This project uses [Pixi](https://pixi.sh/latest/installation/) to manage dependencies and ensure a consistent development environment. You won't need to use `conda` or `virtualenv` separately or manually install `IsaacSim` or `IsaacLab`. Just follow the steps below to get started.

1. **Install Pixi**
    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

2. **Clone the repository**
    ```bash
    git clone https://github.com/bdaiinstitute/relic.git && cd relic
    ```

3. **Install dependencies**
    ```bash
    pixi install
    ```

4. **Activate the environment**
    ```bash
    pixi shell
    ```

Alternatively, you can install the project without Pixi by following the standard installation guides for [IsaacLab](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/isaaclab_pip_installation.html) and its [extensions](https://github.com/isaac-sim/IsaacLabExtensionTemplate/tree/main?tab=readme-ov-file#installation).

## Training
