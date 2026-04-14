# Deployable Autonomous RL Racer (Dockerized)

![DDPG Architecture](https://img.shields.io/badge/Algorithm-DDPG-blue) ![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange) ![Deployment](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED)

## 📌 Project Overview

This repository demonstrates the application of **Deep Reinforcement Learning (DRL)**—specifically the **Deep Deterministic Policy Gradient (DDPG)** algorithm—to an autonomous vehicle control problem.

This module is packaged as an **isolated, reproducible container** using Docker and Docker Compose, supporting **GPU acceleration**.

## 🚀 How to Run with Docker Compose

Ensure you have Docker and Docker Compose installed (with NVIDIA Container Toolkit configured on your Linux/WSL2 host if you wish to use GPU acceleration).

### 1. Build and Run Standard Training
To start a container that will train the agent and map the output weights straight to your host machine's `weights/` directory:

```bash
docker-compose up --build train
```
*The script `improved_ddpg.py` will execute, live-streaming the training metrics to your terminal.* 

### 2. Run Hyperparameter Tuning
To execute systematic Bayesian Optimization on the neural network parameters:

```bash
docker-compose run tune
```
*This will execute `tune_ddpg.py` and output `best_params_f1.txt` directly to your local `weights/` folder.*

## 📂 Project Structure
- `src/`: Contains your logic (`improved_ddpg.py`, `tune_ddpg.py`) and the `gym_gmmcar` environment. Mounted dynamically so you can edit code locally without rebuilding the image!
- `weights/`: Houses your trained Actor-Critic `.h5` files. Mounted directly into the container so your progress is saved outside the container lifecycle.
- `docker-compose.yml`: Defines the `train` and `tune` configurations.
- `Dockerfile`: Prepares a Python environment on top of `tensorflow/tensorflow:latest-gpu`.


## Inference
<video width="320" height="240" controls>
  <source src="src/inference/rltour.mp4" type="video/mp4">
</video>

## Conclusion
Using bayesian optimization in this context is not very relevant, this algorithm DDPG generate too much noise, so all the effort to imrpove the Agent were put on the reward shaping.
