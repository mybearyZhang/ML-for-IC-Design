import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch

import bcq
import utils
import wandb


# Trains BCQ offline
def train_BCQ(replay_buffer, num_actions, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = bcq.discrete_BCQ(
        num_actions,
        device,
        args.BCQ_threshold,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"]
    )

    # Load replay buffer	
    replay_buffer.load('/root/ML-for-IC-Design/data/buffer2/replay_buffer_all.pkl')
    # replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0

    # Initialize wandb
    wandb.init(project="ML_BCQ")

    # Training loop
    while training_iters < args.max_timesteps:
        # for _ in range(int(parameters["eval_freq"])):
        #     policy.train(replay_buffer)
        # evaluations.append(eval_policy(policy, args.env, args.seed))
        # np.save(f"./results/BCQ_{setting}", evaluations)
        # training_iters += int(parameters["eval_freq"])

        for _ in range(int(parameters["save_freq"])):
            policy.train(replay_buffer)

        training_iters += int(parameters["save_freq"])
        print(f"Training iterations: {training_iters}")
        
        policy.save(f"./models/BCQ_{setting}_{training_iters}")

        # Log metrics to wandb
        wandb.log({"Training Iterations": training_iters})

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":

    parameters = {
        # Exploration
        "start_timesteps": 1e3,
        # "initial_eps": 0.1,
        # "end_eps": 0.1,
        # "eps_decay_period": 1,
        "initial_eps": 1,
		"end_eps": 5e-2,
		"eps_decay_period": 6e4,
        # Evaluation
        "eval_freq": 5e3,
        "eval_eps": 0,
        "save_freq": 1e4,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 64,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4
        },
        # "optimizer_parameters": {
        #     "lr": 1e-3
        # },
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="IC-Design")     # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
    parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
    args = parser.parse_args()
    
    print("---------------------------------------")	
    print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(parameters["batch_size"], parameters["buffer_size"], device)

    num_actions = 7
    train_BCQ(replay_buffer, num_actions, device, args, parameters)