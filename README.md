# Description

A repsitory for conducting multi-agent reinforcement learning experiments. 

# Getting Started

## Requirements
- Python 3.10.11
- Conda 23.7.2
  
## Instructions
(Tested using Ubuntu 20.04)
 1. Create the Conda environment from the `environment.yml` file. This will install all the necessary python dependencies on your system.
    ```
    conda env create -f environment.yml
    ```
 2. If you desire to run experiments using the SUMO traffic simulator, install SUMO and set the necessary environment variables (the example below sets the `SUMO_HOME` variable in your `~/.bashrc` file).
    ```
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-doc

    echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
    source ~/.bashrc
    ```
 3. Activate the environment.
    ```
    conda activate mqop
    ```
 4. It is possible to run experiments using command line arguments, but to make it easier to define experiments and maintain tracability for data processing, configurations are defined using configuration files. These files are then passed to the desired module via command line. For example, to run the actor-critic single objective learning algorithm:
    ```
    python ac-indpendent-SUMO.py -c experiments/sumo-4x4-dqn-independent.config
    ```
    NOTE: some files contain the "SUMO" suffix. This indicates that this file was speicifally updated to support using the `sumo-rl` environment but it should still be compatible with other PettingZoo environments as well. Likewise, it should also be possible to use the `sumo-rl` environment with a module that does not contain the "SUMO" suffix. For example, to run the independent DQN with parameter sharing module using a `sumo-rl` configured experiment:
     ```
    python dqn-indpendent-ps.py -c experiments/sumo-2x2-ac-independent-ps.config
    ```
     In the future, the "SUMO" suffix will be removed from file names.
 6. The experiment results are logged in the `csv` directory and named using various elements defined in the configuration file. Additionally, the results can be viewed with TensorBoard. (Please [install tensorboard](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard) to use this feature).
    ```
    tensorboard --logdir runs/
    ```

# Configuring an Experiment
Currently, all pre-defined experiment files are configured to utilize the `sumo-rl` environment but this can be changed by leveraging the `gym-id` configuration parameter.
If you'd like to create your own experiment using a different environment and set of hyperparameters, we recommend using an existing file as a template. See [here](https://github.com/HIRO-group/marl-experiments/tree/main/experiments) for examples of experiment configurations.

## Configuration Parameter Definitions
Coming soon...

- max-cycles: TODO: should be set to sumo-seconds when it exists, this variable tells the learning algorithm when to reset during training, it is not used for analysis purposes so online rollouts will run for all sumo-seconds even if max-cycles is less 

# List of Implemented Algorithms
Coming soon...

# Environments

## PettingZoo

This repository primarily conducts experiments on environments from the PettingZoo library. Environments can be specified through configuration arguments. Specific environment configuration parameters are also configurable.

## SUMO

[sumo-RL](https://github.com/LucasAlegre/sumo-rl) is a Python module that wraps the [SUMO traffic simulator](https://www.eclipse.org/sumo/) in a PettingZoo-style API. This allows for a SUMO simulation to be created in the same style as any PettingZoo library. 

### Understanding the Reward Structure of the SUMO Environment

The SUMO environment uses a default reward structure defined [here](https://github.com/LucasAlegre/sumo-rl#rewards). Maximizing the *change* in total weighting times of all vehicles at a given step may not be the most intuitive reward function. While the method of defining the "best" state of traffic may be up for debate, most experiments in this project use the `queue` reward function which returns the negative number of all vehicles stopped in the environment at the current step. Stopped vehicles are generally bad for traffic flow so the agents attempt to minimize this value. The [wiki](https://github.com/HIRO-group/marl-experiments/wiki/SUMO-and-SUMO%E2%80%90RL-Notes) has additional notes regarding the sumo-rl environment.

# Acknowledgments

We would like to thank the contributors of [CleanRL](https://github.com/vwxyzjn/cleanrl) for the base DQN implementation. We would also like to thank the contributors of [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [sumo-RL](https://github.com/LucasAlegre/sumo-rl) for maintaining MARL environments.
