# Description

A repsitory for conducting multi-agent reinforcement learning experiments. 

# Instructions

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
 4. Specify the number of agents, the seed, and the GPU ID when running the scripts. Each script is self-contained. An example is provided.
    ```
    python dqn-indpendent-ps.py --gym-id mpe.simple_spread_v2 --N 5 --seed 5 --gpu-id 0 --learning-starts 10000
    ```
    Currently, only one file (`dqn-independent-SUMO.py`) supports the use of a configuration file.
    ```
    python dqn-indpendent-SUMO.py -c experiments/sumo-4x4-dqn-independent.config
    ```
 5. The experiment results are logged in the `csv` directory. Alternatively, the results can be viewed with TensorBoard. (Please install tensorboard in other environment to use the feature).
    ```
    tensorboard --logdir runs/
    ```

# Configuring an Experiment

See [here](https://github.com/HIRO-group/marl-experiments/tree/main/experiments) for examples of experiment configurations.

# Environments

## PettingZoo

This repository primarily conducts experiments on environments from the PettingZoo library. Environments can be specified through configuration arguments. Specific environment configuration parameters are also configurable.

## SUMO

[sumo-RL](https://github.com/LucasAlegre/sumo-rl) is a Python module that wraps the [SUMO traffic simulator](https://www.eclipse.org/sumo/) in a PettingZoo-style API. This allows for a SUMO simulation to be created in the same style as any PettingZoo library. 

### Understanding the Reward Structure of the SUMO Environment

The SUMO environment uses a default reward structure defined [here](https://github.com/LucasAlegre/sumo-rl#rewards). Maximizing the *change* in total weighting times of all vehicles at a given step may not be the most intuitive reward function. While the method of defining the "best" state of traffic may be up for debate, most experiments in this project use the `queue` reward function which returns the negative number of all vehicles stopped in the environment at the current step. Stopped vehicles are generally bad for traffic flow so the agents attempt to minimize this value.

# Acknowledgments

We would like to thank the contributors of [CleanRL](https://github.com/vwxyzjn/cleanrl) for the base DQN implementation. We would also like to thank the contributors of [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [sumo-RL](https://github.com/LucasAlegre/sumo-rl) for maintaining MARL environments.
