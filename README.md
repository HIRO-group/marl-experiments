# Description


# Instructions

 1. Create the Conda environment from the `environment.yml` file.
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

# Acknowledgments

We would like to thank the contributors of [CleanRL](https://github.com/vwxyzjn/cleanrl) for the base DQN implementation. We would also like to thank the contributors of [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [sumo-RL](https://github.com/LucasAlegre/sumo-rl) for maintaining MARL environments.
