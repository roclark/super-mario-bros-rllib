# Super Mario Bros. RLLib
A reinforcement learning project designed to learn and complete the original
Super Mario Bros. for the Nintendo Entertainment System using various models
from the RLLib library from Ray.

## Setting up the repository

### Creating a virtual environment
After cloning the repository, it is highly recommended to install a virtual
environment (such as `virtualenv`) or Anaconda to isolate the dependencies of
this project with other system dependencies.

To install `virtualenv`, simply run

```
pip install virtualenv
```

Once installed, a new virtual environment can be created by running

```
virtualenv env
```

This will create a virtual environment in the `env` directory in the current
working directory. To change the location and/or name of the environment
directory, change `env` to the desired path in the command above.

To enter the virtual environment, run

```
source env/bin/activate
```

You should see `(env)` at the beginning of the terminal prompt, indicating the
environment is active. Again, replace `env` with your desired directory name.

To get out of the environment, simply run

```
deactivate
```

### Installing Dependencies
While the virtual environment is active, install the required dependencies by
running

```
pip -r requirements.txt
```

This will install all of the dependencies at specific versions to ensure they
are compatible with one another.

## Training a model

To train a model, use the `train.py` script and specify any parameters that need
to be changed, such as the environment or epsilon decay factors. A list of the
default values for every parameters can be found by running

```
python train.py --help
```

If you desire to run with the default settings, execute the script directly with

```
python train.py
```

The script will train the default environment over a set number of episodes and
display the training progress after the conclusion of every episode. The updates
indicate the episode number, the reward for the current episode, the best reward
the model has achieved so far, a rolling average of the previous 100 episode
rewards, and the current value for epsilon.
