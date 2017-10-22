# Training
This project should contain all our training code for the model car project. For instructions on
contributing, please see CONTRIBUTING.md

# Workflow
Explanation of directory structure:

```
training
|- configs      > configuration files to easily manage training/validation hyperparameters
|- logs         > log files for debugging
|- nets         > all pytorch neural network models stored here
|- preprocess   > scripts for preprocessing video data into h5py files
|- save         > default save location for all nets
```

If you wish to run a new experiment, please add your `config.json` file into the `configs` folder. See `configs/CONFIGS.md` for
a detailed explanation of how to structure your configuration file. To run your experiment, simply use the command line 
command `bdd-docker python Train.py --config <config filepath>`.

If you wish to add a new network, please add your network to the `nets` folder. Set a variable `Net` to point to your class
so the training script can automatically find your network.


# Standards
This is a version of the repository that follows PEP8 guidelines, please comment on GitHub code, 
file an issue, or correct any errors with a pull request.
