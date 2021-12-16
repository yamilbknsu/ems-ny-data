# NYC Simulator for online relocation of ambulances
![Visualization](/img/JSViz_2.png)

## Files in this repository
### Simulation model
* SimulatorBasics: General simulation framework. This file is the heart of the simulation model and is written to easily extend to any simulation paradigm. Further usage and references of this framework can be found [here](https://github.com/yamilbknsu/simulator/).
* Models: Base structural classes of the ambulance relocation model. `EMSModel` is the class extending `SimulatorBasics.Simulator` that contains the global simulation state as well as references to all the other objects in the simulation. This is the starting and ending point of the model. Additionally, `Vehicle` and `Emergency` classes are defined and used across all the simulation.
* SimulationParameters: Contains the `SimulationParameters` class, which encapsulates all the necessary parameters to characterize a setting of the simulation model, including optimization and simulation parameters. In order to execute the model, an instance of this class has to be created and passed along in the constructor of the `EMSModel` class.
* Events: Here are all the events that determine the behavior of the simulation. Each class extends from `SimulatorBasics.Event` and implements an `execute` method that containts the logic to be executed at the specified simulation time.
* Solvers:
  * `OnlineSolvers`: Implementation of the online Relocation, Redeployment and dispatching optimization procedures. Each class in this file extends `RelocationModel` and overrides the behavior. An instance of these classes is passed as a parameter of the model.
  * `ROASolver`: Implementation of the model found on [Enayati et.al (2018)](http://dx.doi.org/10.1016/j.omega.2017.08.001). Just like the previous file, the classes on this script extend `RelocationModel` and override the behavior.
* Generators: Arrival generator classes are defined. If you want to define your own arrival process, it should be included here and extend the class `ArrivalGenerator`.

### Experiment execution

The usage of these files is detailed in the [Usage](#usage) section.
* runOneExperiment: *Manually* execute one instance of the model with hard-coded parameters. Used for feature testing and debugging.
* experimentBuilder: Generate a list of experiments to be executed according to a grid os parameters. The description of each experiment is stored as a dictionary inside the list exported to `experimentsConfig.pickle` by default. This is useful for parallel computing since we can now execute one experiment on each machine and index each one by the position in this list.
* runExperiments: Load the `experimentsConfig.pickle` file and execute one or several replicas of the model. Can be controlled with script input arguments.

### Experimental
* AsyncTesting: Start a server for real-time visualization. *Debugging state*.

## Usage
### Data
Input parameters for the simulation and optimization process must be included in the `Preprocessing Values` folder at the root of the project. Inside this folder, a separate folder for each set of values has to be created with all the `.pickle` files inside. The parameter set used in our study was denominanted `HRDemand` and can be downloaded [here](https://drive.google.com/file/d/1M5g94heUBSk_RVSU-byNW18SiGvUsV7R/view?usp=sharing).

Example of folder structure:
```
Preprocessing Values/
    HRDemand/
        candidate_nodes.pickle
        demand_nodes.pickle
        ...
    SmallExperiment/
        candidate_nodes.pickle
        demand_nodes.pickle
        ...
runExperiments.py
Models.py
...
```

You can use whatever folder and file structure suits you best by changing the parameter loading portion of the experiment execution files.
### Python Environment
The model was last tested using Python `3.6.13` with:
```
geopandas       0.9.0
numpy           1.19.2
pandas          1.1.5
python-igraph   0.7.1.post7
gurobi          9.1.2
```
### Experiments
Experiments execution is designed to run in a system with several parallel machines. For this reason, the intended way to run the simulation experiments is to first define the different configurations to be tested as different tasks and then deliver them to a scheduling system that will take care of allocating them on the available machines.

The `experimentBuilder.py` file loops through all the possible values for the parameters that are going to be tested generating a grid of configurations that is then transformed into a list where each element is a dictionary containing the parameter values for each configuration, including the replication number of the same configuration. This list is then exported as `exprimentsConfig.pickle`.
Then, the `runExperiments.py` file is setup to read this list from the binary file and execute the experiments. For this, two arguments can be passed along with the execution command, namely `-i` indicating the index on the list of the first experiment to run and `-n` indicating the number of experiments to run in this queue.

Then, the ideal way to run this simulation is to generate one task per replication, with `-n 1` and indexed by the task_id of the scheduling system, and let the scheduler assign each replication as machines become available.
## License