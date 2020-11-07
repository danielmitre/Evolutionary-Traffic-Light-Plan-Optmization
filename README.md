First of all, you'll need to install [SUMO](https://sumo.dlr.de/docs/Installing.html).

Then, install our dependencies with pip: `pip install -r requirements.txt` (you may want to do this in a python virtual environment).

Now you're good to run the programs.

## genetic.py

This program will run the genetic algorithm and the simulations. It requires two arguments:
* **dir_path**: the directory to dump the chromossomes, the tls programs, the stdout of the SUMO execution and the tripinfo of the best result.
* **metric_name**: the genetic algorithm currently supports **three** objetive functions:
  * **delay**: the objective is to minimize the average delay experienced by a vehicle in the network.
  * **normal_delay**: the objective is to minimize the average standardized route delay in the network.
  * **max_normal_delay**: the objective is to minimize the maximum standardized route delay in the network.