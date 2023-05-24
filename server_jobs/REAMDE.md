## Server jobs
Here you can find the bash scripts to run training and evaluation on SLURM.

### Launching a job
To schedule a job use 
```sbatch train_speaker.sh```
The bash file will take care of everything.

### Parallel Array jobs
Most of the script support the `--array` option for parallel computing.
Each id (in range [1,6]) is mapped to a domain as follows:
- 1: appliances
- 2: food
- 3: indoor
- 4: outdoor
- 5: vehicles
- 6: all

array job can be launched with 
```sbatch --array 1-6  train_listener.sh```

or if you want to specify a specific domain use 
```sbatch --array=0,2,3 eval_simulator.sh```

### Args
Depending on what you want to run there are various args that you can set. 
Check out the [Param](../src/commons/Params.py) class for available configurations.