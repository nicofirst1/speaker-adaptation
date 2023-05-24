## Evals

This dir contain the python files to evaluate pretrained models. 
Each script logs the result on wandb

### Speaker-listener domains
Evaluate the performances of a pretrained general speaker with a pretrained domain-specific listener.

### Adaptive speaker
The final evaluation pipeline, where speaker, listener and simulator are evaluated.
For each data-sample, the speaker generate a hypothesis that is influenced by the simulator until the
listener correctly predicts the target or a max number of iterations (_s_iter_) is reached.

This script also load a csv file to wandb for the analysis carried out in the [analysis dir](../analysis).
