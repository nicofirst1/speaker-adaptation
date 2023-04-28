
## 28/04/2023
I'm probably late in writing this journal but better late than never. 
I've been working on how to adapt the uva project with the RL part. During this time i have done the following:

### 1. RL dataloader
I came up with a kind of rl env to finetune models. It is actually a dataloader, but the function is similar to an env. 
Basically i take all the images coming from a specific domain, randomly choose 6 and one target, no utteraces of any kind. 
This is called the EcDataset.

### 2. Speaker RL finetuning
With this i managed to succesfully implement Reinforce for finetuning the speaker with the listener. After many attempts, 
I managed to make it work and now the speaker is able to generate utterances that are actually useful for the listener.
However i noticed that this was not helping the adaptation part, since the simulator was not trained to handle the new speaker embeddings. 

### 3. Simulator finetuning
This is a huge point. Basically i started with retraining the simulator with the new speaker and listener, but it was too artificial.
The whole thing could not be ported in an env where the listener is a human since you need retraining everytime. 
So i had an idea, what if i just finetune the simulator with the new speaker embeddings?
This led me to discover a big flow of the uva project, namely the fact that the simulator is not plug and play. Indeed it is
trained everytime with the speak embeddings and the listener utterances togheter! This means that if the speaker changes you have to retrain the simulator all over, 
rendering it effectively non plug and play.

#### 3.1 Simulator new architecture
So, to address the problem i had to come up with a different architecture for the simulator. The first idea was to make it possible
to train the sim without the embeddings (for listener training) and later with both during speaker finetuning.
This lead to the sim becoming very good at predicting the target, but not the listener behavior. 
I noticed that i used the same layers for both utterance and embedding streams in the sim, so i splitted them, still no improvements. 
Then i tried to freeze some of the layers, and, long story short, freezing all the sim layers except the ones dealing with the embeddings was the winning choice.

### Results
I am now comparing, with the adaptive evaluation script, the results of the old sim with the new one. Preliminarly, with domain food, I 
get an imporvement (adapted utt vs generated) of 16% for the old method and 15% for the new one (which is great news!).

### Next steps
The next steps are to clean the code and make it more readable. Then i will try with another domain and see if the results are consistent.
Next I will go back to the domain food and do some hyperparameter tuning to see if i can get even better results than the old method. 
I need to be carefull to compare the architectures fairly (same model size), even tho, for now it seems that the new sim is better while being also smaller.

Finally i want to confirm that, with a different speaker distribution, i'm still able to do adaptation with a finetuned sim.
Optionally, after all this shit, I will see if i can train both the speaker and the sim together. 


