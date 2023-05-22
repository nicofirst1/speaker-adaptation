# 28/04/2023

I'm probably late in writing this journal but better late than never.
I've been working on how to adapt the uva project with the RL part. During this time i have done the following:

### 1. RL dataloader

I came up with a kind of rl env to finetune models. It is actually a dataloader, but the function is similar to an env.
Basically i take all the images coming from a specific domain, randomly choose 6 and one target, no utteraces of any
kind.
This is called the EcDataset.

### 2. Speaker RL finetuning

With this i managed to succesfully implement Reinforce for finetuning the speaker with the listener. After many
attempts,
I managed to make it work and now the speaker is able to generate utterances that are actually useful for the listener.
However i noticed that this was not helping the adaptation part, since the simulator was not trained to handle the new
speaker embeddings.

### 3. Simulator finetuning

This is a huge point. Basically i started with retraining the simulator with the new speaker and listener, but it was
too artificial.
The whole thing could not be ported in an env where the listener is a human since you need retraining everytime.
So i had an idea, what if i just finetune the simulator with the new speaker embeddings?
This led me to discover a big flow of the uva project, namely the fact that the simulator is not plug and play. Indeed
it is
trained everytime with the speak embeddings and the listener utterances togheter! This means that if the speaker changes
you have to retrain the simulator all over,
rendering it effectively non plug and play.

#### 3.1 Simulator new architecture

So, to address the problem i had to come up with a different architecture for the simulator. The first idea was to make
it possible
to train the sim without the embeddings (for listener training) and later with both during speaker finetuning.
This lead to the sim becoming very good at predicting the target, but not the listener behavior.
I noticed that i used the same layers for both utterance and embedding streams in the sim, so i splitted them, still no
improvements.
Then i tried to freeze some of the layers, and, long story short, freezing all the sim layers except the ones dealing
with the embeddings was the winning choice.

### Results

I am now comparing, with the adaptive evaluation script, the results of the old sim with the new one. Preliminarly, with
domain food, I
get an imporvement (adapted utt vs generated) of 16% for the old method and 15% for the new one (which is great news!).

### Next steps

The next steps are to clean the code and make it more readable. Then i will try with another domain and see if the
results are consistent.
Next I will go back to the domain food and do some hyperparameter tuning to see if i can get even better results than
the old method.
I need to be carefull to compare the architectures fairly (same model size), even tho, for now it seems that the new sim
is better while being also smaller.

Finally i want to confirm that, with a different speaker distribution, i'm still able to do adaptation with a finetuned
sim.
Optionally, after all this shit, I will see if i can train both the speaker and the sim together.

# 01/05/2023

New month, same shit.
I manage to clean all the relevant code and also implement a multi thread adaptation pipeline that is now x2 faster than
the previous one (thank chatGPT).
Now the problem seems to be that the code from last time does not really give the same resutls...
First of all, the adapted accuracy for the old model are 10% more than on the old code... i don't know why, but i
checked and everything seems to be correct...
Then, using the updated simulator i get very low adaptation accuracy, I checked and it seems to be tied to the problem
of correctly predicitng
the listener behavior. It mught be the case that i need to go back to the ec finetuining.
Well shit... I just ran an adaptation with a simulator only trained on predicting the list (sim pretraining) and it is
achieving comparable results to the
original sim from the uva project. This basically means that the embedding stream does not even need to be trained to
work...
Good news for the problem inconsistency in the uva project i spotted last time, but bad news for ec pretrain. I want to
see if i can get the same
results woth a finetuned version of the speaker.

# 02/05/2023

here we go again. Of course I tried using a different domain and now it does not work anymore. This whole thing is a
rollercoaster of emotions.
The problem seems to be related to the sim not predicting well enough the listener behavior. I'm now trying with the old
sim to see if the adaptation pipeline is the problem. Ok, the old sim works, so the problem is the new sim. Again the
problem seems to be
related to the sim ability to predict the listener. I'm pretty sure there most be some bug in the data usage. There are
two possible problems:

- The sim is training with a listener that is not the one used for adaptation (different domains) -> nope.
- The sim is training with the right listener but on a different split (seen? unseen?) or (all? train? test?).

I finally understood why... I was using the old simulator for testing... I'm so stupid.

# 03/05/2023

Ok, I've been trying to see what the problem is. I've trying plugging in one of the pretrained models in the sim
pretrain pipeline to see if the loading is the problem but no. I also tried to see the accuracy on test, and it is the
same basically as the one on eval. So the problem may tied on how i estimate the accuracy.

Ok, I basically regenerated all the data for sim pretraining, with the new speaker. Even tho the weights are always the
same, I tought the different architecture can be the reason, even tho i'm not really sure about it. Plus, i'm using a
smaller model size (64 vs 1024). Even with 64 there is a bit overfit (75 vs 95 on train vs test). I'm now trying with 32
to see if it is better.
Still a big overfit (73 vs 93). I will try with 16 now, never heard of a model so small, at leas now training is faster
lol.
Still overfit (70 vs 82), at least is a little bit better. I will try with 8 now, but i'm not really sure it will work.
In any case i will probably go back to 32 and use some dropout. So with 8 the overfit is much better but that bc the
whole accuracy went down quite a bit (66 vs 74). I will now try 64 with dropout and increased batch size (32 ->64). Much
better, at least on the overfit side (75 vs 80), i think i'll keep it like this, now let's go back to the original
problem.

I finally spotted the error... There is no error, simply the double stream architecture fucks up the sim ability to
predict the listener when i also use the embeddings. At least there is no real bug in the code, but i need to find a way
to make it work.
Let's see if adding a temp to the embedding vector helps.

It worksssss, yas!!
I need to test on other domains tho, cross fingers!
It works also with other domains!

Now, for the final test. I will finetune a speaker with a listener and then use that with this new sim to see if i can
get even higher accuracies. If i can then I'm done!

# 04/05/2023

Ok, today is more of a bonus day. The aim is to have the finetuned speaker show higher golden acc improvement than the
normal one.
I tried, but it seems like the accuracy is not changed, not even the generated one, which is weird. The finetuned
speaker, aka ec_speaker, reaches an accuracy of 50% with the listener on the ec dataset, but on the normal dataset it
stays around 24%.
I can understand that the distributions between the datasets are different, but this is still weird. Plus, there is one
thing i need to check, even tho the seed, speaker and list are the same, the generated accuracy for all the runs varies
sligthly, I'm not sure why, but i'll start with this.

Dataset and model outputs are the same for different runs... Everything is the same, maybe I misread the accuracy.
I think I found the culript here, the multithreading. It is likely that the threads are working differently depending on
the amount of computation the pc is doing a the time, so the order of the batches is not the same. I'll try to run two
consecutive runs with one thread and check if they are the same and how much does it vary from the multithread version.
In general the std between different runs that are supposed to be the same is 0.5%, which i believe is negligible.
Indeed, the multithread is the problem, but again, it can be ignored.

Anyway, the problem is still there, the ec_speaker does not improve the golden accuracy.  
I tried evaluating the new speaker and compare it with the old (on test... i know it's illegal, but it was fast). The
ec_speaker seems to perform better... I'm kind of puzzled here. I will try with a better evaluation pipeline, but it is
still very weird.

So, testing with the val split resulted in 24% vs 23% fine-tuned vs normal. I don't understand, might it be because the
distribution of images is much different?
Maybe i should take into account that the distribution tends to have images from the same domain. Maybe that is also why
the list is so good (40% ood) right from the start.

I think i found the problem, during finetuning i forgot to specify the data domain, so i was finetuning on data only
coming from the same domain. I will try again now.

# 05/05/2023

Now the finetuning is using the right data, but the results are not encouraging. The speaker is not improving the
listener accuracy. I will try with some experiments while doing my german exercises, let's see.

# 08/05/2023

It looks like the eval results oscillate between comopletely random (16%) and twice the normal accuracy (30%). I'm not,
sure why this is...
The data does not change during the epochs, I will check the domain accuracies, maybe everytime there is a focus on one
domain only? No, a peak does not correlate to one domain increase only, usually it correlates to multiple actually.

Well well well... I implemented a way to check the difference between acc before train and after train, and it seems
that the whole thing does not work. I have no idea why, i suspect the loss function is wrong but i don't know how to fix
it tho. I think i will mostly focus on the other part for now.

# 10/05/2023

So, I was trying to debug and noticed that with lr = 0 the loss is not constant... Actually it seems to show a periodic
pattern with period 10 (epochs). I have no idea why this is but maybe i should check data.
I think I might have found the problem, nucleus sampling is a stochastic process, so it is always slightly different
even
when i give the same inputs. Plus i set the seed to pic always the same images every 10 epochs, thus the
periodicity. Very cool debugging round, but i havent still solved the problem. Should i start with making the sampling
deterministic perhaps?

# 15/05/2023

I found a weird error. The nucleus sampling (or more generally the hypothesis generation) act differently when the
batchsize changes...
This code has more holes than a scolapasta.
No wait, of course the batchsize has an effect, the whole backprop changes with different batchsizes... I'm an idiot.

# 16/05/2023

So, it worked. I managed to get an improved accuracy of 15% on the listener. I tested this model with the simulator and
the results are mixed.
On one hand the original (speaker generated) accuracy is much higher, which of course means a decreased improvement when
coupled with the sim (from 8.5% to -0.2%). On the other hand, the golden improvement is higher (from 5% to 6%). overall
the sim seems to have difficulties predicting the list behavior with the finetuned speaker utterances since they diverse
very much form the original ones. Indeed i have a drop in sim-list-acc from 71% to 45%. This would require a retraining
of the sim but it is useless sine i would be training on jibberish. I think this part should be considered done and just
a comparison with the sim. In theory i should add a kl regularization to the speaker to make it more similar to the
original distribution.
PFfffff, let's try uff.
Soooo, I think i managed to implement the kl loss correclty. I'm not super sure. Anyhow, the speak finetuned with this
additional loss has a 10% improvement on list accuracy, which is not bad. However the adaptive pipeline has both
original and golden improv negative. Which is super weird since it means that the speak distribution has changed more
than before...

# 18/05/2023

Ok, i've been focusing on this for way too long. Even tho i managed to finetune the speaker, the results with the
simulator are not what i want. It is time to move on.
I think i should really focus more on the online adaptation, since it seems to be the most important point of concern
for the reviewers.
I am not really sure how to tackle it, since it involves learning to predict the listener behavior on the fly.
Is it imitation learning? I don't think so, since i don't have a dataset of the listener behavior.
I need to figure out how to smartly use the setting i have, let's do some brainstorming.

So, i have a setting with a domain specific listener is interacting with a speaker. Since the listener is domain
specific there should be some kind of accuracy loss when interacting, which is what i want to address with the sim. In
my case the sim is untrained, it starts as a randomly initialized nn. I have already shown how the split architecture
does not need any kind of training on the embedding side, if this holds true, it means that i will start with a
completely uninitialized sim. I bet this will not really work, but it can be my starting point. What i was thinking
about is the fact that i have control over the speaker generation. So if i teach the sim to change the speaker
generationg part then i can use the backprop to learn the direction of improvement for the listener. This is probably
easier to learn at the same time how to modify the sim embeddings and how to predic the listener. So, step 2 should be
having a sim that is very good at changing the speaker caption is a particular direction and then pair it with a
listener and do some kind of meta learning. It reminds me a bit of adversarial attack (maybe i just have a fixation on
this), but hear me out: if i have a sim that is good at changing the speaker caption in a particular direction, then i
can randomly change and observe the list behavior. If the list behavior is good, then i can reinforce the sim, if it is
bad, then i can punish it. This is a bit like the adversarial attack, but instead of having a fixed target, i have a
target that is changing over time. I think this is a good idea, i will try to implement it. (omg the last two sentences
are generated with code completion and they sound exactly as i would have written them, this is scary).

Ok so to recap:

1. build a pipeline where frozen speak and list are interacting and the sim is trying to both predict the list and to
   change the speak caption in a particular direction.
2. Pretrain the sim to stear the speak caption in a particular direction. and then pair it with the listener as said
   above.

I can do it!

# 19/05/2023

Ok, so i managed to implement the first step (even tho i should debug it more) and test it with the fast adaptation
script. Good news: it is better than an untrained model (adapted acc 31 vs 40), but it is worse than the old mode (40 vs
47). I mean, it is a good new tho, it is working while being trained in an online fashion, so nothing to worry about.
I'm unsure how to proceed tho. On one hand, I could keep this online training and just add a rl pretraining of the sim
as listed yesterday. On the other hand i can try to train both at the same time as i did month ago. When i tried it back
in the days it didnt' work since the architecture was different (shared weights), but now it is not the case anymore and
it might be interesting to check if it works. Before that i should maybe debug the whole thing first. I mean, if step 2
doesn't work at least i addressed the issues in the original paper.
Note to self, changing the embed temp deos not change anything.

Before going on with step 2 i wanted to try something in the middle. Right now i'm training the sim to predict the list
but not to stir h0. I just want to see if that works or not. 