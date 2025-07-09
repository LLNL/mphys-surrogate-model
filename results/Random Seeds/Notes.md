This folder shows the results from the `random_seed_variation.py` script contained
in `ng_scripts`. The models are broken out by folder, and the results are in each folder.
These results show that the random seed choice doesn't matter a ton, but you still don't 
want to be unlucky and choose a bad seed. Some follow-on tests make it seems like the actual
seed may not matter a whole lot - the worst seed had better performance than the best seed!
But, the worst seed had a stranger convergence curve than the best seed.

It's a bit tangential, but there are four extra folders in Batch Size 128 in AE-AR.
The folders show different seeds, but "Best Seed 1000 Epochs" shows that maybe going for a long
time without learning rate decay or early stopping is important. There is also a bug
that Emily has discussed where the later epoch models don't perform well. Is this related?
I think in general, focusing too much on the seed isn't super important. Once you find
some good general parameters with Optuna, try a couple random seeds.

There are now seed results for all three models. I don't think it's worth putting
that much more time into this investigation, though the script will remain if necessary.