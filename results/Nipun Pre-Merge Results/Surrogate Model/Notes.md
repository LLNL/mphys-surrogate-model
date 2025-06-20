# NOTES

## Latent space tests
* Contained within `Latent Space Tests` folder. 
  * The 04-25 and 04-26 runs have random seeds 0, 1, 2, 3 in chronological order
  * Code has changed since making these plots
* Latent space is visualized using plotly
* See `*_latent_space.html` for examples (html as of now is not saved in git)
* When training just the autoencoder, it seems that the latent space is almost 2D, but not quite (manifold?)
* This seems to be true whether the loss function involves kl-divergence or wasserstein distance
* Makes kind of a fan with handle shape (see 2025-05-01) for best example
  * later time steps make the handle

## Loss function tests
* 2025-05-02T14/46/15_0dc2351309bf4851b1e027333890d13b: KL Divergence (Baseline)  
* 2025-05-02T14/58/03_9f69ddd8d89546d09b4c1c8eaa5a25d9: Wasserstein. `p=2, blur=0.05`  
* 2025-05-02T15/14/21_85453bc187514b67804ceb933ca38ede: Wasserstein. `p=1, blur=0.05`  
* 2025-05-02T15/35/16_093720e6408f4114b8183fce7da57e45: Wasserstein. `p=1, blur=0.01`  

# Best model progression
(Ordered from worst to best)
1. Loss Function Tests/2025-05-02T14/46/15_0dc2351309bf4851b1e027333890d13b
2. Optuna Baseline. Only slightly worse than baseline.
3. Hand Tuned Baseline (Made after making initialization changes, save best model, model improvements from SINDy paper).
   Has following params as of this commit:
   ```
   random_seed = 0
   num_epochs = 200
   batch_size = 10
   n_latent = 3
   n_lag = 1
   lr = 1e-3
   wd = 1e-3
   lr_sched = False
   do_early_stopping = True
   cnn_flag = False
   tol = 1e-8
   w_recon = 1
   w_dx = 1
   w_dz = 1
   loss_type = "kld"  # Should be "kld" or "wass"
   ```
4. ReduceLROnPlateau, implemented default scheduler. Minor improvements. Goes from 8.58 to 8.68 (sometimes 8.69)
   At this time the latent space does not have a manifold shape anymore and is a 3d "frond"

# Optuna Baseline_200
Results from the Optuna test done in Nipun's repo pre-merge.

# Seed100_Epoch100
Testing 100 different random seeds for 100 epochs each. Interesting behavior occurs where
models converge bimodally. They have similar performance but one set of models is definitely better than 
the other set.