This folder has the results for Optuna trials for the condensation dataset. 
Each sub folder is split by model type, and contains a "true" run of the 
model with the parameters found in the optuna run. Optuna is trying to 
minimize the RMSE for the entire training set. More info to come.

AE-SINDy_Poly_Latent_1 contains the very first Optuna tune for the
AESindyThermo model. More hand tuning is probably needed, but this and
the next run will just show what happens by default. The full epoch
run is in erfCond_FFNN_latent2_order2_tr1000_lr9.184941089801632e-05_bs14_weights1.0-3426975.0-342697504.0_3426975.0
in the same folder and the performance isn't amazing.

AE-SINDy_Lambda1MW_Lambda4_1 contains the Optuna tune using the 
latent dim and poly order found in AE-SINDy_Poly_Latent_1, and varies
the lambda1_metaweight and lambda4, which is the weight on loss_weight_sindy_S.
The full epoch run is in erfCond_FFNN_latent2_order2_tr1000_lr0.00012466702443702146_bs109_weights1.0-6882013.0-688201280.0_1.9763186450207038
in the same folder, and the performance isn't amazing.

At this point, I think hand tuning is required. One idea is to skip
Champion's recommendation for weights and just optimize over all of them.
The good news is the shape of the dx distributions are similar even if the values
aren't.