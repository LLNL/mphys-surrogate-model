# NOTES

# SINDy model
All the notes in this section are for the Polynomial Order 2 model, in the folder "Poly2 Model"

## Best model progression
1. Emily Baseline. Couldn't do it at the time, but pulled code for commit d5e161577a7f9fbb0dd7ff453818c1c6b67c2b5c and 
   reran after commenting new code out of models.py
2. ReduceLROnPlateau. It seems the scheduler helps with the convergence, goes from about 8.08 to 8.25
3. Optuna Tuned. Quite good! Very close to AE-AR model, on par with black box model. 
   Random seed is 1. Need to see random seed variability.

----------
# Black box dzdt
All in this section are for the black box model (`BB`), in the folder "Black Box Model"

## Learning rate tests
Emily baseline is 5e-4, probably too high
* 2025-05-24T18/26/07_914ec1f8f257470a980dbd50b91ff9dd: 1e-5, probably too low
* 2025-05-24T19/27/55_abd31c1d3a204e6397943b437d719803: 9e-4, too high, spike at end
* 2025-05-24T19/30/45_f1e7ba5c404b47709692ab72c6b8e62e: 9e-5, okay, slow but no spike at end
* 2025-05-24T19/33/10_54fcdda6fa4b4fb2aef0189ec37a387e: 2.5e-4, probably good for hand tuning will stick for now

## Lambda specification
To calculate lamba1 ratio as specified in supplementary materials
```
xx = np.squeeze(train_data.x)  # Remove extra dim
dx = np.squeeze(train_data.dx)  
xxl2 = np.linalg.norm(xx, ord=2, axis=1)**2  # Axis may not be necessary but too lazy to check
dxl2 = np.linalg.norm(dx, ord=2, axis=1)**2
lambda1 = xxl2.sum()/dxl2.sum()
```
This seems to work well! Still worth tuning but showed good improvements.

## Best model progression
1. Weight Initialization. Still has bad ending but saving best model.
2. Emily Baseline
3. Champion Lambda. Has better learning rate and Champion et al. recs for lambdas are good. 
   Currently, not implementing regularization loss. Should we?
   Getting close to autoregressive model with this one
   Random seed is 1
4. Optuna Tuned. Getting close to AE-AR model

# Optuna Tests
1. Sindy_BB: Optuna BB test
2. Sindy_Poly2: Order 2 polynomial test
3. Sindy_Poly2_Bugfix: I don't remember exactly what the bugfix was here, I think it had something to do with the lambda2
   specification. This is the one to look at, but I'll try to duplicate the result