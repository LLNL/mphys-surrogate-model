This folder has the results for Optuna trials for the box dataset. Each sub folder is split by
model type, and contains a "true" run of the model with the parameters
found in the optuna run. Optuna is trying to minimize the mean wasserstein 
distance for the entire training set. Current tests are with box data.

We can get decent results with a second degree polynomial for AE-SINDy,
see AE-SINDy_2025-07-11T12/08/26_d89a4eb0fb4945028106fcf78349108c folder.
While the results aren't there, I think they're similar to a third degree
polynomial.

Using the results found in the above tuning, a separate optuna tune was 
done to find the right values for the Champion et al. "lambda1_metaweight",
batch size, and learning rate. These results are in the
AE-SINDy_2025-07-13T21/06/35_e5f19b9991f3446f80daca8cf52f3612 folder. While
the overall results are worse, they're subjectively pretty close. I think 
this shows that one can do a second order polynomial SINDy and still do a 
decent job. I think it is still worth playing with regularization.

More consistent Optuna trials are in the ERF dataset folder