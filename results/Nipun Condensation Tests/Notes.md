# Notes
This folder contains results from attempts to improve the Condensation AE-SINDy model.
Since the verification plotting isn't complete yet, there's a chance that the "baseline"
will change as those functions get fleshed out. The following is a description of the 
folders and the changes made to them.

* `Baseline1` The performance of the model straight from Emily when allowed to train
              at <1000 epochs. This is the first "real" test of the model.
* `Baseline2` The performance of the model after some (but not all) of the diagnostic
              plots are made. Performance is still poor.
* `Baseline3` Performance of the model after Emily has implemented dx normalization.
              No tuning has happened, but performance increased due to the normalization.
              Likely the previous troubles were due to optimizer struggles. A previous
              autoencoder was not loaded.
* `Baseline4` Optuna showed very little sensitivity to the loss weights, which was
              surprising. However, this test was the first one which showed good
              autoencoder performance, specifically getting the bimodal distribution
              in run 10 of the reconstruction plot. Maybe better initialization
              is needed? It's strange that the shape of the distributions in
              the heatmap comparison plot is roughly correct, but the values aren't.