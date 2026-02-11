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