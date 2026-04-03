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
* `Baseline5` Claude had several suggestions to improve performance. I didn't want to
              do a separate test for each suggestion, so the notes are here combined
              into one test.  
              1. *Kaiming*: Recommended to switch from Xavier to Kaiming initialization
              because ReLUs are used. Not a huge change but makes it a little less
              "fuzzy". Will keep the change.  
              2. *Repeated Calculations*: The lines of code
              `pred_x_recon = model.decoder(model.encoder(batch_x)); z = model.encoder(batch_x)`
              run the encoder twice, and those gradients are accumulated, and the graph
              recieves twice the gradient, messing with the weights and doubling
              computation/memory. The code is switched to 
              `z = model.encoder(batch_x); pred_x_recon = model.decoder(z)`. Also
              doesn't produce a huge change, which is in some ways heartening. Still,
              the change will be kept.  
              3. *Remove detach*: The old code had the line `zz = z.clone().detach().requires_grad_()`,
              and the `detach()` term meant that `loss_dx` couldn't send gradients
              backward, so the encoder wasn't actually trained by the `loss_dx`
              term. This doesn't improve performance, but shortens the sawtooth
              loss region significantly, which is promising.  
              Claude also suggested playing with the loss weights, but that will
              be saved for the next test.

Testing on the AE-SINDy model has ended as of April 3, 2026. Domain knowledge suggests
that AE-SINDy isn't a very good model for this task.