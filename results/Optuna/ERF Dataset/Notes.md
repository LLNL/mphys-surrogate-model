# Notes

The ERF dataset is the more "canonical" dataset, and attempts were made
to make the Optuna tests more rigorous. The following parameters were 
varied for each model.

## AE-AR
```
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)
    layer1_size = trial.suggest_int("layer1_size", 20, 180)
    layer2_size = trial.suggest_int("layer2_size", 20, 180)
    layer3_size = trial.suggest_int("layer3_size", 20, 180)
    w_dx = trial.suggest_float("w_dx", 0.1, 1.9)
    w_dz = trial.suggest_float("w_dz", 0.1, 1.9)
```

## AE-NNdzdt
```
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)
    layer1_size = trial.suggest_int("layer1_size", 20, 60)
    layer2_size = trial.suggest_int("layer2_size", 20, 60)
    layer3_size = trial.suggest_int("layer3_size", 20, 60)
    lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
```


## AE-SINDy
```
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)
    poly_order = trial.suggest_int("poly_order", 2, 3)
    lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
```