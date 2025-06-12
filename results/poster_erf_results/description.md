**NOTE:**
For all models, the final model state is saved and plotted, rather than the "BEST" model.


# SINDy
`FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pkl`
```aiignore
params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "poly_order": 3,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    # "lambda3_sparsity": 0.0, TODO: sequential thresholding
    "CNN": False,
    "print_frequency": 1,
}

torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])

output_directory = "../results/poster_erf_results/ae_sindy"
test_ids = [0, 10, 20, 30]
tplt = [0, 5, -1]
```

# Neural-network dz/dt
`FFNN_latent3_layers(40, 40, 40)_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_7850dd2260f34875baf55452f4311d60.pkl`

```
params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    "layer_size": (40, 40, 40),
    "CNN": False,
    "print_frequency": 1,
}

torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])

output_directory = "../results/poster_erf_results/ae_nndzdt"
test_ids = [0, 10, 20, 30]
tplt = [0, 5, -1]
```

# Autoregressor
`erf_FFNN_latent3_order(10, 20, 10)_tr100_lr0.001_bs128_weights1-1_f74ecd55b87a43e2bfde2bf9973fdf10.pth`
```aiignore
params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "n_lag": 1,
    "w_recon": 1,
    "w_dx": 1,
    "w_dz": 1,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "layer_size": (10, 20, 10),
    "CNN": False,
    "print_frequency": 1,
}
```