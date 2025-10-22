"""
Export pytorch weights to fortran compatible matrices with ftorch
"""

import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch

from eval_models_testdata import get_model

if __name__ == "__main__":
    ae_sindy = get_model("SINDy")
    model_dir = Path(
        "../results/Optuna/ERF Dataset/AE-SINDy_LimParams/erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc"
    )
    model_files = list(model_dir.glob(f"*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found")
    ae_sindy.load_state_dict(torch.load(model_files[0], weights_only=True))

    ae_sindy.eval()

    enc = ae_sindy.encoder
    dec = ae_sindy.decoder
    deriv = ae_sindy.dzdt

    # example input: 64 bins
    example_x = torch.randn(64)
    example_z = torch.randn(4)
    example_l = torch.randn(3)

    traced_encoder = torch.jit.trace(enc, example_x)
    traced_encoder.save("../data/ftorch_weights/encoder_model.pt")
    traced_decoder = torch.jit.trace(dec, example_l)
    traced_decoder.save("../data/ftorch_weights/decoder_model.pt")
    traced_dzdt = torch.jit.trace(deriv, example_z)
    traced_dzdt.save("../data/ftorch_weights/dzdt_model.pt")
