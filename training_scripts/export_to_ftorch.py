import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from eval_models_testdata import get_model

if __name__ == "__main__":
    ae_sindy = get_model("SINDy")
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

