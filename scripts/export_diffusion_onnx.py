#!/usr/bin/env python3
"""Export a Stable Diffusion pipeline to ONNX.

Exports each component (text_encoder, unet, vae_decoder) as a separate
ONNX file under the output directory.

Usage:
    python export_diffusion_onnx.py <model_id> <output_dir>

Example:
    python export_diffusion_onnx.py runwayml/stable-diffusion-v1-5 models/sd-v1-5
"""

import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import torch
from diffusers import StableDiffusionPipeline


def export_text_encoder(pipe, output_dir):
    te_dir = os.path.join(output_dir, "text_encoder")
    os.makedirs(te_dir, exist_ok=True)
    te_path = os.path.join(te_dir, "model.onnx")
    print(f"  text_encoder -> {te_path}", file=sys.stderr)

    dummy_input_ids = torch.ones(1, 77, dtype=torch.long)
    pipe.text_encoder.config.return_dict = False
    with torch.no_grad():
        torch.onnx.export(
            pipe.text_encoder,
            (dummy_input_ids,),
            te_path,
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=18,
        )


def export_unet(pipe, output_dir):
    unet_dir = os.path.join(output_dir, "unet")
    os.makedirs(unet_dir, exist_ok=True)
    unet_path = os.path.join(unet_dir, "model.onnx")
    print(f"  unet -> {unet_path}", file=sys.stderr)

    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([1.0])
    encoder_hidden_states = torch.randn(1, 77, 768)
    pipe.unet.config.return_dict = False
    with torch.no_grad():
        torch.onnx.export(
            pipe.unet,
            (sample, timestep, encoder_hidden_states),
            unet_path,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            opset_version=18,
        )


def export_vae_decoder(pipe, output_dir):
    vae_dir = os.path.join(output_dir, "vae_decoder")
    os.makedirs(vae_dir, exist_ok=True)
    vae_path = os.path.join(vae_dir, "model.onnx")
    print(f"  vae_decoder -> {vae_path}", file=sys.stderr)

    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            return self.vae.decode(latent_sample).sample

    latent = torch.randn(1, 4, 64, 64)
    decoder = VaeDecoder(pipe.vae)
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (latent,),
            vae_path,
            input_names=["latent_sample"],
            output_names=["sample"],
            opset_version=18,
        )


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_id> <output_dir>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_id}...", file=sys.stderr)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")

    print("Exporting components:", file=sys.stderr)
    export_text_encoder(pipe, output_dir)
    export_unet(pipe, output_dir)
    export_vae_decoder(pipe, output_dir)

    # Save tokenizer alongside
    try:
        pipe.tokenizer.save_pretrained(output_dir)
        print(f"  tokenizer -> {output_dir}", file=sys.stderr)
    except Exception as e:
        print(f"  Warning: could not save tokenizer: {e}", file=sys.stderr)

    print(f"Done. Components exported to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
