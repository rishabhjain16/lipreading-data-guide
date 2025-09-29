### Edited from https://github.com/nguyenvulebinh/AV-HuBERT-S2S to convert Fairseq AV-Hubert models to HF format
#Not tested


import torch
import os
from transformers import Speech2TextConfig
from src.model.avhubert2text import AV2TextForConditionalGeneration

# Path to the original Fairseq checkpoint
FAIRSEQ_CKPT_PATH = "fairseq_model.pt"

# Output directory to save the converted Hugging Face model
OUTPUT_DIR = "converted-avhubert-model"

# Mapping from Fairseq-style keys to Hugging Face-style keys
MAPPING_FAIRSEQ_TO_HF = {
    "pos_conv.0": "pos_conv_embed.conv",
    "self_attn.k_proj": "attention.k_proj",
    "self_attn.v_proj": "attention.v_proj",
    "self_attn.q_proj": "attention.q_proj",
    "self_attn.out_proj": "attention.out_proj",
    "self_attn_layer_norm": "layer_norm",
    "fc1": "feed_forward.intermediate_dense",
    "fc2": "feed_forward.output_dense",
    "final_layer_norm": "final_layer_norm",
}

def rename_keys(state_dict, mapping):
    renamed_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for fairseq_key, hf_key in mapping.items():
            if fairseq_key in key:
                new_key = key.replace(fairseq_key, hf_key)
        renamed_dict[new_key] = value
    return renamed_dict

def convert_fairseq_to_hf(fairseq_ckpt_path, output_dir):
    # Load Fairseq checkpoint
    fairseq_ckpt = torch.load(fairseq_ckpt_path, map_location="cpu")
    model_weights = fairseq_ckpt['model']

    # Extract encoder weights
    encoder_weights = {
        key.replace("encoder.w2v_model.", ""): value
        for key, value in model_weights.items()
        if "encoder.w2v_model" in key
    }

    # Extract decoder weights
    decoder_weights = {
        key.replace("decoder.", ""): value
        for key, value in model_weights.items()
        if key.startswith("decoder.")
    }

    # Rename keys to Hugging Face format
    encoder_weights = rename_keys(encoder_weights, MAPPING_FAIRSEQ_TO_HF)
    decoder_weights = rename_keys(decoder_weights, MAPPING_FAIRSEQ_TO_HF)

    # Load into Hugging Face model
    config = Speech2TextConfig()
    model = AV2TextForConditionalGeneration(config)

    model.encoder.load_state_dict(encoder_weights)
    model.decoder.load_state_dict(decoder_weights)

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    print(f"Model successfully converted and saved to '{output_dir}'.")

# Run the conversion
convert_fairseq_to_hf(FAIRSEQ_CKPT_PATH, OUTPUT_DIR)
