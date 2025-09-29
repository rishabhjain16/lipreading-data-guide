# model_analysis_utils.py
# inspect_fairseq_checkpoint() – View top-level keys and sample weights from a Fairseq checkpoint.
# inspect_encoder_decoder_keys() – Count and preview encoder/decoder keys.
# compare_key_formats() – Compare Fairseq vs Hugging Face key formats.
# check_model_loading() – Diagnose missing/unexpected keys when loading weights.
# print_model_summary() – Print model architecture and total parameter count.



import torch

def inspect_fairseq_checkpoint(ckpt_path):
    """
    Inspect the top-level structure and sample model keys in a Fairseq checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Top-level checkpoint keys:", list(ckpt.keys()))
    if 'model' in ckpt:
        model_keys = list(ckpt['model'].keys())
        print(f"\nTotal model parameters: {len(model_keys)}")
        print("Sample model keys:")
        for key in model_keys[:20]:
            print(f"  {key}")
    else:
        print("No 'model' key found in checkpoint.")

def inspect_encoder_decoder_keys(model_weights):
    """
    Separate and preview encoder and decoder keys from a Fairseq model state dict.
    """
    encoder_keys = [k for k in model_weights if "encoder.w2v_model" in k]
    decoder_keys = [k for k in model_weights if k.startswith("decoder.")]

    print(f"\nEncoder keys count: {len(encoder_keys)}")
    print("Sample encoder keys:")
    for k in encoder_keys[:10]:
        print(f"  {k}")

    print(f"\nDecoder keys count: {len(decoder_keys)}")
    print("Sample decoder keys:")
    for k in decoder_keys[:10]:
        print(f"  {k}")

def compare_key_formats(fairseq_keys, hf_keys):
    """
    Compare Fairseq-style keys and Hugging Face-style keys.
    """
    print("\nFairseq-style keys (sample):")
    for k in fairseq_keys[:10]:
        print(f"  {k}")

    print("\nHugging Face-style keys (sample):")
    for k in hf_keys[:10]:
        print(f"  {k}")

def check_model_loading(model, state_dict):
    """
    Check for missing or unexpected keys when loading a state dict into a model.
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("\nModel loading diagnostics:")
    print(f"Missing keys: {len(missing_keys)}")
    for k in missing_keys:
        print(f"  {k}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    for k in unexpected_keys:
        print(f"  {k}")

def print_model_summary(model):
    """
    Print the model architecture and total number of parameters.
    """
    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")