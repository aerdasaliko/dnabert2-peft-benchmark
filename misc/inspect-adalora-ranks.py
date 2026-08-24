from safetensors.torch import load_file

state_dict = load_file("/path-to-safetensors/adapter_model.safetensors")
for k, v in state_dict.items():
    if "lora_E" in k:
        active = (v.abs() > 1e-8).sum().item()
        total = v.numel()
        print(f"{k}: final rank = {active}/{total}")