import torch
print(f"GPU dispo: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nom: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
else:
    print("Aucun GPU detecte")
