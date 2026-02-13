import torch

ckpt_path = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/clutterMixed_/Optical_sweep_clutter_mix_lr0002/model_final.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Print keys to see what's inside
print("Keys in checkpoint:", ckpt.keys())

# Try to read the config if stored
if "cfg" in ckpt:
    print("Original Config:\n", ckpt["cfg"])
else:
    print("No config saved inside the checkpoint.")
