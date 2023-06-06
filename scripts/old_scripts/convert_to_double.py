import torch
import torchvision

# Get pytorch model
base_path = "/home/lolo/omav_ws/src/rotors_simulator/rotors_description/models/"
model_path = base_path + "T_c.pt"
pytorch_model = torch.jit.load(model_path)

# Prepare stuff
pytorch_model.double()

# Export
out_path = base_path + "T_c_double.pt"
pytorch_model.save(out_path)



