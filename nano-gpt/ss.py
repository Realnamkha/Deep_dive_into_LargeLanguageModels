#check for gpu
import torch
if torch.backends.mps.is_available():
   mps_device = torch.device("mps")
   x = torch.ones(1, device=mps_device)
   print (x)
else:
   print ("MPS device not found.")
   
import torch

if torch.backends.mps.is_available():
    print("MPS device is available and being used.")
else:
    print("MPS device is not available.")