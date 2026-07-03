#Abhijit Somala
# Simple file to clear cache to prevent bias. This helped address an issue I was having earlier.

import torch
torch.cuda.empty_cache()
