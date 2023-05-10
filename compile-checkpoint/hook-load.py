import torch
import os
import glob

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.l(x))

def load_inductor_state_dict_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # Retrieve the cache directory from the state dict
    cache_dir = state_dict.get('cache_dir', 'cache')

    # Remove the cache directory key from the state dict
    if 'cache_dir' in state_dict:
        del state_dict['cache_dir']

    # Instantiate the cache files to disk
    cache_files = [path for path in state_dict if not path.startswith('._')]
    for path in cache_files:
        file_path = os.path.join(cache_dir, path)
        data = state_dict[path]
        with open(file_path, 'wb') as file:
            file.write(data)
        
        # Remove the cache file from the state dict
        del state_dict[path]
    
    print(state_dict)

# Instantiate the model
model = ToyModel().cuda()

# Register the load_state_dict_pre_hook
model._register_load_state_dict_pre_hook(load_inductor_state_dict_hook)

# Load the checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# # Load the model's state_dict
# model.load_state_dict(checkpoint['model_state_dict'])
