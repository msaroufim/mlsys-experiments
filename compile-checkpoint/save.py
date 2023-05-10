import torch
import os
import glob

torch.set_float32_matmul_precision('high')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'cache'

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.cache_dir = 'cache'
        self.cache_files = glob.glob(os.path.join(self.cache_dir, '**'), recursive=True)

        # Register a buffer to store the cache contents
        self.register_buffer('cache', None)

    def forward(self, x):
        return self.relu(self.l(x))

# Instantiate the model
model = ToyModel().cuda()

# Save the model state_dict and cache contents
checkpoint = {
    'model_state_dict': model.state_dict(),
}

# Add binary data of cache files and directories to the checkpoint
for path in model.cache_files:
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            binary_data = file.read()
            checkpoint[os.path.relpath(path, model.cache_dir)] = binary_data

print(checkpoint)

# Save the checkpoint dictionary
torch.save(checkpoint, 'model_checkpoint.pth')
