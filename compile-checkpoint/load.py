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

def instantiate_cache(cache_dir):
    cache_files = glob.glob(os.path.join(cache_dir, '**'), recursive=True)
    cache_files = [path for path in cache_files if not os.path.isdir(path)]
    return cache_files

# Instantiate the model
model = ToyModel().cuda()

# Set the cache directory
cache_dir = 'cache'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir

# Instantiate the cache files
cache_files = instantiate_cache(cache_dir)

# Load the checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# Instantiate the cache files to disk
for path in cache_files:
    with open(path, 'wb') as file:
        file.write(checkpoint[os.path.relpath(path, cache_dir)])

# Remove the cache files from the checkpoint
for path in cache_files:
    checkpoint.pop(os.path.relpath(path, cache_dir), None)

# Load the model's state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Print the tree structure of the cache folder
print(f"Cache folder structure:\n{cache_dir}")
for root, dirs, files in os.walk(cache_dir):
    level = root.replace(cache_dir, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")