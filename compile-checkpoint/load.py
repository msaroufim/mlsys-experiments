import torch
import os
import glob

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.cache_dir = 'cache'
        self.cache_files = glob.glob(os.path.join(self.cache_dir, '**'), recursive=True)
        self.cache_files = [path for path in self.cache_files if not os.path.isdir(path)]

    def forward(self, x):
        return self.relu(self.l(x))

# Instantiate the model
model = ToyModel().cuda()

# Load the checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# Set the cache directory
os.environ['TORCHINDUCTOR_CACHE_DIR'] = model.cache_dir

# Instantiate the cache files
for path in model.cache_files:
    with open(path, 'wb') as file:
        file.write(checkpoint[os.path.relpath(path, model.cache_dir)])

# Remove the cache files from the checkpoint
for path in model.cache_files:
    checkpoint.pop(os.path.relpath(path, model.cache_dir), None)

# Load the model's state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Print the tree structure of the cache folder
print(f"Cache folder structure:\n{model.cache_dir}")
for root, dirs, files in os.walk(model.cache_dir):
    level = root.replace(model.cache_dir, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")