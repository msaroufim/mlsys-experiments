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

model = ToyModel().cuda()
opt_model = torch.compile(model)

opt_model(torch.randn(10,10).cuda())


checkpoint = {
    'model_state_dict': model.state_dict(),
    'cache_files': [os.path.relpath(file_path, model.cache_dir) for file_path in model.cache_files]
}

# Add binary data for cache files
for file_path in model.cache_files:
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            checkpoint[os.path.relpath(file_path, model.cache_dir)] = file.read()

print(checkpoint)
torch.save(checkpoint, 'model_checkpoint.pth')
