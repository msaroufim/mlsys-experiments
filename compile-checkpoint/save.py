import torch
import os
import glob

torch.set_float32_matmul_precision('high')
cache_dir = 'cache'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.l(x))

def save_model_with_cache(model, cache_dir):
    cache_files = glob.glob(os.path.join(cache_dir, '**'), recursive=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'cache_files': [os.path.relpath(file_path, cache_dir) for file_path in cache_files]
    }

    # Add binary data for cache files
    for file_path in cache_files:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                checkpoint[os.path.relpath(file_path, cache_dir)] = file.read()

    return checkpoint

model = ToyModel().cuda()
opt_model = torch.compile(model)

opt_model(torch.randn(10,10).cuda())

checkpoint = save_model_with_cache(model, cache_dir)

print(checkpoint)
torch.save(checkpoint, 'model_checkpoint.pth')