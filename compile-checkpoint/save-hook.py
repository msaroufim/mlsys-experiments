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
        self.cache_dir = cache_dir

    def forward(self, x):
        return self.relu(self.l(x))

def cache_state_dict_hook(module, state_dict, prefix, local_metadata):
    cache_files = glob.glob(os.path.join(module.cache_dir, '**'), recursive=True)

    for file_path in cache_files:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                state_dict[prefix + os.path.relpath(file_path, module.cache_dir)] = file.read()

model = ToyModel().cuda()
opt_model = torch.compile(model)

opt_model(torch.randn(10,10).cuda())

# Register the cache_state_dict_hook on the model
model._register_state_dict_hook(cache_state_dict_hook)

# Save the model's state_dict
state_dict = model.state_dict()
print(state_dict)
torch.save(state_dict, 'model_checkpoint.pth')