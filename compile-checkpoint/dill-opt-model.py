import dill

import torch
torch.set_float32_matmul_precision('high')

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.l(x))

model = ToyModel().cuda()

opt_model = torch.compile(model)

opt_model(torch.randn(10,10).cuda())

print(dir(opt_model))


# TypeError: cannot pickle 'ConfigModuleInstance' object
# dill has issues with dynamic classes
# I believe lots of stuff are dynamic in pt 2.0
dill.dump(opt_model, open('model_checkpoint.pth', 'wb'))

dill.load(open('model_checkpoint.pth', 'rb'))(torch.randn(10,10).cuda())