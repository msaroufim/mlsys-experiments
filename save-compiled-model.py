import torch
import torch.nn as nn

class MyCustomModule(nn.Module):
    def __init__(self):
        super(MyCustomModule, self).__init__()
        # Your module layers and initialization here
        self.optimized_module = None
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        self.linear(x)

    def compile(self, *args, **kwargs):
        self.optimized_module = torch.compile(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.optimized_module:
            return self.optimized_module(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the optimized_module from the state
        state.pop("optimized_module", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.optimized_module = None

m = MyCustomModule()

import time 

tic = time.time()
m(torch.randn(1, 10))
toc = time.time()

print(f"First inference time: {toc - tic} seconds")

m.compile(backend="inductor")

tic = time.time()
m(torch.randn(1, 10))
toc = time.time()

print(f"Compiled inference time: {toc - tic} seconds")

torch.save(m, "model.pt")
