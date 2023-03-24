import torch
import torch.nn as nn

class MyCustomModule(nn.Module):
    def __init__(self):
        super(MyCustomModule, self).__init__()
        self.optimized_module = None
        self.persistent_compilation = True
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
        if self.persistent_compilation:
            self.compile()

m = MyCustomModule()

import time 

tic = time.time()
m(torch.randn(1, 10))
toc = time.time()

# On my mac this takes 0.00046896934509277344 s
print(f"First inference time: {toc - tic} seconds")

m.compile(backend="inductor")

tic = time.time()
m(torch.randn(1, 10))
toc = time.time()

# On my mac this takes First inference time: 6.145598888397217 seconds
print(f"Compiled inference time: {toc - tic} seconds")

# And this now no longer crashes
# Because we are not pickling a frame
torch.save(m, "model.pt")

# This will actually load a model and then automatically compile it
# Thanks Alban for this idea
torch.load(m)
