import torch
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench
from torchvision.models import resnet18

# Number of CUDA cores (or SPs) per SM for various Nvidia GPU architectures
cores_per_sm = {
    'Ampere': 128,
    'Turing': 64, 
    'Volta': 64, 
    'Pascal': 128, 
    'Maxwell': 128, 
    'Kepler': 192, 
    'Fermi': 32, 
    'Tesla': 8
}

device = torch.device('cuda')
properties = torch.cuda.get_device_properties(device)
sm_count = properties.multi_processor_count
clock_speed_ghz = torch.cuda.clock_rate() * 1e-6

# You'll need to manually set the architecture to the correct value for your GPU
def get_architecture():
    cc = torch.cuda.get_device_capability()
    if cc[0] == 8:
        return 'Ampere'
    elif cc[0] == 7:
        return 'Turing'  # or 'Volta'
    elif cc[0] == 6:
        return 'Pascal'
    elif cc[0] == 5:
        return 'Maxwell'
    elif cc[0] == 3:
        return 'Kepler'
    else:
        raise ValueError(f'Unknown compute capability: {cc}')

device = torch.device('cuda')
architecture = get_architecture()

cores_per_sm = cores_per_sm[architecture]

# Each core can perform 2 operations per cycle (one multiply and one accumulate)
operations_per_core = 2

# Calculate the theoretical peak FLOPs (in TFLOPs)
max_flops = sm_count * cores_per_sm * operations_per_core * clock_speed_ghz

def get_flop_utilization(f):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        f()
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    ms_per_iter = do_bench(f)
    iters_per_second = 1e3/ms_per_iter
    actual_flops = iters_per_second * total_flops / 1e12  # in TFLOPs
    flop_utilization = actual_flops / max_flops  # Ratio of actual FLOPs to max FLOPs
    print(f"{flop_utilization * 100} %")

model = resnet18().cuda() #.half()
inp = torch.randn(128, 3, 224, 224, device='cuda', dtype=torch.float32)
get_flop_utilization(lambda: model(inp).sum().backward())

# compiled_model = torch.compile(model)
# get_flop_utilization(lambda: compiled_model(inp).sum().backward())
