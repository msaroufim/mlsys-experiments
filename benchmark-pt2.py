import torch
import torch._dynamo as torchdynamo
import time
import warnings
import prettytable
from typing import Optional

warnings.filterwarnings("ignore")

torchdynamo.reset()

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(2, 2))

    def forward(self, x):
        return x * self.weight

model = ToyModel().cuda()

def bench_loop(model, sample_input, num_iters, is_training=False, optimizer=None):
    durations = []
    for _ in range(num_iters):
        start = time.time()
        
        if is_training and optimizer:
            optimizer.zero_grad()
            output = model(sample_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        else:
            model(sample_input)
        
        end = time.time()

        if sample_input.get_device() >= 0:
            torch.cuda.synchronize()

        durations.append(end - start)
    
    return sum(durations) / num_iters

def benchmark_compile_inference(model: torch.nn.Module, sample_input: torch.Tensor, num_iters: int = 5, backend: Optional[str] = None, mode="default", is_training=False, optimizer=None):
    """
    Use this utility to benchmark torch.compile
    """
    if backend:
        try:
            opt_model = torch.compile(model, backend=backend, mode=mode)
            
            # Compilation only happens after the first inference
            compilation_time = bench_loop(opt_model, sample_input, 1, is_training, optimizer)

        except:
            print(f"Failed to compile {backend} with mode {mode} Make sure you've installed the correct dependencies since they're not installed by default")
            return None, None
    else:
        opt_model = model
        compilation_time = None

    # Benchmark
    running_time = bench_loop(opt_model, sample_input, num_iters, is_training, optimizer)
    
    return compilation_time, running_time

def create_optimizer(model, optimizer_name="SGD", learning_rate=0.01):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
def bench_all(model, sample_input, num_iters, is_training=False, optimizer_name=None):
    if next(model.parameters()).is_cuda:
        print("Your model is loaded GPU")
        if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
            print("Your GPU supports tensor cores, we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
            torch.set_float32_matmul_precision("high")


    table = prettytable.PrettyTable()
    table.field_names = ["Train/Inference", "Backend", "Mode", "Compilation Time", "Average Running Time", "Speedup"]

    optimizer = None
    if is_training and optimizer_name:
        optimizer = create_optimizer(model, optimizer_name)

    eager_time = None
    torchdynamo.reset()
    _, eager_time = benchmark_compile_inference(model, sample_input, num_iters, None, None, is_training, optimizer)
    table.add_row([("Training" if is_training else "Inference"), "Eager", "-", "-", eager_time, "-"])

    for backend in torchdynamo.list_backends():
        if backend == "ipex":  # ipex has an annoying import error it prints
            continue

        if backend == "inductor":
            for mode in list(torch._inductor.list_mode_options().keys()) + [None]:
                if mode == "default":
                    continue
                torchdynamo.reset()
                compilation_time, running_time = benchmark_compile_inference(model, sample_input, num_iters, backend, mode, is_training, optimizer)
                if running_time is not None:
                    speedup = (eager_time - running_time) / eager_time if eager_time else None
                    table.add_row([("Training" if is_training else "Inference"), backend, mode or "-", compilation_time or "-", running_time, speedup or "-"])
        else:
            torchdynamo.reset()
            compilation_time, running_time = benchmark_compile_inference(model, sample_input, num_iters, backend, None, is_training, optimizer)
            if running_time is not None:
                speedup = (eager_time - running_time) / eager_time if eager_time else None
                table.add_row([("Training" if is_training else "Inference"), backend, "-", compilation_time or "-", running_time, speedup or "-"])

    print(table)

if __name__ == "__main__":
    print("===== Inference =====")
    bench_all(model, torch.ones(1024, 2, 2).cuda(), 5)
    print("\n===== Training =====")
    bench_all(model, torch.ones(1024, 2, 2).cuda(), 5, is_training=True, optimizer_name="SGD")
