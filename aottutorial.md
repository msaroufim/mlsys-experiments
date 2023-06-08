## Run file

```python

# TORCH_LOGS=output_code python test_wrapper.py
import torch
import torch._inductor.config

torch._inductor.config.cpp_wrapper = True

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))

x = torch.randn((32, 64))
y = torch.randn((32, 64))

m = ToyModel().cuda()

opt_m = torch.compile(m)

opt_m(x, y)
```

This will print a file which you can actually run using python and nothing else

```python
-06-08 22:32:44,594] torch._inductor.graph.__output_code: [DEBUG] Output code: 

import torch
from torch._inductor.codecache import CppWrapperCodeCache

cpp_wrapper_src = (
'''

#include "/tmp/torchinductor_ubuntu/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void cpp_fused_add_cos_sin_0(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2048L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
            auto tmp1 = tmp0.sin();
            auto tmp3 = tmp2.cos();
            auto tmp4 = tmp1 + tmp3;
            tmp4.store(out_ptr0 + static_cast<long>(i0));
        }
    }
}
std::vector<at::Tensor> inductor_entry_cpp(const std::vector<at::Tensor>& args) {
    at::Tensor primals_1 = args[0];
    at::Tensor primals_2 = args[1];
    at::Tensor primals_3 = args[2];
    at::Tensor primals_4 = args[3];

    c10::optional<at::Scalar> optional_scalar;
    c10::optional<c10::string_view> optional_string;
    torch::List<c10::optional<at::Scalar>> optional_list;
        auto buf0 = at::empty_strided({32L, 64L}, {64L, 1L}, at::device(at::kCPU).dtype(at::kFloat));
        cpp_fused_add_cos_sin_0((float*)(primals_3.data_ptr()), (float*)(primals_4.data_ptr()), (float*)(buf0.data_ptr()));
        primals_3.reset();
        primals_4.reset();
        auto buf1 = at::empty_strided({32L, 10L}, {10L, 1L}, at::device(at::kCPU).dtype(at::kFloat));
        at::addmm_out(buf1, primals_2, buf0, at::as_strided(primals_1, {64L, 10L}, {1L, 64L}, 0L), 1, 1);
        primals_1.reset();
        primals_2.reset();
        return {buf1, buf0};
}
'''
)

module = CppWrapperCodeCache.load(cpp_wrapper_src, 'inductor_entry_cpp', 'czenwgemzbe2etzbh7hzhnwjhyamvwirgodyjlly75fayy4tp3rx', False)

def _wrap_func(f):
    def g(args):
        args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]
        return f(args_tensor)
    return g
call = _wrap_func(module.inductor_entry_cpp)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 64), (64, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
```

Speficically we need to change 

```python
if __name__ == "__main__":
  call(torch.randn(10,64)
```

## Relevant files to study
* Inductor code cache
* Inductor codegen wrapper
* Inductor benchmark utils

### CUDA specific things

CUDA also needs us to package some cubin files which are the compiled triton kernels, example: 

```python
triton_poi_fused_add_cos_sin_0 = loadKernel("/tmp/torchinductor_ubuntu/xe/cxey2xzk5wbxvqwxeo3qeojp4lru5eyss2yi7v7see5jqneoefmp.cubin", "triton_poi_fused_add_cos_sin_0_0d1d2d3d");
```

So we need to package these and make sure they're on the host machine when we come up with the final solution
