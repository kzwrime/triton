import os,sys
import torch
import triton
import add

from triton.tools.compile import aot_compile_wrapper, align
from triton.backends.compiler import BaseBackend, GPUTarget
from itertools import product

kernel_func = add.add_kernel
kernel_name="add_kernel"
kernel_output_name="add_kernel"
output_path_with_prefix="./kernels/add_kernel"
default_grid=(1024,1,1)

_ = None

# Given all combinations
func_args = {
    # "x_ptr": [('*fp32', _, align_16)],
    # "y_ptr": [('*fp32', _, align_16)],
    # "output_ptr": [('*fp32', _, _)],
    # "n_elements": [('i32', 1, _)],
    # "BLOCK_SIZE": [64, 128],
#     "x_ptr": [('*fp32', _, align_16), ('*fp32', _, _)],
#     "y_ptr": [('*fp32', _, align_16), ('*fp32', _, _)],
#     "output_ptr": [('*fp32', _, align_16), ('*fp32', _, _)],
#     "n_elements": [('i32', 1, _), ('i32', _, _), ('i32', _, align_16)],
#     "BLOCK_SIZE": [64, 128],

    # "x_ptr": [('*fp32', _, align(16))],
    # "y_ptr": [('*fp32', _, align(16))],
    # "output_ptr": [('*fp32', _, align(16))],
    # "n_elements": [('i32', _, _)],
    # "BLOCK_SIZE": [1024],
    
    "x_ptr": [('*fp32', _, align(16)), ('*fp32', _, align(8)), ('*fp32', _, _)],
    "y_ptr": [('*fp32', _, align(16)), ('*fp32', _, align(8)), ('*fp32', _, _)],
    "output_ptr": [('*fp32', _, align(16)), ('*fp32', _, align(8)), ('*fp32', _, _)],
    "n_elements": [('i32', _, _)],
    "BLOCK_SIZE": [512, 1024],
    
    
}

keys, values = zip(*func_args.items())

# product is used to get all combinations.
# If you only need a specified series of kernel, you are free to create specific config.
_all_configs = product(*values)

all_configs = []
for config_values in _all_configs:
    config = dict(zip(keys, config_values))
    
    # Exclude combinations that do not meet your requirements.
    if config["x_ptr"][0] != config["y_ptr"][0]:
        continue
    if config["x_ptr"][2] != config["y_ptr"][2]:
        continue
    if config["x_ptr"][2] != config["output_ptr"][2]:
        continue

    all_configs.append(config_values)
    
print(f"num of kernels = {len(all_configs)}")

# specific target if you need
# target = GPUTarget("cuda", 90, 32)
target = None

def generate(i_config_values):
    config_i = i_config_values[0]
    if config_i % 100 == 0:
        print(f"config_i={config_i}")
    config_values = i_config_values[1]
    config = dict(zip(keys, config_values))

    aot_compile_wrapper(
        kernel_func,
        config_values,
        kernel_name,
        kernel_output_name,
        output_path_with_prefix,
        default_grid,
        num_warps = 4,
        num_stages = 2,
        target = target,
        # debug = True
    )
    return 0

for i_config_values in enumerate(all_configs):
    generate(i_config_values)

# If you have a lot of kernel, you can use multiple processes to compile in parallel.
# uncomment the following and adjust NUM_PROC

# NUM_PROC = 17   # Set to the number of processes you need.
# from multiprocessing import Process, Pool
# with Pool(NUM_PROC) as pool:
#     results = pool.map(generate, enumerate(all_configs), chunksize=10)