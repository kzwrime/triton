import binascii
import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import triton
import triton.backends
from triton.backends.nvidia.driver import ty_to_cpp

desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""

# mnemonic
def align(x):
    return [['tt.divisibility', x]]

# each element of specialization_data represents (argname, type, constexpr_value, attrs)
# if an argument (without tl.constexpr hint) should be spec to 1, let it's constexpr_value = 1
# e.g.
# add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE)'s specialization_data
# specialization_data = [
#     ('*fp32', None, [['tt.divisibility', 16],]),
#     ('*fp32', None, [['tt.divisibility', 16],]),
#     ('*fp32', None, [['tt.divisibility', 16],]),
#     ('i32', None, None),
#     1024,
# ]
def aot_compile_wrapper(
    kernel_jit_func: triton.runtime.JITFunction,
    specialization_data: Union[list, dict],
    kernel_name: str,
    out_name: str,
    out_path: str,
    grid: list,
    num_warps = None,
    num_stages = None,
    target = None,
    debug = False,
):
    
    signature = dict()
    constants = dict()
    attrs = dict()
    
    # optional: num_warps, num_stages, num_ctas
    len_spec = len(specialization_data)
    len_args = len(kernel_jit_func.arg_names)
    assert len_args <= len_spec and len_spec <= len_args + 3, f"len_args={len_args}, len_spec={len_spec}"
    if isinstance(specialization_data, list):
        if num_warps is None:
            num_warps = specialization_data[len_args]
        if num_stages is None:
            num_stages = specialization_data[len_args+1]
        for i, karg_name in enumerate(kernel_jit_func.arg_names):
            sdata = specialization_data[i]
            if not isinstance(sdata, (list, tuple)):
                signature[karg_name] = 'constexpr'
                constants[(i, )] = sdata
            else:
                type = sdata[0]
                constexpr_value = sdata[1]
                arg_attrs = sdata[2]

                if constexpr_value != None:
                    type = 'constexpr'
                    constants[(i, )] = constexpr_value
                else:
                    attrs[(i,)] = list(arg_attrs) if arg_attrs != None else []
                signature[karg_name] = type
    elif isinstance(specialization_data, dict):
        if num_warps is None:
            num_warps = specialization_data['num_warps']
        if num_stages is None:
            num_stages = specialization_data['num_stages']
        for i, karg_name in enumerate(kernel_jit_func.arg_names):
            sdata = specialization_data[karg_name]
            if not isinstance(sdata, (list, tuple)):
                signature[karg_name] = 'constexpr'
                constants[(i, )] = sdata
            else:
                type = sdata[0]
                constexpr_value = sdata[1]
                arg_attrs = sdata[2]

                if constexpr_value != None:
                    type = 'constexpr'
                    constants[(i, )] = constexpr_value
                else:
                    attrs[(i,)] = list(arg_attrs) if arg_attrs != None else []
                signature[karg_name] = type
    else:
        raise ValueError(f"{type(specialization_data)} specialization_data's type should be list or dict")
    
    aot_compile(
        kernel_jit_func,
        signature,
        constants,
        attrs,
        kernel_name,
        out_name,
        out_path,
        grid,
        num_warps,
        num_stages,
        target = target,
        debug = debug,
    )

accept_constant_types = (int, float, bool)
def constant_to_ty(s):
    if isinstance(s, bool):
        return "i32"
    elif isinstance(s, int):
        return "i64"
    elif isinstance(s, float):
        return "fp64"
    else:
        raise NotImplementedError()
def constant_to_cstr(s):
    if isinstance(s, bool):
        return str(int(s))
    else:
        return str(s)

def main(args):
    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    grid = args.grid.split(",")
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
    sig_hash = hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = 'constexpr'
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]
    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}

    aot_compile(kernel_jit_func = kernel,
                signature = signature,
                constants = constants,
                attrs = attrs,
                kernel_name = args.kernel_name,
                out_name = out_name,
                out_path = out_path,
                grid = grid,
                num_warps = args.num_warps,
                num_stages = args.num_stages,
    )

def aot_compile(
    kernel_jit_func: triton.runtime.JITFunction,
    signature: dict,
    constants: dict,
    attrs: dict,
    kernel_name: str,
    out_name: str,
    out_path: str,
    grid: list,
    num_warps: int,
    num_stages: int,
    target = None,
    debug = False,
):
    out_name = out_name if out_name else kernel_name
    out_path = Path(out_path) if out_path else Path(out_name)

    assert len(grid) == 3
    
    src = triton.compiler.ASTSource(fn=kernel_jit_func, constexprs=constants, signature=signature, attrs=attrs)
    opts = {"num_warps": num_warps, "num_stages": num_stages}
    ccinfo = triton.compile(src, target=target, options=opts)
    if ccinfo.metadata.global_scratch_size > 0:
        raise RuntimeError("AOT compiling kernels with global scratch requirements is not yet implemented")

    sig_hash = ccinfo.hash
    constants = ccinfo.src.constants

    constexpr_indices = [i for (i, p) in enumerate(kernel_jit_func.params) if p.is_constexpr]
    constexpr_indices = set(constexpr_indices)

    arg_names = []
    arg_types = []
    arg_names_not_1 = []
    arg_types_not_1 = []
    
    for i, arg_name in enumerate(kernel_jit_func.arg_names):
        constval = constants.get((i, ), None)
        if constval is None:
            arg_names.append(arg_name)
            arg_types.append(signature[arg_name])
            arg_names_not_1.append(arg_name)
            arg_types_not_1.append(signature[arg_name])
        elif not i in constexpr_indices and constval == 1:
            arg_names.append(arg_name)
            arg_types.append("i32")

    # dump C stub code
    suffix = ''
    for i, ty in enumerate(signature.values()):
        suffix += str(i)
        if constants.get((i, ), None) == 1:
            suffix += 'c'
        attr_list = attrs.get((i, ), [])
        for attr in attr_list:
            if attr[0] == 'tt.divisibility':
                suffix += f'd{attr[1]}'
                break
            else:
                raise NotImplementedError("AOT only support 'tt.divisibility' in attrs now.")
        suffix += 'X'
    suffix = suffix[:-1]

    if debug:
        print(suffix)
        ptx = ccinfo.asm["ptx"].replace('\n', '\n// ')
    else:
        ptx = "// no ptx"

    const_sig = 'x'.join([str(v) for v in constants.values()])
    meta_sig = f"warps{num_warps}xstages{num_stages}"
    doc_string = [f"{kernel_jit_func.arg_names[k[0]]}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={num_warps}", f"num_stages={num_stages}"]

    func_name = '_'.join([out_name, sig_hash, suffix])
    asm = ccinfo.asm["cubin"]  # store binary data once
    hex_ = str(binascii.hexlify(asm))[2:-1]
    params = {
        "kernel_name": func_name,
        "triton_kernel_name": kernel_name,
        "bin_size": len(asm),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names_not_1, arg_types_not_1)]),
        "full_signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]),
        "constexpr": "; ".join(
            [
                f"{ty_to_cpp(constant_to_ty(constval))}, {kernel_jit_func.arg_names[vi[0]]}, {constant_to_cstr(constval)}"
                for vi, constval in constants.items()
                if type(constval) in accept_constant_types and vi[0] in constexpr_indices
            ] + [f"int, num_warps, {num_warps}", f"int, num_stages, {num_stages}"]),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names_not_1] + ["&global_scratch"]),
        "num_args": len(arg_names_not_1) + 1,
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": num_warps,
        "algo_info": '_'.join([const_sig, meta_sig]),
        "gridX": grid[0],
        "gridY": grid[1],
        "gridZ": grid[2],
        "_placeholder": "",
        "ptx": ptx
    }
    for ext in ['h', 'c']:
        template_path = Path(__file__).parent / "extra" / "cuda" / f"compile.{ext}"
        with out_path.with_suffix(f".{sig_hash}_{suffix}.{ext}").open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))

if __name__ == "__main__":

    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    args = parser.parse_args()

    main(args)