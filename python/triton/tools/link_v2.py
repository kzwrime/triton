from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union, Dict
import glob

from dataclasses import dataclass
from collections import defaultdict

def _exists(x):
    return x is not None


class LinkerError(Exception):
    pass


@dataclass
class KernelLinkerMeta:
    orig_kernel_name: str
    arg_names: Sequence[str]
    arg_ctypes: Sequence[str]
    sizes: Sequence[Union[int, None]]
    sig_hash: str
    triton_suffix: str
    suffix: str
    num_specs: int
    constexpr_arg_types: Sequence[str]
    constexpr_arg_names: Sequence[str]
    constexpr_arg_vals: Sequence[str]
    
    def __hash__(self) -> int:
        return hash(self.sig_hash)
    """ number of specialized arguments """


class HeaderParser:

    def __init__(self) -> None:
        import re

        # [kernel_name, c signature]
        self.linker_directives = re.compile("//[\\s]*tt-linker:[\\s]*([\\w]+):(.+):(.+):(.+)")
        # [name, hash, suffix]
        self.kernel_name = re.compile("^([\\w]+)_([\\w]+)_([\\w]+)$")
        # [(type, name)]
        self.c_sig = re.compile("[\\s]*(\\w+)\\s(\\w+)[,]?")
        # [d|c]
        self.arg_suffix = re.compile("[c,d]")

        self.kernels = defaultdict(list)

    def extract_linker_meta(self, header: str):
        for ln in header.splitlines():
            if ln.startswith("//"):
                m = self.linker_directives.match(ln)
                if _exists(m):
                    ker_name, c_sig, algo_info, c_constexpr_sig = m.group(1), m.group(2), m.group(3), m.group(4)
                    name, sig_hash, suffix = self._match_name(ker_name)
                    c_types, arg_names = self._match_c_sig(c_sig)
                    constexpr_arg_types, constexpr_arg_names, constexpr_arg_vals = list(zip(*[line.split(", ") for line in c_constexpr_sig.split("; ")]))
                    num_specs, sizes = self._match_suffix(suffix, c_sig)
                    self._add_kernel(
                        "_".join([name, algo_info]),
                        KernelLinkerMeta(
                            orig_kernel_name=name,
                            arg_names=arg_names,
                            arg_ctypes=c_types,
                            sizes=sizes,
                            sig_hash=sig_hash,
                            triton_suffix=suffix,
                            suffix=suffix,
                            num_specs=num_specs,
                            constexpr_arg_types=constexpr_arg_types,
                            constexpr_arg_names=constexpr_arg_names,
                            constexpr_arg_vals=constexpr_arg_vals,
                        ),
                    )

    def _match_name(self, ker_name: str):
        m = self.kernel_name.match(ker_name)
        if _exists(m):
            name, sig_hash, suffix = m.group(1), m.group(2), m.group(3)
            return name, sig_hash, suffix
        raise LinkerError(f"{ker_name} is not a valid kernel name")

    def _match_c_sig(self, c_sig: str):
        m = self.c_sig.findall(c_sig)
        if len(m):
            tys, args = [], []
            for ty, arg_name in m:
                tys.append(ty)
                args.append(arg_name)
            return tys, args

        raise LinkerError(f"{c_sig} is not a valid argument signature")

    @staticmethod
    def _match_first_non_digit_index(s):
        return next((i for i, char in enumerate(s) if not char.isdigit()), None)

    def _match_suffix(self, suffix: str, c_sig: str):
        args = c_sig.split(",")
        _attrs = suffix.split("X")
        attrs = dict()
        for attr in _attrs:
            pos = self._match_first_non_digit_index(attr)
            if pos != None:
                attrs[int(attr[:pos])] = attr[pos:]
        num_specs = 0
        sizes = []
        # scan through suffix, first find the index,
        # then see if it is followed by d or c
        for i in range(len(args)):
            attr = attrs.get(i, None)
            if attr != None:
                num_specs += 1
                sizes.extend([None] * (i - len(sizes)))
                assert attr[0] in ('c', 'd'), f"attr={attr} is invalid"
                if attr[0] == 'c':
                    sizes.append(1)
                else:
                    sizes.append(int(attr[1:]))
            if not (i < len(args) - 1):
                sizes.extend([None] * (len(args) - len(sizes)))
        return num_specs, sizes

    def _add_kernel(self, name: str, ker: KernelLinkerMeta):
        if name in self.kernels:
            last: KernelLinkerMeta = self.kernels[name][-1]

            for cur, new_ in zip(last.arg_ctypes, ker.arg_ctypes):
                if cur != new_:
                    raise LinkerError(
                        f"Mismatched signature for kernel {name}: \n\texisting sig is: {','.join(last.arg_ctypes)}\n\tcurrent is: {','.join(ker.arg_ctypes)}"
                    )

        self.kernels[name].append(ker)


def gen_signature_with_full_args(m):
    return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])

def gen_constexpr_signature_with_full_args(m):
    return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.constexpr_arg_types, m.constexpr_arg_names)])


def gen_signature(m):
    arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
    arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
    sig = ", ".join([f"{ty} {arg}" for ty, arg in zip(arg_types, arg_names)])
    return sig


# generate declarations of kernels with meta-parameter and constant values
def make_algo_decls(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    return f"""
CUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])});
CUresult {name}_with_grid(CUstream stream, {gen_signature_with_full_args(metas[-1])}, unsigned int gridx, unsigned int gridy, unsigned int gridz);
void load_{name}();
void unload_{name}();
    """


# generate declarations of kernels with meta-parameter and constant values
def make_global_decl(meta: KernelLinkerMeta) -> str:
    return f"""
CUresult {meta.orig_kernel_name}_default(CUstream stream, {gen_signature_with_full_args(meta)});
CUresult {meta.orig_kernel_name}(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id);
void load_{meta.orig_kernel_name}();
void unload_{meta.orig_kernel_name}();
    """


# generate dispatcher function for kernels with different meta-parameter and constant values
def make_default_algo_kernel(meta: KernelLinkerMeta) -> str:
    src = f"CUresult {meta.orig_kernel_name}_default(CUstream stream, {gen_signature_with_full_args(meta)}){{\n"
    src += (f"  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n")
    src += "}\n"

    return src

def _make_kernel_hints_dispatcher(name: str, metas: Sequence[KernelLinkerMeta], with_grid = False) -> str:
    src = ""
    if with_grid:
        src += (f"CUresult {name}_with_grid(CUstream stream, {gen_signature_with_full_args(metas[-1])}, unsigned int gridx, unsigned int gridy, unsigned int gridz){{")
    else:
        src += (f"CUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])}){{")
    src += "\n"
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        cond_fn = (  #
            lambda val, hint: f"({val} % {hint} == 0)"  #
            if hint >= 2 and hint % 2 == 0  #
            else f"({val} == {hint})"  #
            if hint == 1  #
            else None)
        conds = " && ".join([  #
            cond_fn(val, hint)  #
            for val, hint in zip(meta.arg_names, meta.sizes)  #
            if hint is not None
        ])
        src += (f"  if ({conds})\n" if any(meta.sizes) else "if (1)\n"
                )  # Edge case where no specializations hence no dispatching required
        arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
        if with_grid:
            src += f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}_with_grid(stream, {', '.join(arg_names)}, gridx, gridy, gridz);\n"
        else:
            src += f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n"
    src += "\n"
    src += "  return CUDA_ERROR_INVALID_VALUE;\n"
    src += "}\n"

    return src

# generate dispatcher function for kernels with different integer value hints
def make_kernel_hints_dispatcher(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    src = f"// launcher for: {name}\n"
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        src += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {gen_signature(meta)});\n"
        src += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}_with_grid(CUstream stream, {gen_signature(meta)}, unsigned int gridx, unsigned int gridy, unsigned int gridz);\n"
        # src += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {gen_signature(meta)});\n"
        # src += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}_with_grid(CUstream stream, {gen_signature(meta)}, unsigned int gridx, unsigned int gridy, unsigned int gridz);\n"
    src += "\n"

    src += _make_kernel_hints_dispatcher(name, metas)
    src += _make_kernel_hints_dispatcher(name, metas, with_grid=True)

    for mode in ["load", "unload"]:
        src += f"\n// {mode} for: {name}\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += f"void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
        src += f"void {mode}_{name}() {{"
        src += "\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += (f"  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n")
        src += "}\n"
    return src

def flatten_metas(metas_list: list) -> list:
    metas_list = [meta for name, meta in parser.kernels.items()]
    metas = []
    for meta in metas_list:
        metas.extend(meta)
    for meta in metas:
        try:
            assert meta.constexpr_arg_types == metas[0].constexpr_arg_types
            assert meta.constexpr_arg_names == metas[0].constexpr_arg_names
        except AssertionError:
            print((f"meta.constexpr_arg_types={meta.constexpr_arg_types}, meta.constexpr_arg_names={meta.constexpr_arg_names};"
                   f"metas[0].constexpr_arg_types={metas[0].constexpr_arg_types}, metas[0].constexpr_arg_names={metas[0].constexpr_arg_names}"
            ))
    return metas

def make_constexpr_dispatcher_header(kernels: dict) -> str:
    metas = flatten_metas(kernels.values())
    meta = metas[0]
    src = f"CUresult {meta.orig_kernel_name}_dispatcher_with_grid(CUstream stream, {gen_signature_with_full_args(meta)}, {gen_constexpr_signature_with_full_args(meta)}, unsigned int gridx, unsigned int gridy, unsigned int gridz);"
    return src

def make_constexpr_dispatcher(kernels: dict) -> str:
    metas = flatten_metas(kernels.values())
    meta = metas[0]
    src = f"CUresult {meta.orig_kernel_name}_dispatcher_with_grid(CUstream stream, {gen_signature_with_full_args(meta)}, {gen_constexpr_signature_with_full_args(meta)}, unsigned int gridx, unsigned int gridy, unsigned int gridz){{"
    src += "\n"
    # print(metas)

    conds_set = dict()
    conds_dict = dict()
    # conds_dict = dict()

    cond_fn = (  #
        lambda val, hint: f"({val} % {hint} == 0)"  #
        if hint >= 2 and hint % 2 == 0  #
        else f"({val} == {hint})"  #
        if hint == 1  #
        else None)

    # {
    #     "arg0": {
    #         [
    #             (priority=0, "arg0 == 1", [kernels])
    #             (priority=0, "arg0 == 4", [kernels])
    #             (priority=16, "arg0 % 16 == 0", [kernels])
    #         ]
    #     },
    #     "arg1"...
    # }

    conds_arg_select = dict()

    for arg_name in meta.constexpr_arg_names:
        conds_arg_select[arg_name] = defaultdict(list)
    for arg_name in meta.arg_names:
        conds_arg_select[arg_name] = defaultdict(list)


    for meta in metas:
        for arg_name, hint in zip(meta.arg_names, meta.sizes):
            if hint is not None:
                if hint >= 2 and hint % 2 == 0:
                    # TODO align 优先级
                    conds_arg_select[arg_name][(hint, f"({arg_name} % {hint} == 0)")].append(meta)
                else:
                    # constant 常量的优先级是 10000，最高优先级
                    # 注意，var_arg == 1 出现两次，还有一次在 constexpr 中
                    conds_arg_select[arg_name][(10000, f"({arg_name} == {hint})")].append(meta)
    
        for arg_name, value in zip(meta.constexpr_arg_names, meta.constexpr_arg_vals):
            conds_arg_select[arg_name][(10000, f"({arg_name} == {value})")].append(meta)
    
    for key in conds_arg_select.keys():
        conds_arg_select[key] = sorted(list(conds_arg_select[key].items()), reverse=True)

    conds_arg_select_list = list(conds_arg_select.items())
    conds_arg_select_list = sorted(conds_arg_select_list, key = lambda listx: len(listx[1]))

    conds_arg_select_keys = [item[0] for item in conds_arg_select_list]
    conds_arg_select_list = [item[1] for item in conds_arg_select_list]
    conds_arg_select_list = [item for item in conds_arg_select_list if len(conds_arg_select_list) > 0]
    
    def select_kernels(kernels: set, arg_i, cinst_list, matched_conds: list):
        # print(f">>>>>>>>>>>>>>>>>>>>>>> arg_i={arg_i} <<<<<<<<<<<<<<<<<<<<<<")
        # print(f"\n\n{str(kernels)}")
        # print(''.join(cinst_list))
        if len(kernels) == 0:
            cinst_list.append(
                f'{"  " * (arg_i+1)}printf("%s:%d No kernel match for arg {conds_arg_select_keys[arg_i-1]}! '
                f'Already matched: {", ".join(matched_conds[:-1]) if len(matched_conds) > 1 else "None"}\\n", __FILE__, __LINE__);\n'
            )
            return
            
        if arg_i == len(conds_arg_select_list):
            assert len(kernels) == 1, str(kernels)
            meta = list(kernels)[0]
            arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
            cinst_list.append(f"{'  ' * (arg_i+1)}return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}_with_grid(stream, {', '.join(arg_names)}, gridx, gridy, gridz);\n")
        else:
            
            # arg_info = conds_arg_select[conds_arg_select_keys[arg_i]]
            arg_info = conds_arg_select_list[arg_i]
            else_kernels = kernels.copy()
            for _i, cond_kernels in enumerate(arg_info):
                cond = cond_kernels[0][1]
                ckernels = cond_kernels[1]
                if_str = f"{'if' if _i == 0 else 'else if'}"
                cinst_list.append(f"{'  ' * (arg_i+1)}{if_str} ({cond}){{\n")
                select_kernels(kernels.intersection(ckernels), arg_i+1, cinst_list, matched_conds + [cond.replace('%', '%%')])
                cinst_list.append(f"{'  ' * (arg_i+1)}}} // {if_str} ({cond})\n")
                else_kernels = else_kernels.difference(ckernels)
            if_str = f"{'{' if len(arg_info) == 0 else 'else {'}"
            cinst_list.append(f"{'  ' * (arg_i+1)}{if_str} // {conds_arg_select_keys[arg_i]}\n")
            select_kernels(else_kernels, arg_i+1, cinst_list, matched_conds + [conds_arg_select_keys[arg_i]])
            cinst_list.append(f"{'  ' * (arg_i+1)}}} // {conds_arg_select_keys[arg_i]}\n")
            
    cinst_list = []
    select_kernels(set(metas), 0, cinst_list, [])
    src += ''.join(cinst_list)

    src += "\n"
    src += "  return CUDA_ERROR_INVALID_VALUE;\n"
    src += "}\n"

    return src

# generate dispatcher function for kernels with different meta-parameter and constant values
def make_kernel_meta_const_dispatcher(meta: KernelLinkerMeta) -> str:
    src = f"CUresult {meta.orig_kernel_name}(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id){{\n"
    src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels));\n"
    src += f"  return {meta.orig_kernel_name}_kernels[algo_id](stream, {', '.join(meta.arg_names)});\n"
    src += "}\n"
    
    src += f"CUresult {meta.orig_kernel_name}_with_grid(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id, unsigned int gridx, unsigned int gridy, unsigned int gridz){{\n"
    src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels_with_grid));\n"
    src += f"  return {meta.orig_kernel_name}_kernels_with_grid[algo_id](stream, {', '.join(meta.arg_names)}, gridx, gridy, gridz);\n"
    src += "}\n"

    return src


# generate definition of function pointers of kernel dispatchers based on meta-parameter and constant values
def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
    # the table of hint dispatchers
    src = f"typedef CUresult (*kernel_func_t)(CUstream stream, {gen_signature_with_full_args(meta)});\n"
    src += f"kernel_func_t {meta.orig_kernel_name}_kernels[] = {{\n"
    for name in names:
        src += f"  {name},\n"
    src += "};\n"

    src += "\n"
    src += f"typedef CUresult (*kernel_with_grid_func_t)(CUstream stream, {gen_signature_with_full_args(meta)}, unsigned int gridx, unsigned int gridy, unsigned int gridz);\n"
    src += f"kernel_with_grid_func_t {meta.orig_kernel_name}_kernels_with_grid[] = {{\n"
    for name in names:
        src += f"  {name}_with_grid,\n"
    src += "};\n"

    return src


# generate definition for load/unload functions for kernels with different meta-parameter and constant values
def make_kernel_load_def(names: str, meta: KernelLinkerMeta) -> str:
    src = ""
    for mode in ["load", "unload"]:
        src += f"void {mode}_{meta.orig_kernel_name}(void){{\n"
        for name in names:
            src += f"  {mode}_{name}();\n"
        src += "}\n\n"
    return src


def make_get_num_algos_decl(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
    return src


def make_get_num_algos_def(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
    src += f"  return (int)(sizeof({meta.orig_kernel_name}_kernels) / sizeof({meta.orig_kernel_name}_kernels[0]));\n"
    src += "}\n"
    return src


desc = """
Triton ahead-of-time linker:

This program takes in header files generated by compile.py, and generates a
single entry-point responsible for dispatching the user's input to the right
kernel given the specializations that were compiled.

Example usage:
python link.py /path/to/headers/*.h -o kernel_name
"""

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "headers",
        nargs="+",
        help="Paths to header files to link. Must include linker directive annotations (autogenerated by ttc)",
    )
    parser.add_argument("--out", "-o", type=Path, help="Out filename")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="String to prefix kernel dispatcher names",
    )
    args = parser.parse_args()

    # If the number of kernel is large, bash can not pass them together
    # So we need to support wildcards in python.
    if len(args.headers) == 1:
        if '*' in args.headers[0]:
            headers = glob.glob(args.headers[0])
    else:
        headers = args.headers

    # metadata
    parser = HeaderParser()
    includes = []
    for header in headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        parser.extract_linker_meta(h_str)

    # for name, meta in parser.kernels.items():
    #     print("==============================")
    #     print(f"name = {name}")
    #     print(f"meta = {meta}")

    # generate headers
    algo_decls = [make_algo_decls(name, meta) for name, meta in parser.kernels.items()]
    meta_lists = [meta for name, meta in parser.kernels.items()]
    meta = meta_lists[0][0]
    get_num_algos_decl = make_get_num_algos_decl(meta)
    global_decl = make_global_decl(meta)
    constexpr_dispatcher_header = make_constexpr_dispatcher_header(parser.kernels)
    with args.out.with_suffix(".h").open("w") as fp:
        out = "#include <cuda.h>\n"
        out += "#ifdef __cplusplus\n"
        out += 'extern "C"{\n'
        out += "#endif // __cplusplus\n"
        out += "\n".join(algo_decls)
        out += "\n"
        out += get_num_algos_decl
        out += "\n"
        out += global_decl
        out += "\n"
        out += constexpr_dispatcher_header
        out += "\n"
        out += "#ifdef __cplusplus\n"
        out += '}\n'
        out += "#endif // __cplusplus\n"
        fp.write(out)

    # generate source
    defs = [make_kernel_hints_dispatcher(name, meta) for name, meta in parser.kernels.items()]
    names = [name for name in parser.kernels.keys()]
    func_pointers_def = make_func_pointers(names, meta)
    meta_const_def = make_kernel_meta_const_dispatcher(meta)
    load_unload_def = make_kernel_load_def(names, meta)
    get_num_algos_def = make_get_num_algos_def(meta)
    default_algo_kernel = make_default_algo_kernel(meta)
    constexpr_dispatcher = make_constexpr_dispatcher(parser.kernels)
    with args.out.with_suffix(".c").open("w") as fp:
        out = ""
        out += "#include <cuda.h>\n"
        out += "#include <stdint.h>\n"
        out += "#include <assert.h>\n"
        out += "#include <stdio.h>\n"
        out += "\n"
        out += "\n".join(defs)
        out += "\n"
        out += func_pointers_def
        out += "\n"
        out += get_num_algos_def
        out += "\n"
        out += meta_const_def
        out += "\n"
        out += load_unload_def
        out += "\n"
        out += constexpr_dispatcher
        out += "\n"
        out += default_algo_kernel
        fp.write(out)
