'''
This script will go through each of the source directories and use `clang.cindex`
to extract the CUDA kernels and their called __device__ functions from the source code.
The extracted code is saved to a JSON file for easy parsing.
'''

import os
import argparse
import glob
from pprint import pprint
import re
from tqdm import tqdm
import subprocess
import shlex
import json
import clang.cindex
from clang.cindex import Cursor, CursorKind, StorageClass

# these will be used globally in this program
# mainly for consistency. They are absolute (full) paths
ROOT_DIR = ''
SRC_DIR = ''
BUILD_DIR = ''
LIBCLANG_PATH = ''

def setup_dirs(buildDir, srcDir, libclangPath):
    global ROOT_DIR
    global SRC_DIR
    global BUILD_DIR
    global LIBCLANG_PATH

    LIBCLANG_PATH = os.path.abspath(libclangPath)
    assert os.path.isfile(LIBCLANG_PATH)

    ROOT_DIR = os.path.abspath(f'{srcDir}/../')
    assert os.path.exists(ROOT_DIR)

    SRC_DIR = os.path.abspath(f'{srcDir}')
    BUILD_DIR = os.path.abspath(f'{buildDir}')

    assert os.path.exists(SRC_DIR)
    assert os.path.exists(BUILD_DIR)

    print('Using the following directories:')
    print(f'ROOT_DIR     = [{ROOT_DIR}]')
    print(f'SRC_DIR      = [{SRC_DIR}]')
    print(f'BUILD_DIR    = [{BUILD_DIR}]')

    return


def get_runnable_targets():
    # gather a list of dictionaries storing executable names and source directories
    files = glob.glob(f'{BUILD_DIR}/*')
    execs = []
    for entry in files:
        # check we have a file and it's an executable
        if os.path.isfile(entry) and os.access(entry, os.X_OK):
            basename = os.path.basename(entry)
            execSrcDir = os.path.abspath(f'{SRC_DIR}/{basename}')

            # check we have the source code too
            assert os.path.isdir(execSrcDir)

            execDict = {'basename':basename, 
                        'exe':entry, 
                        'src':execSrcDir }
            execs.append(execDict)

    return execs


def modify_kernel_names_for_some_targets(targets:list):
    for target in targets:
        basename = target['basename']

        if basename == 'assert-cuda':
            if 'testKernel' in target['kernelNames']:
                target['kernelNames'].remove('testKernel')
        if basename == 'atomicIntrinsics-cuda':
            print('atomicIntrinsics', target)

    return targets


def get_kernel_names_from_target(target:dict):

    basename = target['basename']
    srcDir = target['src']

    cuobjdumpCommand = f'cuobjdump --list-text {BUILD_DIR}/{basename} | cu++filt'
    knamesResult = subprocess.run(cuobjdumpCommand, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert knamesResult.returncode == 0

    toRegex = knamesResult.stdout.decode('UTF-8')

    matches = re.findall(r'(?<= : x-).*(?=\(.*\)\.sm_.*\.elf\.bin)', toRegex)

    # check if any matches are templated, so we drop the return type and angle brackets
    cleanNames = []
    for match in matches:
        if ('<' in match) or ('>' in match):
            parts = re.split(r'<|>', match)
            cleanName = parts[0].split()[-1] if ' ' in parts[0] else parts[0]
        else:
            cleanName = match
        cleanNames.append(cleanName)

    # deduplicate and return
    return list(set(cleanNames))


def get_kernel_names(targets:list):
    assert len(targets) != 0
    for target in tqdm(targets, desc='Gathering kernel names'): 
        knames = get_kernel_names_from_target(target)
        target['kernelNames'] = knames
    return targets


def read_file_section(file_path, start_line, start_column, end_line, end_column):
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()
        if start_line > len(lines) or end_line > len(lines):
            return ''
        start_line_idx = start_line - 1
        end_line_idx = end_line - 1
        if start_line_idx == end_line_idx:
            line = lines[start_line_idx]
            return line[start_column-1:end_column-1].strip()
        else:
            # First line
            code = [lines[start_line_idx][start_column-1:]]
            # Middle lines
            for i in range(start_line_idx + 1, end_line_idx):
                code.append(lines[i])
            # Last line
            code.append(lines[end_line_idx][:end_column-1])
            return ''.join(code).strip()

# this is used to remove any unneeded build arguments like `-c`
# that we don't need for static analysis
def process_compile_command(command_str):
    args = shlex.split(command_str)
    if not args:
        return []
    # Remove the compiler (nvcc, g++, etc.)
    args = args[1:]
    filtered_args = []
    i = 0
    source_file = None
    while i < len(args):
        arg = args[i]
        if arg == '-c':
            i += 1
        elif arg == '-o':
            i += 2
        elif arg.startswith(('-I', '-D', '-U')):
            filtered_args.append(arg)
            i += 1
        elif arg.startswith(('-std=', '--std=')):
            filtered_args.append(arg)
            i += 1
        elif arg.endswith(('.cu', '.c', '.cpp', '.cxx', '.cc', '.cuh', '.h')):
            source_file = arg
            i += 1
        else:
            # Skip other arguments (like -gencode, etc.)
            i += 1
    # Add -x cuda if source file is a .cu file
    if source_file and source_file.endswith('.cu'):
        filtered_args.extend(['-x', 'cuda'])
    return filtered_args


def extract_code_from_cursor(cursor):
    if not cursor.location.file:
        return ''
    file_path = cursor.location.file.name
    start_line = cursor.extent.start.line
    start_col = cursor.extent.start.column
    end_line = cursor.extent.end.line
    end_col = cursor.extent.end.column
    return read_file_section(file_path, start_line, start_col, end_line, end_col)



def is_cuda_kernel(cursor: Cursor) -> bool:
    """Check if a cursor represents a CUDA kernel (__global__ function)."""
    for child in cursor.get_children():
        if child.kind == CursorKind.CUDAGLOBAL_ATTR:
            return True
    return False

def is_device_function(cursor: Cursor) -> bool:
    """Check if a cursor represents a CUDA __device__ function."""
    for child in cursor.get_children():
        if child.kind == CursorKind.CUDADEVICE_ATTR:
            return True
    return False

def get_called_device_functions(global_cursor):
    device_cursors = set()
    visited = set()

    def _traverse(cursor):
        if cursor in visited:
            return
        visited.add(cursor)
        for child in cursor.get_children():
            if child.kind == CursorKind.CALL_EXPR:
                called_cursor = child.referenced
                if (called_cursor and 
                    called_cursor.kind == CursorKind.FUNCTION_DECL and
                    is_device_function(called_cursor) and 
                    called_cursor.is_definition()):
                    if called_cursor not in device_cursors:
                        device_cursors.add(called_cursor)
                        _traverse(called_cursor)
            _traverse(child)

    _traverse(global_cursor)
    return list(device_cursors)


def gather_kernels(targets):
    # assuming the compile_commands.json file is in the build_dir (where it gets built by default)
    compile_commands_path = os.path.join(BUILD_DIR, 'compile_commands.json')
    if not os.path.exists(compile_commands_path):
        raise FileNotFoundError(f"compile_commands.json not found in {BUILD_DIR}")
    
    with open(compile_commands_path, 'r') as f:
        compile_db = json.load(f)

    clang.cindex.Config.set_library_file(LIBCLANG_PATH)

    for target in tqdm(targets, desc='Extracting kernels'):
        src_dir = target['src']
        kernel_names = target['kernelNames']
        target['kernels'] = {}

        # Find relevant compile commands
        target_commands = []
        for entry in compile_db:
            file_path = os.path.abspath(entry['file'])
            if os.path.commonpath([file_path, src_dir]) == src_dir:
                target_commands.append(entry)

        # Parse each relevant source file
        for entry in target_commands:
            file_path = entry['file']
            args = process_compile_command(entry['command'])
            index = clang.cindex.Index.create()
            try:
                tu = index.parse(file_path, args=args)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                continue

            # Traverse AST for CUDA kernels (__global__ functions)
            for cursor in tu.cursor.walk_preorder():
                if (cursor.kind == CursorKind.FUNCTION_DECL and 
                    cursor.is_definition() and 
                    is_cuda_kernel(cursor)):
                    kernel_name = cursor.spelling
                    if kernel_name in kernel_names:
                        # Extract global function code
                        global_code = extract_code_from_cursor(cursor)
                        if not global_code:
                            continue

                        # Get called device functions
                        device_cursors = get_called_device_functions(cursor)
                        device_code = [extract_code_from_cursor(dc) for dc in device_cursors if extract_code_from_cursor(dc)]

                        all_code = [global_code] + device_code

                        # don't allow empty strings
                        all_code = [code for code in all_code if code]

                        # Deduplicate
                        unique_code = []
                        seen = set()
                        for code in all_code:
                            if code not in seen:
                                seen.add(code)
                                unique_code.append(code)

                        if kernel_name not in target['kernels']:
                            target['kernels'][kernel_name] = []
                        target['kernels'][kernel_name].extend(unique_code)

        # Deduplicate across multiple files
        for kernel in target['kernels']:
            unique = []
            seen = set()
            for code in target['kernels'][kernel]:
                if code not in seen:
                    seen.add(code)
                    unique.append(code)
            target['kernels'][kernel] = unique

    return targets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--buildDir', type=str, default='../build', help='Directory containing built executables')
    parser.add_argument('--srcDir', type=str, default='../src', help='Directory containing source files')
    parser.add_argument('--outfile', type=str, default='./scraped-cuda-kernels.json', help='Output JSON file')
    parser.add_argument('--libclangPath', type=str, default='/usr/lib/llvm-18/lib/libclang-18.so.1', help='Path to the libclang.so library file')

    args = parser.parse_args()

    setup_dirs(args.buildDir, args.srcDir, args.libclangPath)

    print('Starting CUDA kernel gathering process!')

    targets = get_runnable_targets()
    targets = get_kernel_names(targets)
    targets = modify_kernel_names_for_some_targets(targets)

    # Filter to only targets with '-cuda' in their basename
    targets = [t for t in targets if '-cuda' in t['basename']]

    results = gather_kernels(targets)

    # Convert to list of dicts for JSON serialization
    output = []
    for target in targets:
        entry = {
            'basename': target['basename'],
            'exe': target['exe'],
            'src': target['src'],
            'kernelNames': target['kernelNames'],
            'kernels': target['kernels']
        }
        output.append(entry)

    with open(args.outfile, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Saved results to {args.outfile}")

    return


if __name__ == "__main__":
    main()