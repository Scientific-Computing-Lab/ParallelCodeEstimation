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
from clang.cindex import Cursor, CursorKind, Config

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

    return targets


def get_kernel_names_from_target(target:dict):

    basename = target['basename']
    srcDir = target['src']

    cuobjdumpCommand = f'cuobjdump --list-text {BUILD_DIR}/{basename} | cu++filt'
    knamesResult = subprocess.run(cuobjdumpCommand, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert knamesResult.returncode == 0

    toRegex = knamesResult.stdout.decode('UTF-8')

    matches = re.findall(r'(?<= : x-)(?:void )?[\w\-]*(?=[\(\<].*[\)\>]\.sm_.*\.elf\.bin)', toRegex)

    # check if any matches are templated, so we drop the return type and angle brackets
    cleanNames = []
    for match in matches:
        if 'void ' in match:
            assert len(match.split()) == 2
            match = match.split()[1]
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
'''
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
        # if we have an options file, we need to read it in
        else:
            # Skip other arguments (like -gencode, etc.)
            i += 1
    # Add -x cuda if source file is a .cu file
    if source_file and source_file.endswith('.cu'):
        filtered_args.extend(['-x', 'cuda'])
    return filtered_args
'''

def extract_code_from_cursor(cursor):
    if not cursor.location.file:
        return ''
    # if the code we are trying to extract is from some external library
    # no within HeCBench, we're going to avoid adding it in
    if SRC_DIR not in os.path.abspath(cursor.location.file.name):
        return ''
    file_path = cursor.location.file.name
    start_line = cursor.extent.start.line
    start_col = cursor.extent.start.column
    end_line = cursor.extent.end.line
    end_col = cursor.extent.end.column
    return read_file_section(file_path, start_line, start_col, end_line, end_col)



def is_global_function(cursor: Cursor) -> bool:
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

# in cases of calls to 'atomicAdd' which are handled by an external CUDA library,
# the compiler thinks these are function declarations, each with their own unique
# declaration. We instead just check the spelling to be sure we only include
# the function once.
# TODO: this is technically overlooking some repeat overload definitions, no?
def is_cursor_in_set(cursor, set):
    for item in set:
        if item.spelling == cursor.spelling: 
            return True

# in the event that getOverloadedDecl doesn't return anything
# we will manually search for the definition
def find_overloads_manually(target_cursor):
    """Search the entire AST from `root_cursor` for overload candidates matching the name of `target_cursor`."""
    root_cursor = target_cursor._tu.cursor
    target_name = target_cursor.spelling
    overloads = []

    # Recursive AST traversal from the root
    def traverse(node):
        # Check if the node is a function/template with the target name
        if (
            node.spelling == target_name
            and (
                node.kind == CursorKind.FUNCTION_DECL
                or node.kind == CursorKind.FUNCTION_TEMPLATE
            )
        ):
            overloads.append(node)
        
        # Recurse into children
        for child in node.get_children():
            traverse(child)

    # Start traversal from the root
    traverse(root_cursor)
    return overloads


def get_overloaded_cursors_manual(cursor):
    index = 0
    config = Config()
    found = set()
    while True:
        overload_cursor = config.lib.clang_getOverloadedDecl(cursor, index)
        if not overload_cursor:
            break
        if not is_cursor_in_set(overload_cursor, found):
            found.add(overload_cursor)
        index += 1

    # if for some reason we can't find the overload, search
    # from the root of the TU
    if len(found) == 0:
        print(f'PERFORMING MANUAL SEARCH for {cursor.spelling}')
        found = find_overloads_manually(cursor)

    assert len(found) == 1
    return list(found)[0]


def get_called_device_functions(global_cursor):
    device_cursors = set()
    visited = set()

    # hoping we have enough stack space for this recursion
    def _traverse(cursor):
        if cursor in visited:
            return

        visited.add(cursor)

        # sometimes we get an OVERLOADED_DECL_REF which is a templated function
        # that clang hasn't yet resolved. Thus we'll need to manually find the
        # cursor it corresponds to. This cursor also has 0 children, so we need
        # to get the cursor it corresponds to, to continue searching.
        if cursor and cursor.kind == CursorKind.OVERLOADED_DECL_REF:
            #print(f"Overloaded reference {cursor.spelling} at {cursor.location}, num_children {len(list(cursor.get_children()))}")
            #print(f'USR [{cursor.displayname}]')
            ref_cursor = get_overloaded_cursors_manual(cursor)
            #print(f"  Candidate: {ref_cursor.spelling} (Kind: {ref_cursor.kind})")
            #print(f"- {ref_cursor.spelling} (type: {ref_cursor.type.spelling}) (location: {ref_cursor.location})")
            # continue traversing the referenced cursor, as it might make other __device__ calls
            if ref_cursor not in device_cursors:
                device_cursors.add(ref_cursor)
                _traverse(ref_cursor)

        for child in cursor.get_children():
            #if cursor.spelling == 'processSmallNet':
                #print('\tchild', child.spelling, child.kind)
            if child.kind == CursorKind.CALL_EXPR:
                called_cursor = child.referenced
                #if called_cursor:
                #    print('called cursor', called_cursor.spelling)
                if (called_cursor and 
                    (called_cursor.kind in [CursorKind.FUNCTION_DECL, CursorKind.FUNCTION_TEMPLATE]) and
                    (is_device_function(called_cursor) or is_global_function(called_cursor)) and 
                    called_cursor.is_definition()):
                    if called_cursor not in device_cursors:
                        device_cursors.add(called_cursor)
                        _traverse(called_cursor)
            _traverse(child)

    _traverse(global_cursor)

    return list(device_cursors)

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
        # some of the build commands have an options file with extra
        # build flags that we need to take into account
        elif arg == '--options-file':
            # Read the options file
            options_file = args[i+1]
            try:
                options_file_full_path = os.path.join(BUILD_DIR, options_file)
                with open(options_file_full_path, 'r') as f:
                    content = f.read()
                    # Split into arguments using shlex to handle quotes, spaces, etc.
                    options = shlex.split(content)
                    filtered_args.extend(options)
            except Exception as e:
                print(f"Error processing options file {options_file_full_path}: {e}")
            i += 2  # Skip the filename
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

    # force the compiler to resolve OVERLOAD_DECL_REFs to functions
    # this doesn't seem to change anything though :(
    filtered_args.extend(['-fno-delayed-template-parsing'])
    return filtered_args


def gather_kernels(targets):
    compile_commands_path = os.path.join(BUILD_DIR, 'compile_commands.json')
    if not os.path.exists(compile_commands_path):
        raise FileNotFoundError(f"compile_commands.json not found in {BUILD_DIR}")
    
    with open(compile_commands_path, 'r') as f:
        compile_db = json.load(f)

    Config.set_library_file(LIBCLANG_PATH)

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

        # Parse each relevant source file and its includes
        for entry in target_commands:
            file_path = entry['file']
            args = process_compile_command(entry['command'])
            print('args', args)
            index = clang.cindex.Index.create()
            processed_files = set()  # Track processed files to avoid duplicates

            # Process main file
            if file_path not in processed_files:
                processed_files.add(file_path)
                try:
                    tu = index.parse(file_path, args=args)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    continue

                # Process main TU
                process_translation_unit(tu, target, kernel_names, src_dir)

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


def process_translation_unit(tu, target, kernel_names, src_dir):
    # Traverse AST for CUDA kernels (__global__ functions) and device functions
    for cursor in tu.cursor.walk_preorder():
        #print('are we actually walking this?')
        if (cursor.kind in [CursorKind.FUNCTION_DECL, CursorKind.FUNCTION_TEMPLATE] and 
            cursor.is_definition() and 
            is_global_function(cursor) and
            cursor.spelling in kernel_names):

            #print('got a hit!', cursor.spelling)

            kernel_name = cursor.spelling
            # Extract global function code
            global_code = extract_code_from_cursor(cursor)
            if not global_code:
                continue

            # Get called device functions
            device_cursors = get_called_device_functions(cursor)

            # if the code we extract is a nonempty string, keep it
            # when the code we're pulling is from outside the HeCBench files, we skip adding it
            device_code = [extract_code_from_cursor(dc) for dc in device_cursors if extract_code_from_cursor(dc)]

            all_code = [global_code] + device_code
            all_code = [code for code in all_code if code]

            # Deduplicate
            unique_code = []
            seen_code = set()
            for code in all_code:
                if code not in seen_code:
                    seen_code.add(code)
                    unique_code.append(code)

            if kernel_name not in target['kernels']:
                target['kernels'][kernel_name] = []
            target['kernels'][kernel_name].extend(unique_code)


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

    #targets = [targets[281]]
    #pprint(targets)

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