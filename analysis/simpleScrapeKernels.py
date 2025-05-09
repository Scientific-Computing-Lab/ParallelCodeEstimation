'''
This is a simple script designed to go through the source files
of each target program and extracts the C/C++ files with the 
target kernel definition, declarations, and invocations. 
It also extracts the file that defines main().
These files get collated together and added to a scraped 
kernels database.

The purpose of this approach is that it's simple to implement,
and will give us a baseline on whether or not we need to cut
down the context for inference/training.
We later have a script that's going to visualize the scraped
data and drop any samples that have high token counts.
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


# these will be used globally in this program
# mainly for consistency. They are absolute (full) paths
ROOT_DIR = ''
SRC_DIR = ''
BUILD_DIR = ''


def setup_dirs(buildDir, srcDir): #libclangPath):
    global ROOT_DIR
    global SRC_DIR
    global BUILD_DIR

    #LIBCLANG_PATH = os.path.abspath(libclangPath)
    #assert os.path.isfile(LIBCLANG_PATH)

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
    files = sorted(glob.glob(f'{BUILD_DIR}/*'))
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

def get_cuobjdump_kernel_SASS(target, mangledKernelName):
    basename = target['basename']
    srcDir = target['src']

    # if we have a CUDA code being requested
    if not ('omp_offloading' in mangledKernelName):
        cuobjdumpCommand = f'cuobjdump --function={mangledKernelName} --dump-sass {BUILD_DIR}/{basename}'
        #print(shlex.split(cuobjdumpCommand))
        commandResult = subprocess.run(cuobjdumpCommand, cwd=srcDir, shell=True, 
                                      timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        sassCode = commandResult.stdout.decode('UTF-8')

        assert commandResult.returncode == 0, f"ERROR: input executable might not contain device code:\n[{sassCode}]"
    # llvm-objcopy --dump-section=.llvm.offloading=offload ../build/meanshift-omp
    # clang-offload-packager-18 offload --image=file=output.o
    # cuobjdump --function=funcName --dump-sass output.o
    # otherwise we have an OMP kernel we want to extract SASS from
    else:
        firstCommand = f'llvm-objcopy --dump-section=.llvm.offloading=offload {BUILD_DIR}/{basename}'
        commandResult = subprocess.run(firstCommand, cwd=f'{ROOT_DIR}/analysis', shell=True, 
                                      timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        resultText = commandResult.stdout.decode('UTF-8')
        assert commandResult.returncode == 0, f"ERROR: [{resultText}]"

        # may need to change this to be generic clang-offload-packager
        secondCommand = f'clang-offload-packager-18 offload --image=file=output.o'
        commandResult = subprocess.run(secondCommand, cwd=f'{ROOT_DIR}/analysis', shell=True, 
                                      timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        resultText = commandResult.stdout.decode('UTF-8')
        assert commandResult.returncode == 0, f"ERROR: [{resultText}]"

        thirdCommand = f'cuobjdump --function={mangledKernelName} --dump-sass output.o'
        commandResult = subprocess.run(thirdCommand, cwd=f'{ROOT_DIR}/analysis', shell=True, 
                                      timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        resultText = commandResult.stdout.decode('UTF-8')
        assert commandResult.returncode == 0, f"ERROR: [{resultText}]"

        sassCode = resultText

    assert 'Function : ' in sassCode
    assert mangledKernelName in sassCode

    return sassCode

    ##print(target, 'toRegex', toRegex)

    #reMatches = re.finditer(r'((?<= : x-)|(?<= : x-void ))[\w\-\:]*(?=[\(\<].*[\)\>](?:(?:(?:(?:\.sm_)|(?: \(\.sm_)).*\.elf\.bin)|(?: \[clone)))', toRegex, re.MULTILINE)

    #matches = [m.group() for m in reMatches]

    ## keep non-empty matches
    #matches = [m for m in matches if m]

    #return matches


def get_cuobjdump_kernels_demangled(target, filter='cu++filt'):
    basename = target['basename']
    srcDir = target['src']

    cuobjdumpCommand = f'cuobjdump --list-text {BUILD_DIR}/{basename} | {filter}'
    #print(shlex.split(cuobjdumpCommand))
    knamesResult = subprocess.run(cuobjdumpCommand, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert knamesResult.returncode == 0

    toRegex = knamesResult.stdout.decode('UTF-8')
    #print(target, 'toRegex', toRegex)

    # this step will drop some of the undesirable kernels
    reMatches = re.finditer(r'((?<= : x-)|(?<= : x-void ))[\w\-\:]*(?=[\(\<].*[\)\>](?:(?:(?:(?:\.sm_)|(?: \(\.sm_)).*\.elf\.bin)|(?: \[clone)))', toRegex, re.MULTILINE)

    matches = [m.group() for m in reMatches]

    # keep non-empty matches
    matches = [m for m in matches if m]

    return matches

def get_cuobjdump_kernels_mangled(target):
    basename = target['basename']
    srcDir = target['src']

    cuobjdumpCommand = f'cuobjdump --list-text {BUILD_DIR}/{basename}'
    #print(shlex.split(cuobjdumpCommand))
    knamesResult = subprocess.run(cuobjdumpCommand, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    toRegex = knamesResult.stdout.decode('UTF-8')

    if 'does not contain device code' in toRegex:
        return []
    else:
        assert knamesResult.returncode == 0, f"ERROR: {toRegex}"

    #print(target, '\ntoRegex', toRegex)

    reMatches = re.finditer(r'((?<= : x-)|(?<= : x-void ))[\w\-\:]*(?=(?:(?:(?:(?:\.sm_)|(?: \(\.sm_)).*\.elf\.bin)|(?: \[clone)))', toRegex, re.MULTILINE)

    matches = [m.group() for m in reMatches]

    # keep non-empty matches
    matches = [m for m in matches if m]

    return matches


def get_cuobjdump_kernels(target, mangled=False):
    basename = target['basename']
    cleanNames = list()

    if mangled:
        cleanNames = get_cuobjdump_kernels_mangled(target)
    else:
        matches = get_cuobjdump_kernels_demangled(target, 'cu++filt')

        # if there are no matches, it's sometimes that the mangled names
        # are not unmangleable by cu++filt, so we use c++filt or llvm-cxxfilt
        if len(matches) == 0:
            matches = get_cuobjdump_kernels_demangled(target, 'c++filt')

            if len(matches) == 0:
                matches = get_cuobjdump_kernels_demangled(target, 'llvm-cxxfilt')

                if len(matches) == 0:
                    assert not does_grep_show_global_defs(target), f'__global__ defs exist for {basename}, but they are NOT in compiled executable'

        # this is something we need to consider later
        # this function should really return the demangled names with all the templating and class syntax
        # check if any matches are templated, so we drop the return type and angle brackets
        for match in matches:
            # any kernel that's actually a library function, we omit
            if ('cub::' in match):
                continue
            if ('<' in match) or ('>' in match):
                parts = re.split(r'<|>', match)
                cleanName = parts[0].split()[-1] if ' ' in parts[0] else parts[0]
            else:
                cleanName = match

            # if the name has any :: let's grab the last element
            if ('::' in cleanName):
                cleanName = cleanName.split('::')[-1]

            cleanNames.append(cleanName)

    return list(set(cleanNames))


def get_objdump_kernels_mangled(target):
    basename = target['basename']
    srcDir = target['src']

    objdumpCommand = f'objdump -t --section=omp_offloading_entries {BUILD_DIR}/{basename}'
    knamesResult = subprocess.run(objdumpCommand, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert knamesResult.returncode == 0

    toRegex = knamesResult.stdout.decode('UTF-8')

    #matches = re.findall(r'(?<=\.omp_offloading\.entry\.)(__omp_offloading.*)(?=\n)', toRegex)
    reMatches = re.finditer(r'(?<=\.omp_offloading\.entry\.)(__omp_offloading.*)(?=\n)', toRegex, re.MULTILINE)

    matches = [m.group() for m in reMatches]

    # keep non-empty matches
    matches = [m for m in matches if m]
    # all the OMP codes should have at least one offload region
    assert len(matches) != 0

    return matches


def get_objdump_kernels_demangled(target):
    matches = get_objdump_kernels_mangled(target)

    assert does_grep_show_omp_pragmas(target), f"{target['basename']} doesn't have any target regions!"

    # when we build for OpenMP, there's a section with all the kernel names
    cleanNames = list(set(matches))
    cleanNames = [demangle_omp_kernel_name(name) for name in cleanNames]

    return list(set(cleanNames))
    

def get_objdump_kernels(target, mangled=True):
    if mangled:
        return get_objdump_kernels_mangled(target)
    else:
        return get_objdump_kernels_demangled(target)


# technically this could give a false negative
# because a kernel may be pragmaed out at build time
# but this would say some kernels do exist
def does_grep_show_global_defs(target):
    srcDir = target['src']

    command = f'grep -rni "__global__"'
    grep_results = subprocess.run(command, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    # we get a return code of 1 if no matches are found
    assert grep_results.returncode == 0 or grep_results.returncode == 1

    returnData = grep_results.stdout.decode('UTF-8').strip()

    # returns True if not empty, False if empty
    return (returnData != '')


# simple sanity check to make sure we actually have an omp program
# that can be offloaded to the GPU
def does_grep_show_omp_pragmas(target):
    srcDir = target['src']

    command = f'grep -rni "#pragma.*omp.*target"'
    grep_results = subprocess.run(command, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    # we get a return code of 1 if no matches are found
    assert grep_results.returncode == 0 or grep_results.returncode == 1

    returnData = grep_results.stdout.decode('UTF-8').strip()

    # returns True if not empty, False if empty
    return (returnData != '')

def demangle_cuda_kernel_name(mangledName):

    filterCommand = f'llvm-cxxfilt {mangledName}'
    
    #print(shlex.split(cuobjdumpCommand))
    demangleResult = subprocess.run(filterCommand, shell=True, timeout=5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert demangleResult.returncode == 0
    
    demangled = demangleResult.stdout.decode('UTF-8').strip()

    return demangled


def demangle_omp_kernel_name(mangledName):

    regex = r'_(?:[^_]+_){4}(.*)(_l[\d]+)'
    matches = re.finditer(regex, mangledName, re.MULTILINE)

    matches = [i for i in matches]
    assert len(matches) == 1

    cleanName = ''
    for match in matches:
        groups = match.groups()
        assert len(groups) == 2
        cleanName = groups[0]
        break

    filterCommand = f'llvm-cxxfilt {cleanName}'
    
    #print(shlex.split(cuobjdumpCommand))
    demangleResult = subprocess.run(filterCommand, shell=True, timeout=5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert demangleResult.returncode == 0
    
    demangled = demangleResult.stdout.decode('UTF-8').strip()

    return demangled


def get_kernel_names_from_target(target:dict, mangled=False):

    basename = target['basename']

    cleanNames = list()

    #print('getting kernel names for', basename)
    if '-cuda' in basename:
        # preserving original behavior with demangled names
        cleanNames = get_cuobjdump_kernels(target, mangled=False)

    # it's an OMP program, need to use regular objdump
    # these names are going to be mangled
    else:
        # to keep default behavior, we're keeping them mangled for now
        cleanNames = get_objdump_kernels(target, mangled=True)

    #assert len(matches) != 0
    # if the program doesn't have any kernels defined in its source code
    # the matches list will be empty, indicating we should skip sampling
    # this program as the source code is usually some external private
    # library.
    return cleanNames


def get_kernel_names(targets:list):
    assert len(targets) != 0
    for target in tqdm(targets, desc='Gathering kernel names'): 
        knames = get_kernel_names_from_target(target)
        if len(knames) == 0:
            bname = target['basename']
            print(f'{bname} DOES NOT HAVE ANY KERNELS!')
        target['kernelNames'] = knames
    return targets


def get_kernel_sass_codes(targets:list):
    assert len(targets) != 0
    for target in tqdm(targets, desc='Gathering kernel SASS codes!'): 
        basename = target['basename']
        target['sass'] = {}
        if '-cuda' in basename:
            #print(basename)
            mangledNames = get_cuobjdump_kernels_mangled(target)
            kernelNames = target['kernelNames']

            #print('mangledNames', mangledNames)
            #print('kernelNames', target['kernelNames'])
            # if we didn't grab any kernels on the kernelName pass, don't try to get the SASS
            if len(kernelNames) == 0:
                continue
            #if len(mangledNames) == 0:
            #    print(f'{basename} DOES NOT HAVE ANY KERNELS!')
            #assert len(mangledNames) != 0
            for mkname in mangledNames:
                # filter out any mangled names that don't appear in the kernelNames
                omitThisName = True
                dkname = demangle_cuda_kernel_name(mkname)
                for kname in kernelNames:
                    if kname in dkname:
                        omitThisName = False

                if omitThisName:
                    continue
                sassCode = get_cuobjdump_kernel_SASS(target, mkname)
                target['sass'][f'{dkname}'] = sassCode 
        else:
            #print(basename)
            mangledNames = get_objdump_kernels_mangled(target)
            #print('mangledNames', mangledNames)
            assert len(mangledNames) != 0
            for mkname in mangledNames:
                sassCode = get_cuobjdump_kernel_SASS(target, mkname)
                #dkname = demangle_omp_kernel_name(mkname)
                target['sass'][f'{mkname}'] = sassCode 

    return targets


def find_files_with_extension(dir, ext):
    filenames = list(sorted(glob.glob(f'{dir}/**/*.{ext}', recursive=True)))
    return filenames


def amalgamate_files_into_string(files):
    joined = ''
    for srcFile in files:
        filename = os.path.basename(srcFile)
        #print(f'working on file {srcFile}')
        with open(srcFile, 'r', encoding='utf8', errors='replace') as file:
            data = file.read()
            joined += f'-----------------------------------\n'
            joined += f'{filename}\n'
            joined += f'-----------------------------------\n'
            joined += f'{data}\n\n'
    return joined



def gather_kernels_simple(targets):

    for target in tqdm(targets, desc='Extracting kernels'):
        src_dir = target['src']
        kernel_names = target['kernelNames']
        target['kernels'] = {}

        # grab all the source files (.cu, .c, .cpp, .cuh, .cc, .h, .hpp)
        exts = ['cu', 'c', 'cc', 'h', 'hpp', 'cpp', 'cxx', 'cuh']
        found = [find_files_with_extension(src_dir, ext) for ext in exts]
        files = []
        [files.extend(flist) for flist in found]

        # remove any duplicate files that may come up
        files = list(set(files))

        # amalgamate all the source files and set it as the "context" of a kernel
        # if a program has multiple kernels, they will all have the same context
        joined = amalgamate_files_into_string(files)

        basename = target['basename']
        for kern in kernel_names:
            target['kernels'][kern] = joined
            if '-cuda' in basename:
                assert kern in joined, f'[{target["basename"]}][{kern}] Kernel not found in any of the pulled source code!'
            # for now, we're skipping the check on omp codes in making sure
            # the kernels actually exist
            

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--buildDir', type=str, default='../build', help='Directory containing built executables')
    parser.add_argument('--srcDir', type=str, default='../src', help='Directory containing source files')
    parser.add_argument('--outfile', type=str, default='./simple-scraped-kernels-with-sass.json', help='Output JSON file')
    parser.add_argument('--skipSASS', action='store_true', default=False, help='Leave out SASS from the scrape')

    #parser.add_argument('--libclangPath', type=str, default='/usr/lib/llvm-18/lib/libclang-18.so.1', help='Path to the libclang.so library file')

    args = parser.parse_args()

    setup_dirs(args.buildDir, args.srcDir)#, args.libclangPath)

    print('Starting CUDA kernel gathering process!')

    targets = get_runnable_targets()
    targets = get_kernel_names(targets)
    targets = modify_kernel_names_for_some_targets(targets)

    if not args.skipSASS:
        targets = get_kernel_sass_codes(targets)

    # Filter to only targets with '-cuda' in their basename
    #targets = [t for t in targets if '-omp' in t['basename']]

    #targets = [targets[281]]
    #pprint(targets)

    gather_kernels_simple(targets)

    # Convert to list of dicts for JSON serialization
    #output = []
    #for target in targets:
    #    entry = {
    #        'basename': target['basename'],
    #        'exe': target['exe'],
    #        'src': target['src'],
    #        'kernelNames': target['kernelNames'],
    #        'kernels': target['kernels']
    #    }
    #    output.append(entry)

    with open(args.outfile, 'w') as f:
        json.dump(targets, f, indent=4)

    print(f"Saved results to {args.outfile}")

    return


if __name__ == "__main__":
    main()