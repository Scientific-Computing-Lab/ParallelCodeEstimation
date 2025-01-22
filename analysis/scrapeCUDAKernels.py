'''
This script will go through each of the source directories and use `grep`
to extract the files containing the kernels of interest.
Once the files are identified, we use some regex to pull out the kernels.
We then save all the extracted kernels to a JSON file for easy parsing.
'''

import os
import argparse
import pandas as pd
import glob
from pprint import pprint
import re
from tqdm import tqdm
import subprocess
import shlex
from io import StringIO
import numpy as np
import json
import clang.cindex
import json

# these will be used globally in this program
# mainly for consistency. They are absoule (full) paths
ROOT_DIR = ''
SRC_DIR = ''
BUILD_DIR = ''

def setup_dirs(buildDir, srcDir):
    global ROOT_DIR
    global SRC_DIR
    global BUILD_DIR

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
    # the list of dicts will later have run command information added to them
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
            target['kernelNames'].remove('testKernel')


    return targets


def get_kernel_names_from_target(target:dict):

    basename = target['basename']
    srcDir = target['src']

    cuobjdumpCommand = f'cuobjdump --list-text {BUILD_DIR}/{basename} | cu++filt'
    #print(shlex.split(cuobjdumpCommand))
    knamesResult = subprocess.run(cuobjdumpCommand, cwd=srcDir, shell=True, 
                                  timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert knamesResult.returncode == 0

    toRegex = knamesResult.stdout.decode('UTF-8')
    #print(target, 'toRegex', toRegex)

    matches = re.findall(r'(?<= : x-)[\w\-]*(?=\(.*\)\.sm_.*\.elf\.bin)', toRegex)

    # check if any matches are templated, so we drop the return type and angle brackets
    cleanNames = []
    for match in matches:
        if ('<' in match) or ('>' in match):
            cleanName = re.findall(r'(?<= ).*(?=<)', match)[0]
        else:
            cleanName = match
        cleanNames.append(cleanName)

    # if the program doesn't have any kernels defined in its source code
    # the matches list will be empty, indicating we should skip sampling
    # this program as the source code is usually some external private
    # library.
    return list(set(cleanNames))



def get_kernel_names(targets:list):
    assert len(targets) != 0
    for target in tqdm(targets, desc='Gathering kernel names'): 
        knames = get_kernel_names_from_target(target)
        target['kernelNames'] = knames
    return targets




def read_file_section(file_path, start_line, start_column, end_line, end_column):
    with open(file_path, 'r', errors='ignore') as file:
        #print(f'opening file {file_path} {start_line}:{end_line}')
        lines = file.readlines()
        # if for some reason they are asking for an index out of range
        if (start_line > len(lines)) | (end_line-1 > len(lines)):
            return ''
        elif start_line != end_line:
            lines = lines[start_line - 1:end_line]
            result = [lines[0][start_column-1:]]
            result = result + lines[1:len(lines)-1]
            result = result + [lines[-1][:end_column]]
            return ''.join(result).rstrip().lstrip()
        else:
            result = lines[start_line - 1]
            return result.rstrip().lstrip()


def search_ast_for_matches(cursor, kernelName:str):

    stack = [cursor]
    matches = []

    while len(stack) != 0:
        current = stack.pop()
        for child in current.get_children():
            stack.append(child)

        if (current.spelling == kernelName) & ((current.kind == clang.cindex.CursorKind.FUNCTION_DECL) | (current.kind == clang.cindex.CursorKind.FUNCTION_TEMPLATE)):
            matches.append(current)

    return matches


def get_source_lines_of_kernel(filename:str, kernelName:str):
    index = clang.cindex.Index.create()

    try:
        tu = index.parse(filename)
    except:
        # sometimes the files are not C/C++, this will skip these
        #print(f'error in clang reading file [{filename}]')
        return []

    #print(f'searching in file {filename}')
    cursor = tu.cursor
    matches = search_ast_for_matches(cursor, kernelName)

    # if we got multiple matches, it is often due to function declaration and then later definition
    kernelSources = []

    if len(matches) != None:
        for match in matches:
            start = match.extent.start
            end = match.extent.end
            kernelSource = read_file_section(filename, start.line, start.column, end.line, end.column)
            # sometimes the clang ast will give the wrong file lines? Not sure if it's expanding source lines unintentionally
            # instead we just check and make sure that the lines that were grabbed contain the kernel name
            if kernelName in kernelSource:
                kernelSources.append(kernelSource)

    return kernelSources



def find_files_with_kernel_name(dir:str, kernelName:str):

    command = f'grep -rl "{kernelName}"'
    grepOutput = subprocess.run(command, cwd=dir, 
                                  shell=True, timeout=300, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT)

    assert grepOutput.returncode == 0

    foundFiles = grepOutput.stdout.decode('UTF-8').replace('\n', ' ').split()

    # drop any that end in 'ncu-rep' or 'sqlite'
    foundFiles = [filename for filename in foundFiles if not(('-report.ncu-rep' in filename) | ('.sqlite' in filename))]

    result = [f'{dir}/{filename}' for filename in foundFiles]

    #print('foundFiles', foundFiles)
    return result 

def regex_search_for_kernel_in_file(filename:str, kernelName:str):


    return 

def gather_kernels(targets:list, outfileName:str):

    assert len(targets) != 0

    for target in tqdm(targets, desc='Scraping kernel source codes!'):
        basename = target['basename']
        parentName = basename[0:basename.find('-cuda')]
        kernelNames = target['kernelNames']
        srcDir = target['src']

        kernels = {}

        for kernelName in kernelNames:
            # search in the cuda directory to start
            finds = find_files_with_kernel_name(srcDir, kernelName)

            codeCandidates = []
            for filename in finds:
                srcCodes = get_source_lines_of_kernel(filename, kernelName)
                codeCandidates = codeCandidates + srcCodes

            # in the rare event that we can't find any code candidates 
            # (this happens to particles-cuda for some reason...)
            # we will grep search the function declaration candidates, include a couple lines before
            # then manually push/pop curly braces of the source file till we complete the
            # function text. From here we push this extracted text to a temporary file
            # for clang to parse
            if len(codeCandidates) == 0:
                print(f'skipping kernel {kernelName} in {basename}')
                continue

            assert len(codeCandidates) != 0, f'Unable to find source for kernel [{kernelName}] of [{basename}]'

            # for now, if there are multiple code candidates, we'll pick the biggest one
            selected = ''
            for srcCode in codeCandidates:
                if len(srcCode) > len(selected):
                    selected = srcCode

            #print(f'total code candidates for kernel: [{kernelName}] in [{basename}]: {len(codeCandidates)}')
            #print(f'biggest code candidate for kernel: [{kernelName}] in [{basename}]: \n[{selected}]')
            kernels[kernelName] = selected

        target['kernels'] = kernels


    return targets


def has_already_been_sampled(basename:str, kernelName:str, df:pd.DataFrame): 
    return False


def load_exisiting_data(targets:list, outfileName:str):

    if os.path.isfile(outfileName):
        with open(outfileName, 'r') as file:
            data = json.load(file) 
            assert type(data) == list

        # let's merge the two lists
        for dtarg in data:
            dname = dtarg['basename']
            for target in targets:
                name = target['basename']
                if dname == name: 
                    pass



    else:
        return targets

    return

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--buildDir', type=str, required=False, default='../build', help='Directory containing all the built executables')
    parser.add_argument('--srcDir', type=str, required=False, default='../src', help='Directory containing all the source files for the target executables')
    parser.add_argument('--outfile', type=str, required=False, default='../scraped-cuda-kernels.json', help='Output JSON file with extracted kernels')

    args = parser.parse_args()

    setup_dirs(args.buildDir, args.srcDir)

    # let's check if rodinia has been downloaded, if not, download it
    print('Starting CUDA kernel gathering process!')

    targets = get_runnable_targets()
    targets = get_kernel_names(targets)
    targets = modify_kernel_names_for_some_targets(targets)
    #targets = load_exisiting_data(targets, args.outfile)

    #targets = targets[0:100]

    results = gather_kernels(targets, args.outfile)

    # save the gathered kernels to a file
    with open(args.outfile, "w") as fp:
        json.dump(results, fp, indent=4) 

    pprint(results)

    return



if __name__ == "__main__":
    main()