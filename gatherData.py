'''
This script is to be executed after all the codes have been built
into the ./build dir of this project.

Here we will run each of the codes using the default `make run` arguments.
We were able to build at least 400 codes, so we will rip their `make run` arguments
and then execute them with the same args.

We will use `ncu` to record the roofline data of each run.

We don't really put much error checking in here because we are mainly trying to 
get something that works. Later on we may build this program up some more.
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
import csv

# these will be used globally in this program
# mainly for consistency. They are absoule (full) paths
DOWNLOAD_DIR = ''
ROOT_DIR = ''
SRC_DIR = ''
BUILD_DIR = ''

def setup_dirs(buildDir, srcDir):
    global DOWNLOAD_DIR
    global ROOT_DIR
    global SRC_DIR
    global BUILD_DIR

    ROOT_DIR = os.path.abspath(f'{srcDir}/../')
    assert os.path.exists(ROOT_DIR)

    DOWNLOAD_DIR = os.path.abspath(f'{ROOT_DIR}/downloads')

    if not os.path.exists(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)

    SRC_DIR = os.path.abspath(f'{srcDir}')
    BUILD_DIR = os.path.abspath(f'{buildDir}')

    assert os.path.exists(DOWNLOAD_DIR)
    assert os.path.exists(SRC_DIR)
    assert os.path.exists(BUILD_DIR)

    print('Using the following directories:')
    print(f'ROOT_DIR     = [{ROOT_DIR}]')
    print(f'DOWNLOAD_DIR = [{DOWNLOAD_DIR}]')
    print(f'SRC_DIR      = [{SRC_DIR}]')
    print(f'BUILD_DIR    = [{BUILD_DIR}]')

    return

def has_rodinia_datasets():
    return os.path.isdir(f'{SRC_DIR}/data')

# because so many codes depend on rodinia, it gets its own setup function
def download_rodinia_and_extract():
    print('Downloading Rodinia Data...')

    command = f'wget http://www.cs.virginia.edu/~skadron/lava/Rodinia/Packages/rodinia_3.1.tar.bz2 && tar -xf ./rodinia_3.1.tar.bz2 rodinia_3.1/data && mv ./rodinia_3.1/data {SRC_DIR}/data'
    print('executing command', command)
    result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)

    assert result.returncode == 0
    assert has_rodinia_datasets()

    print('Rodinia download and unzip complete!')

    return

def run_setup_command_for_file(targetFile, command, targetDir=DOWNLOAD_DIR):
    if not os.path.isfile(targetFile):
        result = subprocess.run(args=command, cwd=targetDir, shell=True)
        assert result.returncode == 0
    return


def download_files_for_some_targets(targets):
    for target in targets:
        basename = target['basename']
        srcDir = target['src']

        if basename == 'gc-cuda':
            tFile = f'{srcDir}/../mis-cuda/internet.egr'
            command = f'wget --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/internet.egr && mv ./internet.egr {srcDir}/../mis-cuda/'
            run_setup_command_for_file(tFile, command)

        elif 'bitcracker' in basename:
            tFile = os.path.normpath(f'{srcDir}/../bitcracker-cuda/hash_pass/img_win8_user_hash.txt')
            command = f'wget --no-check-certificate https://raw.githubusercontent.com/oneapi-src/Velocity-Bench/refs/heads/main/bitcracker/hash_pass/img_win8_user_hash.txt && mv ./img_win8_user_hash.txt {tFile}'
            if not os.path.isfile(tFile):
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0
            # idk why the function isn't working, just doing it manually here for now
            #run_setup_command_for_file(tFile, command)

            tFile = os.path.normpath(f'{srcDir}/../bitcracker-cuda/hash_pass/user_passwords_60000.txt')
            command = f'wget --no-check-certificate https://raw.githubusercontent.com/oneapi-src/Velocity-Bench/refs/heads/main/bitcracker/hash_pass/user_passwords_60000.txt && mv ./user_passwords_60000.txt {tFile}'
            if not os.path.isfile(tFile):
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0
            #run_setup_command_for_file(tFile, command)

        elif 'logic-rewrite' in basename:
            tFile = os.path.normpath(f'{srcDir}/benchmarks/arithmetic/hyp.aig')
            command = f'git clone https://github.com/lsils/benchmarks ./benchmarks'
            if not os.path.isfile(tFile):
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0
            #run_setup_command_for_file(tFile, command)

        elif 'permutate' in basename:
            tFile = os.path.normpath(f'{srcDir}/../permutate-cuda/test_data/truerand_1bit.bin')
            command = f'git clone https://github.com/yeah1kim/yeah_GPU_SP800_90B_IID ./permutate-cuda && cp -r ./permutate-cuda/test_data {srcDir}/../permutate-cuda/test_data'
            if not os.path.isfile(tFile):
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0


        elif basename == 'cc-cuda':
            if not os.path.isfile(f'{srcDir}/delaunay_n24.egr'):
                command = f'wget --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/delaunay_n24.egr && mv ./delaunay_n24.egr {srcDir}/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif basename == 'mriQ-cuda':
            if not os.path.isfile(f'{srcDir}/datasets/128x128x128/input/128x128x128.bin'):
                command = f'wget --no-check-certificate https://www.cs.ucr.edu/~nael/217-f19/labs/mri-q.tgz && tar -xf ./mri-q.tgz mri-q/datasets && mv ./mri-q/datasets {srcDir}/datasets'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif 'gd-' in basename:
            if not os.path.isfile(f'{srcDir}/gisette_scale'):
                command = f'wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2 && bzip2 -dk ./gisette_scale.bz2 && mv ./gisette_scale {srcDir}/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif 'lanczos-' in basename:
            if not os.path.isfile(f'{srcDir}/data/social-large-800k.txt'):
                command = f'python3 gengraph.py'
                result = subprocess.run(command, cwd=f'{srcDir}/data', shell=True)
                assert result.returncode == 0

        elif basename == 'svd3x3-cuda':
            if not os.path.isfile(f'{srcDir}/Dataset_1M.txt'):
                command = f'wget --no-check-certificate https://raw.githubusercontent.com/kuiwuchn/3x3_SVD_CUDA/refs/heads/master/svd3x3/svd3x3/Dataset_1M.txt && mv ./Dataset_1M.txt {srcDir}/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif basename == 'tsp-cuda':
            if not os.path.isfile(f'{srcDir}/d493.tsp'):
                command = f'wget --no-check-certificate http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/d493.tsp.gz && gunzip ./d493.tsp.gz && mv ./d493.tsp {srcDir}/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif basename == 'hogbom-cuda':
            command = f'mkdir -p ./data'
            result = subprocess.run(command, cwd=srcDir, shell=True)
            assert result.returncode == 0

            if not os.path.isfile(f'{srcDir}/data/dirty_4096.img'):
                command = f'wget --no-check-certificate https://github.com/ATNF/askap-benchmarks/raw/refs/heads/master/data/dirty_4096.img && mv ./dirty_4096.img {srcDir}/data/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

            if not os.path.isfile(f'{srcDir}/data/psf_4096.img'):
                command = f'wget --no-check-certificate https://github.com/ATNF/askap-benchmarks/raw/refs/heads/master/data/psf_4096.img && mv ./psf_4096.img {srcDir}/data/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif basename == 'lr-cuda':
            if not os.path.isfile(f'{srcDir}/assets/data2_arrival.txt'):
                command = f'cp ../lr-sycl/assets.tar.gz ./ && tar -xf assets.tar.gz'
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0

        elif (basename == 'haversine-cuda') or (basename == 'geodesic-cuda'):
            if not os.path.isfile(f'{srcDir}/../geodesic-sycl/locations.txt'):
                command = f'tar -xf ../geodesic-sycl/locations.tar.gz'
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0

        elif basename == 'word2vec-cuda':
            if not os.path.isfile(f'{srcDir}/text8'):
                command = f'wget --no-check-certificate https://mattmahoney.net/dc/text8.zip && unzip ./text8.zip && mv ./text8 {srcDir}/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif basename == 'multimaterial-cuda':
            if not os.path.isfile(f'{srcDir}/volfrac.dat'):
                command = f'tar -xf volfrac.dat.tgz'
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0

        elif basename == 'mcpr-cuda':
            if not os.path.isfile(f'{srcDir}/alphas.csv'):
                command = f'bunzip2 alphas.csv.bz2'
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0

        elif basename == 'sa-cuda':
            if not os.path.isfile(f'{srcDir}/genome.txt'):
                command = f'wget --no-check-certificate https://github.com/gmzang/Parallel-Suffix-Array-on-GPU/raw/refs/heads/master/dc3_gpu/genome.txt && mv ./genome.txt {srcDir}/'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        elif basename == 'mpc-cuda':
            if not os.path.isfile(f'{srcDir}/msg_sp.trace.fpc'):
                command = f'wget --no-check-certificate http://www.cs.txstate.edu/~burtscher/research/datasets/FPdouble/msg_sp.trace.fpc && mv ./msg_sp.trace.fpc {srcDir}/ && cd {srcDir} && make process'
                result = subprocess.run(command, cwd=DOWNLOAD_DIR, shell=True)
                assert result.returncode == 0

        # need to have python2 installed for this to work
        elif basename == 'heat2d-cuda':
            if not os.path.isfile(f'{srcDir}/data.txt'):
                command = f'python2 mkinit.py 4096 8192 data.txt'
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0

        elif (basename == 'frna-cuda') or (basename == 'prna-cuda'):
            if not os.path.isfile(f'{srcDir}/../prna-cuda/HIV1-NL43.seq'):
                command = f'tar -xf ./HIV1-NL43.tar.gz && tar -xf ./data_tables.tar.gz'
                result = subprocess.run(command, cwd=f'{srcDir}/../prna-cuda', shell=True)
                assert result.returncode == 0

        elif basename == 'multimaterial-omp':
            if not os.path.isfile(f'{srcDir}/../multimaterial-cuda/volfrac.dat'):
                command = f'tar -xzf ./volfrac.dat.tgz' 
                result = subprocess.run(command, cwd=f'{srcDir}/../multimaterial-cuda', shell=True)
                assert result.returncode == 0
            if not os.path.isfile(f'{srcDir}/../multimaterial-omp/volfrac.dat'):
                command = f'cp ./volfrac.dat ../multimaterial-omp/' 
                result = subprocess.run(command, cwd=f'{srcDir}/../multimaterial-cuda', shell=True)
                assert result.returncode == 0

        elif (basename == 'grep-cuda') or (basename == 'grep-omp'):
            if not os.path.exists(f'{srcDir}/../grep-cuda/testcases'):
                command = f'tar -xf ./testcase.tar.gz' 
                result = subprocess.run(command, cwd=f'{srcDir}/../grep-cuda', shell=True)
                assert result.returncode == 0
        elif (basename == 'd2q9-bgk-omp') or (basename == 'd2q9-bgk-cuda'):
            if not os.path.exists(f'{srcDir}/Inputs/input_256x256.params'):
                command = f'tar -xf ./test.tar.gz' 
                result = subprocess.run(command, cwd=f'{srcDir}', shell=True)
                assert result.returncode == 0
        elif (basename == 'lr-omp'):
            if not os.path.exists(f'{srcDir}/assets/house.txt'):
                command = f'cp ../lr-sycl/assets.tar.gz ./ && tar -xf ./assets.tar.gz' 
                result = subprocess.run(command, cwd=f'{srcDir}', shell=True)
                assert result.returncode == 0

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




def get_exec_command_from_makefile(makefile):
    assert os.path.isfile(makefile)

    with open(makefile, 'r') as file:
        data = file.read()
        # crazy regex program, but pretty much captures multiline invocations
        # and special makefile cases we've encountered
        #matches = re.findall(r'(?:(?<=\.\/\$\(EXE\))|(?<=\.\/\$\(program\)))(?:[ \n])(?:[^\n\\]*)(?:(?:\\\n[^\n\\]*)+|(?:))', data, re.DOTALL)
        # this regex will just match the Makefile line that has `run` in it, and all the chars 
        # of the first command that happens after. It accounts for line continuations.
        matches = list(re.finditer(r'(?<=run)(?:([\s]*\:[^\n]*[\n][\s]*))(?:[^\n\\]*)(?:(?:\\\n[^\n\\]*)+|(?:))', data, re.DOTALL))

        matches = [i.group() for i in matches]

        #print('matches in ', makefile)
        #print(matches)

        if len(matches) > 0:
            # every makefile should have at most one match
            assert len(matches) == 1

            match = matches[0]
            print(makefile, match)

            # now let's find the part of the string that has $(program) | $(EXE) | $(LAUNCHER)
            # we'll grab the last one that appears
            lastExe  = match.rfind('$(EXE)') + len('$(EXE)')
            lastProg = match.rfind('$(program)') + len('$(program)')
            lastLauncher = match.rfind('$(LAUNCHER)') + len('$(LAUNCHER)')

            startPt = max([lastExe, lastProg, lastLauncher])

            assert startPt > 0

            clean = match[startPt:].rstrip().lstrip().replace('\n', ' ').replace('\\', '')

            # if `LAUNCHER` is the startPt, then the next argument is the executable, we drop it
            if lastLauncher == startPt:
                clean = ' '.join(clean.split()[1:])
                print(f'{makefile} has a hardcoded exe in run! raw:[{match}] fixed:[{clean}]\n')

            #print(f'clean [{clean}]')
            return clean
    return ''

def find_makefiles_in_src_dir(srcDir):
    candidates = list(glob.glob(f'{srcDir}/**/[Mm]akefile*', recursive=True)) 
    # each program should have at least 1 Makefile in some capacity (except for miniFE)
    assert len(candidates) != 0
    return candidates


def get_exe_args(targets:list):
    # read the Makefile of each program, find the line with `run` and then the `./$(program)` invocation
    # check that the $LAUNCHER variable isn't populated
    # extract the remaining arguments for execution
    assert len(targets) != 0
    for target in tqdm(targets, desc='Gathering exe args'): 
        srcDir = target['src']
        # let's read the Makefile, strip the 'run' target of 
        #print(target['basename'])
        makefiles = find_makefiles_in_src_dir(srcDir)
        # find the first makefile that has a run command
        # if none of the makefiles have a run command with arguments, the exeArgs will be empty
        for makefile in makefiles:
            exeArgs = get_exec_command_from_makefile(makefile)
            if exeArgs != '':
                break

        if exeArgs == '':
            print('--------------------------------------------------------')
            print(f'Program {target["basename"]} HAS NO INPUT ARGUMENTS!')
            print('--------------------------------------------------------')
            with open(makefile, 'r') as file:
                toprint = '\n'.join(file.readlines()[-10:])
                print(toprint)
            print('--------------------------------------------------------')
            print('--------------------------------------------------------')
        target['exeArgs'] = exeArgs
    return targets



def modify_exe_args_for_some_targets(targets:list):
    for target in targets:
        basename = target['basename']

        if basename == 'dxtc1-cuda':
            target['exeArgs'] = target['exeArgs'].replace('dxtc1-sycl', 'dxtc2-sycl')
        elif basename == 'softmax-cuda':
            target['exeArgs'] = target['exeArgs'].replace('784 ', '784 1 ')
        # just manually set the kmeans -- it doesn't use the $(program) substring in it's makefile
        # so we don't rip any arguments out for it
        elif basename == 'kmeans-cuda':
            target['exeArgs'] = '-r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup'
        elif basename == 'frna-cuda':
            target['exeArgs'] = '../prna-cuda/HIV1-NL43.seq hiv1-nl43.out'
        elif 'inversek2j' in basename:
            target['exeArgs'] = target['exeArgs'].replace('inverse2kj', 'inversek2j')
        elif 'face-' in basename:
            target['exeArgs'] = '../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-gpu.pgm'
        elif 'srad-' in basename:
            target['exeArgs'] = '1000 0.5 502 458'
        elif 'snicit-' in basename:
            target['exeArgs'] = '-k C'
        elif 'grep-' in basename:
            target['exeArgs'] = '-f ../grep-cuda/testcases/lua.lines.js.txt "\."'
        elif (basename == 'che-cuda') or (basename == 'che-omp'):
            target['exeArgs'] = '1000'
        elif (basename == 'mcmd-cuda') or (basename == 'mcmd-omp'):
            target['exeArgs'] = '../mcmd-cuda/dataset/mcmd.inp'


    return targets



def modify_kernel_names_for_some_targets(targets:list):
    for target in targets:
        basename = target['basename']

        if basename == 'assert-cuda':
            target['kernelNames'].remove('testKernel')


    return targets


def search_and_extract_file(inFile):
    # if the file doesn't exist, let's unpack any tar files
    # in the specified directory
    if not (os.path.isfile(inFile) or os.path.islink(inFile)):
        dirToSearch = os.path.normpath(os.path.dirname(inFile))

        print('searching this dir: ', dirToSearch)
        # go up until we hit the parent dir that ends in -omp -cuda -sycl or -hip
        while (True):
            lastName = os.path.basename(dirToSearch)
            if ('-sycl' in lastName) or ('-cuda' in lastName) or ('-omp' in lastName) or ('-hip' in lastName):
                break
            dirToSearch = os.path.normpath(dirToSearch + '/..')
            print('searching this dir: ', dirToSearch, 'for', inFile)
            print(f'{lastName}, {dirToSearch}')

        print('searching dir', dirToSearch)
        print('trying to find', inFile)

        # do a recursive search
        tarFiles = list(glob.glob(f'{dirToSearch}/**/*.tar.gz', recursive=True))
        tgzFiles = list(glob.glob(f'{dirToSearch}/**/*.tgz', recursive=True))
        zipFiles = list(glob.glob(f'{dirToSearch}/**/*.zip', recursive=True))

        print('tarfiles', tarFiles)
        print('zipfiles', zipFiles)
        print('tgzfiles', tgzFiles)

        for tarFile in tarFiles:
            filename = os.path.basename(tarFile)
            command = f'tar -xf {filename}'
            result = subprocess.run(shlex.split(command), cwd=dirToSearch)
            assert result.returncode == 0
            print('Extracted tar archive:', filename)

        for tgzFile in tgzFiles:
            filename = os.path.basename(tgzFile)
            command = f'tar -xzf {filename}'
            result = subprocess.run(shlex.split(command), cwd=dirToSearch)
            assert result.returncode == 0
            print('Extracted tgz archive:', filename)

        for zipFile in zipFiles:
            filename = os.path.basename(zipFile)
            command = f'unzip {filename}'
            result = subprocess.run(shlex.split(command), cwd=dirToSearch)
            assert result.returncode == 0
            print('Extracted zip archive:', filename)

        # now let's check that the file exists
        assert os.path.exists(inFile)

    return


def check_and_unzip_input_files(targets:list):
    for target in tqdm(targets, desc='Checking input files exist'):
        args = target['exeArgs']
        srcDir = target['src']

        #print(target)

        # if there are any input files, let's try to find them
        inputFiles = re.findall(r'\.+\/[0-9a-zA-Z_\+\-\/\.]*', args)
        if len(inputFiles) > 0:
            for inFile in inputFiles:
                # if the word `output` is in the filename, we skip checking for it
                if not ('output' in inFile.lower()):
                    print(f'looking for input file: {srcDir}/{inFile}')
                    search_and_extract_file(f'{srcDir}/{inFile}')

    return 


'''
There doesn't seem to be some easy way of extracting the kernel names without executing the program.
Here we use `cuobjdump` to get the SASS section names, `cu++filt` to demangle, then regex parsing 
to extract the kernel names. It is possible to get duplicates if the kernels are the same name, but
with different arguments/signatures from being templated. We drop duplicates. 
We don't have a smart way of differentiating these kernels during execution either, as `ncu` is not
able to differentiate the executions at runtime either. Leaving it for future work.

Some kernels make ALL external kernel calls. Because most of the time these are to cuSPARSE
(or some other closed-source NVIDIA library), we end up skipping these codes for execution. 

Another note to make is that this returns all the kernel names from the binary. Sometimes the program
will have conditional logic which prevents the program from ever executing a kernel, thus we wont
have any data on said kernel. (example: deredundancy-cuda). We can't know exactly how the logic will
execute ahead-of-time, thus we'll trigger a profiling of the un-executed kernel. This is okay, we
just need to catch ourselves when no `ncu-rep` file is generated.
'''
def get_kernel_names_from_target(target:dict):

    basename = target['basename']
    srcDir = target['src']

    #print('getting kernel names for', basename)
    if '-cuda' in basename:
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
    # it's an OMP program
    else:
        # when we build for OpenMP, there's a section with all the kernel names
        objdumpCommand = f'objdump -t --section=omp_offloading_entries {BUILD_DIR}/{basename}'
        knamesResult = subprocess.run(objdumpCommand, cwd=srcDir, shell=True, 
                                      timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        assert knamesResult.returncode == 0

        toRegex = knamesResult.stdout.decode('UTF-8')

        matches = re.findall(r'(?<=\.omp_offloading\.entry\.)(__omp_offloading.*)(?=\n)', toRegex)

        # all the OMP codes should have at least one offload region
        assert len(matches) != 0
        cleanNames = matches

    #assert len(matches) != 0
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

'''
Something to consider when executing a target is the fact that Nsight Compute (ncu)
instrumentation for roofline can take some time.
We're able to calculate the roofline for EVERY kernel invocation, thus we can very easily
have hundreds of data points for one single code, but it's at the cost of xtime which can
easily go up 10x (or more) due to sampling each kernel invocation.
We can use the `-c #` flag to limit the number of captures we perform.

What we do is get a list of all the kernels in the program, then invoke the program to capture
2 runs of each kernel. The first run is usually disposable due to warm-up, but in some cases it's
the only run, so we use that. 

Something to note is that some codes (like bitpermute-cuda) will incrementally increase the
problem size of what it feeds the kernel invocations. Our approach fails to be able to
capture the later invocations. This is something we need to later fix to increase the
amount of data we capture. The problem is that we must strike a balance between experimentation
time/data gathering and variety of data. 
Performing an `nsys` call to find all the variety in execution, then calling `ncu` with the
skip launch `-s` flag to gather the different kernel exeuctions is a future step we will consider.
At the least it will double the collection time (if there is one kernel invocation with a singular
grid/block size used). But if we have `n` kernels with at least 3 different invocations each, 
we now have to wait (xtime)+(xtime*n*3) seconds for the desired data. 
We'll look into this later as another data gathering approach. It would be more complete, but
it also will take some more effort/time to gather.
I'd rather we start with a simple (slightly smaller) dataset and see what it can achieve for us
before we decide to do a long data gathering process.
'''
def execute_target(target:dict, kernelName:str):
    # we will run each program from within it's source directory, this makes sure
    # any input/output files stay in those directories

    assert kernelName != None
    assert kernelName != ''

    basename = target['basename']
    exeArgs = target['exeArgs']
    srcDir = target['src']

    reportFileName = f'{basename}-[{kernelName}]-report'
    #ncuCommand = f'ncu -f -o {reportFileName} --section SpeedOfLight_RooflineChart -c 2 -k "regex:{kernelName}"'
    ncuCommand = f'ncu -f -o {reportFileName} --set roofline --metrics smsp__sass_thread_inst_executed_op_integer_pred_on -c 2 -k "regex:{kernelName}"'
    exeCommand = f'{ncuCommand} {BUILD_DIR}/{basename} {exeArgs}'.rstrip()

    print('executing command:', exeCommand)

    # we print the stderr to the stdout for analysis
    # 15 minute timeout for now?
    # cm-cuda goes over 10 mins to run!
    try:
        execResult = subprocess.run(shlex.split(exeCommand), cwd=srcDir, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # temporarily doing this to skip slow runs
    except subprocess.TimeoutExpired:
        return (None, None)

    # temporarily doing this to skip breaking runs
    if execResult.returncode != 0:
        return (None, None)

    assert execResult.returncode == 0, f'Execution error: {execResult.stdout.decode("UTF-8")}'

    # check that the ncu-rep was generated. We'll still get a 0 returncode when it doesn't generate an ncu-rep file
    if os.path.isfile(f'{srcDir}/{reportFileName}.ncu-rep'):
        csvCommand = f'ncu --import {reportFileName}.ncu-rep --csv --print-units base --page raw'
        print('executing command:', csvCommand)
        rooflineResults = subprocess.run(shlex.split(csvCommand), cwd=srcDir, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        assert rooflineResults.returncode == 0, f'Roofline parsing error: {rooflineResults.stdout.decode("UTF-8")}'

        return (execResult, rooflineResults)

    else:
        print(f'No report file generated! -- Kernel [{kernelName}] must not have been invoked during execution!')
        # if the report file didn't get generated, return null, indicating the kernel doesn't exist
        return (None, None)


def roofline_results_to_df(rooflineResults):
    ncuOutput = rooflineResults.stdout.decode('UTF-8')
    #print(ncuOutput)

    stringified = StringIO(ncuOutput)

    df = pd.read_csv(stringified, quotechar='"')

    return df


def str_to_float(x):
    return np.float64(x.replace(',', ''))

'''
The CSV file is the output of the ncu report, containing the raw data
that was sampled for each kernel invocation.
The first row can be skipped because it contains the units of each of the
columns. This will be useful later for checking that we got our units correct.

Formulas for Double-Precision Roofline values:
    Achieved Work: (smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

    Achieved Traffic: dram__bytes.sum.per_second

    Arithmetic Intensity: Achieved Work / Achieved Traffic


Formulas for Single-Precision Roofline values:
    Achieved Work: (smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

    Achieved Traffic: dram__bytes.sum.per_second

    Arithmetic Intensity: Achieved Work / Achieved Traffic

It should be noted that these measurements are all on the level of DRAM. We plan to extend this to L1 + L2 later.

There is only one integer instruction counter: smsp__sass_thread_inst_executed_op_integer_pred_on.sum
In order to use it we need to find a way to request it with NCU.
The `check_connect` kernel of `depixel-cuda` has both INTOPS and FLOPS, but NCU only reports the FLOPS.


This was in the NCU documentation:
ncu --query-metrics-mode suffix --metrics sm__inst_executed --chip ga100

We can get the INTOP metrics like so:
ncu -f -o deleteme --set roofline -c 2 -k "regex:check_connect" --metrics smsp__sass_thread_inst_executed_op_integer_pred_on ../../build/depixel-cuda 2048 2048 10

The problem is that we need to then create the INTOP roofline, which NCU doesn't do, but we can technically empirically get. 

'''
def calc_roofline_data(df):

    # kernel data dataframe
    kdf = df.iloc[1:].copy(deep=True)

    assert kdf.shape[0] != 0

    avgCyclesPerSecond  = kdf['smsp__cycles_elapsed.avg.per_second'].apply(str_to_float)


    #print(avgCyclesPerSecond)

    sumDPAddOpsPerCycle = kdf['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumDPMulOpsPerCycle = kdf['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumDPfmaOpsPerCycle = kdf['derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2'].apply(str_to_float)
    # this is in units of (ops/cycle + ops/cycle + ops/cycle) * (cycle/sec) = (ops/sec)
    kdf['dpPerf'] = (sumDPAddOpsPerCycle + sumDPMulOpsPerCycle + sumDPfmaOpsPerCycle) * avgCyclesPerSecond

    sumSPAddOpsPerCycle = kdf['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumSPMulOpsPerCycle = kdf['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'].apply(str_to_float)
    sumSPfmaOpsPerCycle = kdf['derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2'].apply(str_to_float)
    # this is in units of (ops/cycle + ops/cycle + ops/cycle) * (cycle/sec) = (ops/sec)
    kdf['spPerf'] = (sumSPAddOpsPerCycle + sumSPMulOpsPerCycle + sumSPfmaOpsPerCycle) * avgCyclesPerSecond

    # units of (bytes/sec)
    kdf['traffic'] = kdf['dram__bytes.sum.per_second'].apply(str_to_float)

    kdf['dpAI'] = kdf['dpPerf'] / kdf['traffic']
    kdf['spAI'] = kdf['spPerf'] / kdf['traffic']

    kdf['xtime'] = kdf['gpu__time_duration.sum'].apply(str_to_float)
    kdf['device'] = kdf['device__attribute_display_name']

    kdf['intops'] = kdf['smsp__sass_thread_inst_executed_op_integer_pred_on.sum'].apply(str_to_float)
    # need to scale down the xtime to be in seconds
    kdf['intPerf'] = kdf['intops'] / (1e-9 * kdf['xtime'])
    kdf['intAI'] = kdf['intPerf'] / kdf['traffic']

    timeUnits = df.iloc[0]['gpu__time_duration.sum']
    assert timeUnits == 'ns'

    return kdf


# should include exeArgs as input so we can do multiple
# runs of the same code with different cmdline args
def has_already_been_sampled(basename:str, kernelName:str, df:pd.DataFrame): 

    if df.shape[0] == 0:
        return False

    subset = df[(df['targetName'] == basename) & (df['kernelName'] == kernelName)]

    return subset.shape[0] > 0


def execute_targets(targets:list, dfFilename:str):
    # this will gather the data for the targets into a dataframe for saving
    # if a code can not be executed, we will skip it

    assert len(targets) != 0

    if os.path.isfile(dfFilename):
        df = pd.read_csv(dfFilename)
    else:
        df = pd.DataFrame()

    for target in tqdm(targets, desc='Executing programs!'):
        basename = target['basename']
        kernelNames = target['kernelNames']
        exeArgs = target['exeArgs']
        
        # if the program doesn't define any kernels locally
        if len(kernelNames) == 0:
            print(f'Skipping {basename} due to having no internal defined CUDA kernels!')
            continue

        # we perform one invocation for each kernel
        for kernelName in kernelNames:

            if has_already_been_sampled(basename, kernelName, df):
                print(f'Skipping {basename}:[{kernelName}] due to having already been sampled!')
                continue

            execResult, rooflineResult = execute_target(target, kernelName)
            
            if execResult != None:
                stdout = execResult.stdout.decode('UTF-8')
                assert execResult.returncode == 0, f'error in execution!\n {stdout}'

                rawDF = roofline_results_to_df(rooflineResult)
                roofDF = calc_roofline_data(rawDF)

                subset = roofDF[['Kernel Name', 'traffic', 'dpAI', 'spAI', 'dpPerf', 'spPerf', 'xtime', 'Block Size', 'Grid Size', 'device', 'intops', 'intPerf', 'intAI']].copy()
                subset['targetName'] = basename
                subset['exeArgs'] = exeArgs
                subset['kernelName'] = kernelName

            # if the return value is None, the kernel wasn't executed,
            # so we will add it to the database, but with all NaN values to
            # indicate the kernel doesn't get executed and so we skip it
            # if we try to re-run data gathering
            else:
                # doing this now to skip failing runs
                continue
                dataDict = {'targetName':[basename], 'exeArgs':[exeArgs], 'kernelName':[kernelName]}
                subset = pd.DataFrame(dataDict)

            df = pd.concat([df, subset], ignore_index=True)
            # this does do a lot of redundant writing, but we only expect at most 10k samples, so 
            # it's not too much of a slowdown to write out each time.
            # if it proves too slow, we could switch to parquet format, although this is good for now.
            df.to_csv(dfFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"', index=False, na_rep='NULL')


    # save the dataframe
    #dfFilename = './roofline-data.csv'
    #print(f'Saving dataframe! {dfFilename}')
    print('Sample gathering complete!')

    return df

'''
Here is the formula Nsight Compute (ncu) uses for calculating arithmetic intensity (single-precision):
These formulas are for each cache level. For now, we focus only on DRAM.

DRAM -- DRAM -- DRAM -- DRAM
    Achieved Work: (smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

    Achieved Traffic: dram__bytes.sum.per_second

    Arithmetic Intensity: Achieved Work / Achieved Traffic

    AI is a measure of FLOP/byte

    xtime: gpu__time_duration.sum
    Performance: Achieved Work / xtime

L1 -- L1 -- L1 -- L1 
    Achieved Work: (smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

    Achieved Traffic: derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second

    Arithmetic Intensity: Achieved Work / Achieved Traffic

L2 -- L2 -- L2 -- L2
    Achieved Work: (smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

    Achieved Traffic: derived__lts__lts2xbar_bytes.sum.per_second

    Arithmetic Intensity: Achieved Work / Achieved Traffic


Example execution command:  ncu -f -o test-report --set roofline -c 2 ../../build/haccmk-cuda 1000
This gathers all the data in one run, it might have to do counter multiplexing, not sure. The `-c 2` will indiscriminately
sample just he first two CUDA kernels that are encountered.

Example execution command:  ncu -f -o test-report --set roofline --replay-mode application -c 2 ../../build/haccmk-cuda 1000
This is going to do many repeated runs of the program to capture each of the necessary metrics for roofline.
This will allow us to get the rooflines at the L1, L2, and DRAM levels.
It will do 16 repeated runs of the program though, which can take some time XD.

Because this is a slow process, and some programs might have many kernels, we first perform an nsys run to extract
the major time-consuming kernels. These will be the ones we pass with the `-k` argument to `ncu` so that it only captures
data on that subset of kernels.

Capture the first 5 kernel invocations that match the given regex
ncu -f -o test-report --set roofline -c 5 -k "regex:fill_sig|hgc" ../../build/lulesh-cuda  -i 5 -s 32 -r 11 -b 1 -c 1

Example nsys command:
    nsys profile -f true -o test-report --cpuctxsw=none --backtrace=none ../../build/haccmk-cuda 1000
    nsys stats --report cuda_gpu_kern_sum test-report.nsys-rep
    

Now, this report will tell us the top time-consuming kernels. We'll need to make an `ncu` invocation for each kernel of interest. 

To get the `ncu` data from the command line we can run this command:
    ncu --import test-report.ncu-rep --csv --page raw
It will dump all the collected data and we'll need to read it in and use our formulas to calculate the roofline data.





We can also omit the L1, L2, and DRAM rooflines and just get a high-level roofline.
We can limit the gathered data to one section instead of all the roofline charts getting generated. 

ncu -f -o test-report --section SpeedOfLight_RooflineChart -c 5 -k "regex:fill_sig|hgc" ../../build/lulesh-cuda  -i 5 -s 32 -r 11 -b 1 -c 1
ncu --import test-report.ncu-rep --csv --page raw

Formulas for roofline:
    Achieved Work: (smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

    Achieved Traffic: dram__bytes.sum.per_second (this is in units of bytes(mbytes,gbytes)-per-second)

    Arithmetic Intensity: Achieved Work / Achieved Traffic

Can we calculate the rooflines from the provided data?

'''


def save_run_results(targets:list=None, csvFilename:str='roofline-data.csv'):
    return



def main():


    parser = argparse.ArgumentParser()

    parser.add_argument('--buildDir', type=str, required=False, default='./build', help='Directory containing all the built executables')
    parser.add_argument('--srcDir', type=str, required=False, default='./src', help='Directory containing all the source files for the target executables')
    parser.add_argument('--outfile', type=str, required=False, default='./roofline-data.csv', help='Output CSV file with gathered data')
    parser.add_argument('--targets', type=list, required=False, default=None, help='Optional subset of targets to run')
    parser.add_argument('--forceRerun', action=argparse.BooleanOptionalAction, help='Whether to forcibly re-run already-gathered programs')
    parser.add_argument('--skipRodiniaDownload', action=argparse.BooleanOptionalAction, help='Skip downloading rodinia dataset')


    args = parser.parse_args()

    setup_dirs(args.buildDir, args.srcDir)

    # let's check if rodinia has been downloaded, if not, download it
    if not has_rodinia_datasets():
        if not args.skipRodiniaDownload:
            download_rodinia_and_extract()
        else:
            print('[WARN] Rodinia not detected! User requested to skip rodinia dataset download! Some codes may fail on invocation!')


    print('Starting data gathering process!')

    targets = get_runnable_targets()

    download_files_for_some_targets(targets)

    targets = get_exe_args(targets)
    targets = modify_exe_args_for_some_targets(targets)

    check_and_unzip_input_files(targets)

    targets = get_kernel_names(targets)

    targets = modify_kernel_names_for_some_targets(targets)

    omp_targets = []

    for target in targets:
        #if target['basename'] == 'attention-omp':
        # skip prna for now as it takes a long time to run
        # match-omp stays in GPU memory after being force killed
        if '-omp' in target['basename'] and (not target['basename'] in ['prna-omp', 'memtest-omp', 'metropolis-omp', 'hypterm-omp', 'gc-omp', 'scan-omp', 'frna-omp', 'match-omp', 'lid-driven-cavity-omp']):
            omp_targets.append(target)
            #pprint(target)
            #execute_targets([target], args.outfile)

    #targets = targets[:70]

    #pprint(targets)

    results = execute_targets(omp_targets, args.outfile)
    #results = execute_targets(targets, args.outfile)

    return



if __name__ == "__main__":
    main()