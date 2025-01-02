'''
This script is to be executed after all the codes have been built
into the ./build dir of this project.

Here we will run each of the codes using the default `make run` arguments.
We were able to build at least 400 codes, so we will rip their `make run` arguments
and then execute them with the same args.

We will use `nsys` to record the roofline data of each run.

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



def has_rodinia_datasets(srcDir):
    return os.path.isdir(f'{srcDir}/data')


def download_rodinia_and_extract(srcDir):
    print('Downloading Rodinia Data...')

    command = f'wget http://www.cs.virginia.edu/~skadron/lava/Rodinia/Packages/rodinia_3.1.tar.bz2 && tar -xf ./rodinia_3.1.tar.bz2 rodinia_3.1/data && mv ./rodinia_3.1/data {srcDir}/data'
    result = subprocess.run(command, shell=True)

    assert result.returncode == 0
    assert has_rodinia_datasets(srcDir)

    print('Rodinia download and unzip complete!')

    return


def run_setup_scripts_for_some_targets(targets):
    for target in targets:
        basename = target['basename']
        srcDir = target['src']

        # required data files are unable to be downloaded due to broken links :(
        # leaving this in here for future
        if basename == 'lzss-cuda':
            if not os.path.isfile(f'{srcDir}/tpch.zip'):
                command = f'./get_sample_data.sh'
                result = subprocess.run(command, cwd=srcDir, shell=True)
                assert result.returncode == 0

        elif basename == 'gc-cuda':
            if not os.path.isfile(f'{srcDir}/../mis-cuda/internet.egr'):
                command = f'wget --no-check-certificate https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/internet.egr && mv ./internet.egr {srcDir}/../mis-cuda/'
                result = subprocess.run(command, shell=True)
                assert result.returncode == 0


    return



def get_runnable_targets(buildDir:str, srcDir:str):
    # gather a list of dictionaries storing executable names and source directories
    # the list of dicts will later have run command information added to them
    files = glob.glob(f'{buildDir}/*')
    execs = []
    for entry in files:
        # check we have a file and it's an executable
        if os.path.isfile(entry) and os.access(entry, os.X_OK):
            basename = os.path.basename(entry)
            execSrcDir = f'{srcDir}/{basename}'

            # check we have the source code too
            assert os.path.isdir(execSrcDir)

            execDict = {'basename':basename, 
                        'exe':entry, 
                        'src':execSrcDir }
            execs.append(execDict)

    return execs


def search_and_extract_file(inFile):
    # if the file doesn't exist, let's unpack any tar files
    # in the specified directory
    if not (os.path.isfile(inFile) or os.path.islink(inFile)):
        dirToSearch = os.path.dirname(inFile)

        print('searching dir', dirToSearch)
        print('trying to find', inFile)

        tarFiles = list(glob.glob(f'{dirToSearch}/*.tar.gz'))
        tgzFiles = list(glob.glob(f'{dirToSearch}/*.tgz'))
        zipFiles = list(glob.glob(f'{dirToSearch}/*.zip'))

        print(tarFiles)
        print(zipFiles)
        print(tgzFiles)

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


def get_exec_command_from_makefile(makefile):
    assert os.path.isfile(makefile)

    with open(makefile, 'r') as file:
        for line in file:
            if './$(program) ' in line:
                matches = re.findall(r'(?<=\.\/\$\(program\)).*', line)
                assert len(matches) == 1

                return matches[0]

    return ''



def get_exe_args(targets:list):
    # read the Makefile of each program, find the line with `run` and then the `./$(program)` invocation
    # check that the $LAUNCHER variable isn't populated
    # extract the remaining arguments for execution
    assert len(targets) != 0
    for target in tqdm(targets, desc='Gathering exe args'): 
        srcDir = target['src']
        # let's read the Makefile, strip the 'run' target of 
        exeArgs = get_exec_command_from_makefile(f'{srcDir}/Makefile')
        target['exeArgs'] = exeArgs
    return targets

def modify_exe_args_for_some_targets(targets:list):
    for target in targets:
        basename = target['basename']

        if basename == 'dxtc1-cuda':
            target['exeArgs'] = target['exeArgs'].replace('dxtc1-sycl', 'dxtc2-sycl')

    return targets

def check_and_get_input_files(targets:list):
    for target in tqdm(targets, desc='Checking input files exist'):
        args = target['exeArgs']
        srcDir = target['src']

        print(target)

        # if there are any input files, let's try to find them
        inputFiles = re.findall(r'\.+\/[0-9a-zA-Z_\-\/\.]*', args)
        if len(inputFiles) > 0:
            for inFile in inputFiles:
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

    cuobjdumpCommand = f'cuobjdump --list-text ../../build/{basename} | cu++filt'
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


    #print(matches)

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
    ncuCommand = f'ncu -f -o {reportFileName} --section SpeedOfLight_RooflineChart -c 2 -k "regex:{kernelName}"'
    exeCommand = f'{ncuCommand} ../../build/{basename} {exeArgs}'.rstrip()

    print('executing command:', exeCommand)

    # we print the stderr to the stdout for analysis
    # 5 minute timeout for now?
    execResult = subprocess.run(shlex.split(exeCommand), cwd=srcDir, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert execResult.returncode == 0

    # check that the ncu-rep was generated. We'll still get a 0 returncode when it doesn't generate an ncu-rep file
    if os.path.isfile(f'{srcDir}/{reportFileName}.ncu-rep'):
        csvCommand = f'ncu --import {reportFileName}.ncu-rep --csv --print-units base --page raw'
        print('executing command:', csvCommand)
        rooflineResults = subprocess.run(shlex.split(csvCommand), cwd=srcDir, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert rooflineResults.returncode == 0

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

    timeUnits = df.iloc[0]['gpu__time_duration.sum']
    assert timeUnits == 'ns'

    return kdf


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

                subset = roofDF[['Kernel Name', 'traffic', 'dpAI', 'spAI', 'dpPerf', 'spPerf', 'xtime', 'Block Size', 'Grid Size', 'device']].copy()
                subset['targetName'] = basename
                subset['exeArgs'] = exeArgs
                subset['kernelName'] = kernelName

            # if the return value is None, the kernel wasn't executed,
            # so we will add it to the database, but with all NaN values to
            # indicate the kernel doesn't get executed and so we skip it
            # if we try to re-run data gathering
            else:
                dataDict = {'targetName':[basename], 'exeArgs':[exeArgs], 'kernelName':[kernelName]}
                subset = pd.DataFrame(dataDict)

            df = pd.concat([df, subset], ignore_index=True)


    # save the dataframe
    #dfFilename = './roofline-data.csv'
    print(f'Saving dataframe! {dfFilename}')
    df.to_csv(dfFilename, quoting=csv.QUOTE_NONNUMERIC, quotechar='"', index=False, na_rep='NULL')

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

    # let's check if rodinia has been downloaded, if not, download it
    if not has_rodinia_datasets(args.srcDir):
        if not args.skipRodiniaDownload:
            download_rodinia_and_extract(args.srcDir)


    print('Starting data gathering process!')

    targets = get_runnable_targets(buildDir=args.buildDir, srcDir=args.srcDir)

    run_setup_scripts_for_some_targets(targets)

    targets = get_exe_args(targets)
    targets = modify_exe_args_for_some_targets(targets)

    check_and_get_input_files(targets)

    targets = get_kernel_names(targets)


    #for target in targets:
    #    if target['basename'] == 'lulesh-cuda':
    #        pprint(target)
    #        execute_targets([target])

    targets = targets[:40]
    #pprint(targets)

    results = execute_targets(targets, args.outfile)

    return



if __name__ == "__main__":
    main()