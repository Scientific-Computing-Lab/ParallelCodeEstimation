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


def get_exec_command_from_makefile(makefile):
    assert os.path.isfile(makefile)

    with open(makefile, 'r') as file:
        for line in file:
            if './$(program) ' in line:
                matches = re.findall(r'(?<=\.\/\$\(program\)).*', line)
                assert len(matches) == 1
                return matches[0]

    return


def get_exe_args(targets:list):
    # read the Makefile of each program, find the line with `run` and then the `./$(program)` invocation
    # check that the $LAUNCHER variable isn't populated
    # extract the remaining arguments for execution
    assert len(targets) != 0
    for target in tqdm(targets, desc='Getting run commands'): 
        srcDir = target['src']
        # let's read the Makefile, strip the 'run' target of 
        exeArgs = get_exec_command_from_makefile(f'{srcDir}/Makefile')
        target['exeArgs'] = exeArgs
    return targets



'''
Something to consider when executing a target is the fact that Nsight Compute (ncu)
instrumentation for roofline can take some time.
We're able to calculate the roofline for EVERY kernel invocation, thus we can very easily
have hundreds of data points for one single code. We can use the `-c #` flag to limit the
number of captures we perform.
'''
def execute_target(target:dict):
    # we will run each program from within it's source directory, this makes sure
    # any input/output files stay in those directories

    basename = target['basename']
    exeArgs = target['exeArgs']
    srcDir = target['src']
    exeCommand = f'../../build/{basename} {exeArgs}'.rstrip()

    print('executing command:', exeCommand)

    # we print the stderr to the stdout for analysis
    # 60 second timeout for now?
    result = subprocess.run(shlex.split(exeCommand), cwd=srcDir, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    return result


def execute_targets(targets:list):
    # this will gather the data for the targets into a dataframe for saving
    # if a code can not be executed, we will skip it

    assert len(targets) != 0
    df = pd.DataFrame()

    for target in tqdm(targets, desc='Executing programs!'):
        result = execute_target(target)

        if result.returncode != 0:
            print(result.stdout)

        stdout = result.stdout.decode()
        pprint(stdout)


        # the metrics that we maybe want to gather from ncu 
        '''
        gpu__time_duration.sum [ms]
        device__attribute_display_name
        device__attribute_l2_cache_size
        '''

        '''
        Here is the formula Nsight Compute (ncu) uses for calculating arithmetic intensity (single-precision):
            Achieved Work: (smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed + derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2) * smsp__cycles_elapsed.avg.per_second

            Achieved Traffic: dram__bytes.sum.per_second

            Arithmetic Intensity: Achieved Work / Achieved Traffic

            AI is a measure of FLOP/byte

            xtime: gpu__time_duration.sum
            Performance: Achieved Work / xtime


        Example execution command:  ncu -f -o test-report --set roofline -c 2 ../../build/haccmk-cuda 1000
        '''

    return


def save_run_results(targets:list=None, csvFilename:str='roofline-data.csv'):
    return



def main():

    print('Starting data gathering process!')

    parser = argparse.ArgumentParser()

    parser.add_argument('--buildDir', type=str, required=False, default='./build', help='Directory containing all the built executables')
    parser.add_argument('--srcDir', type=str, required=False, default='./src', help='Directory containing all the source files for the target executables')
    parser.add_argument('--outfile', type=str, required=False, default='roofline-data.csv', help='Output CSV file with gathered data')
    parser.add_argument('--targets', type=list, required=False, default=None, help='Optional subset of targets to run')
    parser.add_argument('--forceRerun', action=argparse.BooleanOptionalAction, help='Whether to forcibly re-run already-executed functions')


    args = parser.parse_args()

    targets = get_runnable_targets(buildDir=args.buildDir, srcDir=args.srcDir)
    targets = get_exe_args(targets)

    targets = targets[:2]
    pprint(targets)
    results = execute_targets(targets)

    return



if __name__ == "__main__":
    main()