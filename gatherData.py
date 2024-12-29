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


def execute_target(exeCommand:str, srcDir:str):
    # we will run each program from within it's source directory, this makes sure
    # any input/output files stay in those directories

    # we print the stderr to the stdout for analysis
    result = subprocess.run(exeCommand.split(' '), cwd=srcDir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    return

def execute_targets(targets:list):
    # this will gather the data for the targets into a dataframe for saving
    # if a code can not be executed, we will skip it

    assert len(targets) != 0
    df = pd.DataFrame()




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
    pprint(targets)

    return



if __name__ == "__main__":
    main()