import pandas as pd
import os
import numpy as np
import re
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt

import json
from pprint import pprint
from tqdm import tqdm

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_core.models import UserMessage, SystemMessage, AssistantMessage
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent
import autogen_core

import subprocess
import shlex
from io import StringIO
import re

import tiktoken
#import plotly.express as px

import csv
from sklearn.model_selection import train_test_split 

import asyncio

print(f"Autogen version: {autogen_core.__version__}")

# GPU specs

# you can get this from deviceQuery
gpuName = 'NVIDIA RTX 3080'

# you can call nvidia-smi -i 0 -q to see what the clock is set to 
# you can also set the clock with nvidia-smi -lgc 1440,1440 for consistent measurements
# vendor specs show the base clock
baseClockHz = 1.440e9

# find these values here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
SPinstPerCyclePerSM = 128
DPinstPerCyclePerSM = 2
intInstPerCyclePerSM = 64

# find this in deviceQuery or GPU vendor specs
numSMs = 68

# we always assume you're doing FMA 
numFMAopPerInst = 2

# conversion multiplier
tflopPerflop = 1e-12

# get this from your GPU vendor specs, mine was 760.3 GB/s
maxBandwidthTBPerSec = 0.7603

spOPMaxPerfTFLOP = SPinstPerCyclePerSM * numSMs * baseClockHz * numFMAopPerInst * tflopPerflop
dpOPMaxPerfTFLOP = DPinstPerCyclePerSM * numSMs * baseClockHz * numFMAopPerInst * tflopPerflop
intOPMaxPerfTFLOP = intInstPerCyclePerSM * numSMs * baseClockHz * numFMAopPerInst * tflopPerflop

spOPMaxPerfTFLOP_noFMA = spOPMaxPerfTFLOP / 2
dpOPMaxPerfTFLOP_noFMA = dpOPMaxPerfTFLOP / 2

print('Max SP TFLOP/s with FMA', round(spOPMaxPerfTFLOP, 3))
print('Max DP TFLOP/s with FMA', round(dpOPMaxPerfTFLOP, 3))
print('Max SP TFLOP/s w/out FMA', round(spOPMaxPerfTFLOP_noFMA, 3))
print('Max DP TFLOP/s w/out FMA', round(dpOPMaxPerfTFLOP_noFMA, 3))
print('Max TINTOP/s', round(intOPMaxPerfTFLOP, 3))

balancePointSPFLOPPerByte = spOPMaxPerfTFLOP / maxBandwidthTBPerSec
balancePointDPFLOPPerByte = dpOPMaxPerfTFLOP / maxBandwidthTBPerSec
balancePointINTOPPerByte = intOPMaxPerfTFLOP / maxBandwidthTBPerSec
print(f'SP Balance Point is at: {round(balancePointSPFLOPPerByte, 2)} flop/byte')
print(f'DP Balance Point is at: {round(balancePointDPFLOPPerByte, 2)} flop/byte')
print(f'INT Balance Point is at: {round(balancePointINTOPPerByte, 2)} intop/byte')

peakPerfGspFLOPs = spOPMaxPerfTFLOP * 1e3
peakPerfGdpFLOPs = dpOPMaxPerfTFLOP * 1e3
peakPerfGINTOPs = intOPMaxPerfTFLOP * 1e3
memBandwidthGBs = maxBandwidthTBPerSec * 1e3

print()
print('These values get passed as LLM context so the model can infer about rooflines:')
print(f'Peak SP GFLOP/s {round(peakPerfGspFLOPs, 3)} with FMA')
print(f'Peak DP GFLOP/s {round(peakPerfGdpFLOPs, 3)} with FMA')
print(f'Peak GINTOP/s {round(peakPerfGINTOPs, 3)} with FMA')




with open('simple-scraped-kernels-CUDA-pruned-with-sass.json', 'r') as file:
    scrapedCUDA = json.load(file)

with open('simple-scraped-kernels-OMP-pruned-with-sass.json', 'r') as file:
    scrapedOMP = json.load(file)


scrapedCodes = scrapedCUDA + scrapedOMP

print('scraped and pruned CUDA programs count', len(scrapedCUDA))
print('scraped and pruned OMP  programs count', len(scrapedOMP))


# the kernelName should be the 'kernelName' from the roofline data dataframe
# FUTURE NOTE: we need to change the simpleScrapeKernels script to emit a JSON dictionary
# instead of a JSON list -- so we don't have to search it each time.
def get_sass_and_source_from_scraped_codes(targetName, kernelName):

  kernelSASS = ''
  kernelSource = ''

  for elem in scrapedCodes:
    basename = elem['basename']
    if basename == targetName:
      # go through the SASS keys to get the kernel
      sassKeys = list(elem['sass'].keys())
      for k in sassKeys:
        if kernelName in k:
          kernelSASS = elem['sass'][k]
          break

      # go through the kernels keys to get the proper source
      sourceKeys = list(elem['kernels'].keys())
      #print(sourceKeys)
      for k in sourceKeys:
        if kernelName in k:
          kernelSource = elem['kernels'][k]
          break

      break

  return (kernelSASS, kernelSource)




dtypes={'Kernel Name':'string', 
        'traffic':np.float64,
        'dpAI':np.float64,
        'spAI':np.float64,
        'dpPerf':np.float64,
        'spPerf':np.float64,
        'xtime':np.float64,
        'Block Size': 'string',
        'Grid Size': 'string',
        'device': 'string',
        "intops": np.float64, 
        "intPerf" : np.float64,
        "intAI": np.float64,
        'targetName': 'string',
        'exeArgs': 'string',
        'kernelName': 'string',
        }


def chat_history_to_json_line(ctxMessages:list):
    jsonDict = {'messages':[]}
    for msg in ctxMessages:
        if type(msg) == SystemMessage:
            role = 'system'
        elif type(msg) == UserMessage:
            role = 'user'
        elif type(msg) == AssistantMessage:
            role = 'assistant'
        else:
            assert False, f'Unknown message type: {type(msg)} of {msg}'
        content = msg.content

        jsonDict['messages'].append({'role':role, 'content':content})

    return json.dumps(jsonDict, allow_nan=False)




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

# this system message combines the device information and the non-detailed few-shot
# examples. We need some few-shot breif examples of the output so that the model 
# knows how to respond.
#systemMessage = '''You are a GPU performance analysis expert that classifies kernels into Arithmetic Intensity Roofline model categories based on their source code characteristics. Your task is to provide one of the following performance boundedness classifications: Compute or Bandwidth.  A kernel is considered Compute bound if its performance is primarily limited by the number of operations it performs, and Bandwidth bound if its performance is primarily limited by the rate at which data can be moved between memory and processing units.
#
#Provide only one word as your response, chosen from the set: ['Compute', 'Bandwidth'].
#**Examples:**
#**Example 1:**
#```
#Kernel Source Code (simplified):
#for i = 0 to 1000000 {
#  a[i] = a[i] + b[i];
#}
#```
#Response: Compute
#
#**Example 2:**
#```
#Kernel Source Code (simplified):
#for i = 0 to 10 {
#  load_data(large_array);   //loads from large memory
#  process_data(large_array); //processes data
#  store_data(large_array);  //stores back to memory
#}
#```
#Response: Bandwidth
#
#Now, analyze the following source codes for the requested CUDA or OpenMP (OMP) target offload kernel of the specified hardware.'''
#

pseudo_code_examples='''**Example 1:**
```
Kernel Source Code (simplified):
for i = 0 to 1000000 {
  a[i] = a[i] + b[i];
}
```
Response: Compute

**Example 2:**
```
Kernel Source Code (simplified):
for i = 0 to 10 {
  load_data(large_array);   //loads from large memory
  process_data(large_array); //processes data
  store_data(large_array);  //stores back to memory
}
```
Response: Bandwidth
'''

with open('../few-shot-examples/cuda_BB_kernel.cu', 'r') as file:
    cuda_BB_example = file.read()

with open('../few-shot-examples/cuda_CB_kernel.cu', 'r') as file:
    cuda_CB_example = file.read()

with open('../few-shot-examples/omp_BB_kernel.cpp', 'r') as file:
    omp_BB_example = file.read()

with open('../few-shot-examples/omp_CB_kernel.cpp', 'r') as file:
    omp_CB_example = file.read()


# each system message will have a compute-bound and a bandwidth-bound example
def make_system_message(exampleType=0, useSASS=False):

  if useSASS:
    assert exampleType == 0
    start_systemMessage = '''You are a GPU performance analysis expert that classifies kernels into Arithmetic Intensity Roofline model categories based on their source code characteristics. Your task is to provide one of the following performance boundedness classifications: Compute or Bandwidth. A kernel is considered Compute bound if its integer, single-precision, or double-precision floating point operations far outweight the number of global memory reads and writes it makes. Alternatively, a kernel can be considered Bandwidth bound if it does not perform enough arithmetic instructions for all the memory transactions it performs.  

Provide only one word as your response, chosen from the set: ['Compute', 'Bandwidth'].
**Examples:**'''
  else:
    start_systemMessage = '''You are a GPU performance analysis expert that classifies kernels into Arithmetic Intensity Roofline model categories based on their source code characteristics. Your task is to provide one of the following performance boundedness classifications: Compute or Bandwidth.  A kernel is considered Compute bound if its performance is primarily limited by the number of operations it performs, and Bandwidth bound if its performance is primarily limited by the rate at which data can be moved between memory and processing units.

Provide only one word as your response, chosen from the set: ['Compute', 'Bandwidth'].
**Examples:**'''

  # just pseudocode as examples
  if exampleType == 0: 
    if useSASS:
      end_systemMessage = '''Now, analyze the following SASS source codes for the requested kernel of the specified hardware.'''
    else:
      end_systemMessage = '''Now, analyze the following source codes for the requested kernel of the specified hardware.'''
    examples = pseudo_code_examples
  # both OMP and CUDA examples
  elif exampleType == 1:
    end_systemMessage = '''Now, analyze the following source codes for the requested CUDA or OpenMP (OMP) target offload kernel of the specified hardware.'''
    examples = f'**Example 1:**\n```{cuda_BB_example}```\nResponse: Bandwidth\n\n'
    examples += f'**Example 2:**\n```{cuda_CB_example}```\nResponse: Compute\n\n'
    examples += f'**Example 3:**\n```{omp_BB_example}```\nResponse: Bandwidth\n\n'
    examples += f'**Example 4:**\n```{omp_CB_example}```\nResponse: Compute\n'
  # just OMP examples
  elif exampleType == 2:
    end_systemMessage = '''Now, analyze the following source codes for the requested OpenMP (OMP) target offload kernel of the specified hardware.'''
    examples = f'**Example 1:**\n```{omp_BB_example}```\nResponse: Bandwidth\n\n'
    examples += f'**Example 2:**\n```{omp_CB_example}```\nResponse: Compute\n'
  # just CUDA examples
  elif exampleType == 3:
    end_systemMessage = '''Now, analyze the following source codes for the requested CUDA kernel of the specified hardware.'''
    examples = f'**Example 1:**\n```{cuda_BB_example}```\nResponse: Bandwidth\n\n'
    examples += f'**Example 2:**\n```{cuda_CB_example}```\nResponse: Compute\n'
  else:
    assert exampleType < 4 and exampleType >= 0, 'Requested example type D.N.E.'

  return start_systemMessage+'\n'+examples+'\n'+end_systemMessage


def make_kernel_info_message(device, exeArgs, kernelName, blockSz, gridSz, language, useSASS=False):
    assert kernelName != ''

    if language == 'OMP':
      cleanKName = demangle_omp_kernel_name(kernelName)
      assert cleanKName != ''
      beginPart = f'Classify the {language} kernel in function [{cleanKName}] as Bandwidth or Compute bound.'
    else:
      # if were prompting for a CUDA code
      cleanKName = kernelName
      beginPart = f'Classify the {language} kernel called [{cleanKName}] as Bandwidth or Compute bound.'

    builtPrompt = f'{beginPart} The system it will execute on is a [{device}] with a peak single-precision performance of {round(peakPerfGspFLOPs,2)} GFLOP/s, a peak double-precision performance of {round(peakPerfGdpFLOPs,2)} GFLOP/s, a peak integer performance of {round(peakPerfGINTOPs,2)} GINTOP/s, and a max bandwidth of {round(memBandwidthGBs,2)} GB/s. The block and grid sizes of the invoked kernel are {blockSz} and {gridSz}, respectively. The executable running this kernel is launched with '

    if exeArgs == '':
      builtPrompt += 'no command line arguments.'
    else:
      builtPrompt += f'the following command line arguments: [{exeArgs}].'

    if useSASS:
      builtPrompt += f' Below is the SASS kernel source code of the requested {language} kernel.'
    else:
      builtPrompt += f' Below is the source code containing the {language} kernel definition and other source code for the executable.'

    return builtPrompt


#async def make_chat_history(kernel_info, kernelCode, exampleType=0, expectedAnswer=''):
async def make_chat_history(kernel_info, kernelCode, exampleType=0, expectedAnswer='', useSASS=False):

    systemMessage = make_system_message(exampleType, useSASS)
    sys_msg = SystemMessage(content=systemMessage)
    kernel_info_msg = UserMessage(source='User', content=kernel_info)
    code_msg = UserMessage(source='User', content=f'```{kernelCode}```')

    if expectedAnswer != '':
      assis_msg = AssistantMessage(source='assistant', content=f'{expectedAnswer}')
      context = UnboundedChatCompletionContext(initial_messages=[sys_msg, kernel_info_msg, code_msg, assis_msg])
    else:
      context = UnboundedChatCompletionContext(initial_messages=[sys_msg, kernel_info_msg, code_msg])


    return context



def writeToFile(filename, lines):
    # going to overwrite the whole file each time
    # it's redundant but the file wont be that large
    # so the speed doesn't matter
    with open(filename, 'w') as jsonLFile:
        jsonLFile.write(lines)


async def write_df_to_jsonl(df, filename, exampleType=0, includeAnswer=False, useSASS=False):
  jsonLLines = ''

  filename += f'-{df.shape[0]}-samples'

  # for each sample we got
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
      targetName = row['targetName']
      kernelName = row['Kernel Name']
      exeArgs = row['exeArgs']
      blockSz = row['Block Size']
      gridSz = row['Grid Size']
      language = row['language']
      device = row['device']
      kernelCode = row['kernelCode']
      expectedAnswer = row['answer']

      infoMsg = make_kernel_info_message(device, exeArgs, kernelName, blockSz, gridSz, language, useSASS)

      chatHist = None
      if includeAnswer:
        chatHist = await make_chat_history(infoMsg, kernelCode, exampleType, expectedAnswer, useSASS=useSASS)
      else:
        chatHist = await make_chat_history(infoMsg, kernelCode, exampleType, useSASS=useSASS)

      assert chatHist != None

      messageHist = await chatHist.get_messages()
      resultStr = chat_history_to_json_line(messageHist)

      jsonLLines = jsonLLines + resultStr + '\n'
      writeToFile(f'{filename}.jsonl', jsonLLines)
  
  return


def is_already_sampled(df, row, trial, temp, topp):

    targetName = row['targetName']
    kernelName = row['Kernel Name']

    if df.shape[0] == 0:
        return False

    resultRow = df[(df['temp'] == temp) & (df['topp'] == topp) & (df['trial'] == trial) & (df['Kernel Name'] == kernelName) & (df['targetName'] == targetName)]

    if resultRow.shape[0] == 0:
        return False

    assert resultRow.shape[0] == 1, f"resultRow.shape = {resultRow.shape}"

    response = resultRow.iloc[0]['llmResponse']

    return response != ''



def bytes_to_ascii(bytesList):
    bytes_arr = bytes(bytesList)
    return bytes_arr.decode('ascii')

# logprobs [ChatCompletionTokenLogprob(token='Bandwidth', logprob=-0.00010902655776590109, top_logprobs=[TopLogprob(logprob=-0.00010902655776590109, bytes=[66, 97, 110, 100, 119, 105, 100, 116, 104]), TopLogprob(logprob=-9.25010871887207, bytes=[67, 111, 109, 112, 117, 116, 101])]

def convert_logprobs_to_json_str(logProbsObj):
    if logProbsObj is None:
        return ''

    toReturn = {}
    for ccTL in logProbsObj:
        toReturn[ccTL.token] = {'logprob':ccTL.logprob, 'topLogprob':dict([(bytes_to_ascii(top.bytes), top.logprob) for top in ccTL.top_logprobs])}

    return json.dumps(toReturn)