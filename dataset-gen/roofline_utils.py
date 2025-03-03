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

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage, SystemMessage, AssistantMessage
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent


import subprocess
import shlex
from io import StringIO
import re

import tiktoken
#import plotly.express as px

import csv
from sklearn.model_selection import train_test_split 


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