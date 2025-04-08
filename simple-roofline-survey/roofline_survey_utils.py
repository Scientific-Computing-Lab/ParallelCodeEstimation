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

import asyncio
import random

print(f"Autogen version: {autogen_core.__version__}")


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


questionTemplate ='''Question: Given a GPU having a global memory with a max bandwidth of {} GB/s and a peak performance of {} GFLOP/s, if a program executed with an Arithmetic Intensity of {} FLOP/Byte and a performance of {} GFLOP/s, does the roofline model consider the program as compute-bound or bandwidth-bound?'''

simpleAnswerTemplate='''Answer: {}'''

cotThoughtTemplate='''Thought: The max bandwidth is {} GB/s, and peak performance is {} GFLOP/s. The balance point is at {} / {} = {} FLOP/Byte. The program's Arithmetic Intensity is {} FLOP/Byte. Because {} {} {}, it is {} the balance point, putting the program in the {} region. The roofline model would consider the program as {}.'''


def gen_prompt_str(maxBandwidth, peakPerf, arithmInten, measuredPerf, promptType='simple'):
  # round all the input numbers
  maxBandwidth = float(format(maxBandwidth, '.2f'))
  peakPerf     = float(format(peakPerf,     '.2f'))
  arithmInten  = float(format(arithmInten,  '.2f'))
  measuredPerf = float(format(measuredPerf, '.2f'))

  question = questionTemplate.format(maxBandwidth, peakPerf, arithmInten, measuredPerf)

  # calculate numbers we will use for COT template
  balancePoint = format(float(peakPerf) / float(maxBandwidth), '.2f')

  if arithmInten <= float(balancePoint):
      inequality = '<'
      ltORgt = 'before'
      region = 'bandwidth-bound'
      answer = 'Bandwidth' 
  else:
      inequality = '>'
      ltORgt = 'after'
      region = 'compute-bound'
      answer = 'Compute' 

  if promptType == 'simple':
    return question + '\n' + simpleAnswerTemplate.format(answer)
  elif promptType == 'cot':
    return question + '\n' + cotThoughtTemplate.format(maxBandwidth, peakPerf, peakPerf, maxBandwidth, balancePoint,
                                                        arithmInten, 
                                                        arithmInten, inequality, balancePoint, ltORgt, region, region) \
                            + '\n' \
                            + simpleAnswerTemplate.format(answer)
  else:
    assert False, "invalid prompt type requested!"



# this will create a random roofline for testing
def gen_random_roofline(minBandwidth, maxBandwidth, minPeakPerf, maxPeakPerf):
    bandwidth = random.uniform(minBandwidth, maxBandwidth)
    peakPerf  = random.uniform(minPeakPerf, maxPeakPerf)

    return bandwidth, peakPerf

# this will take sample points of the desired type within the random roofline
def gen_x_samples_from_roofline(maxBandwidth, peakPerf, numSamples, type):
    samples = []

    # we do some rounding to make printing cleaner
    balancePoint = float(format(float(peakPerf) / float(maxBandwidth), '.2f'))

    while len(samples) < numSamples:
        if type == 'BB':
            # sample slightly before the balance point for safety
            ai = random.uniform(0, balancePoint*0.95)
            # find the max y value under the curve, sample slightly under it
            maxY = ai * maxBandwidth * 0.95
            perf = random.uniform(0, maxY)
            samples.append((ai,perf))
        elif type == 'CB':
            # first generate the AI point, the max we'll go is 10x away from the balance point
            # sample slightly after the balance point for safety
            ai = random.uniform(balancePoint*1.05, 10*balancePoint) 
            # next lets generate a random performance, sample slightly under the max roofline
            perf = random.uniform(0,peakPerf*0.95)
            samples.append((ai,perf))
        else:
            raise Exception(f'Can not generate X samples for given type: [{type}]')

    return samples


start_systemMessage = '''You are a GPU performance analysis expert that classifies kernels into Arithmetic Intensity Roofline model categories based on their source code characteristics. Your task is to provide one of the following performance boundedness classifications: Compute or Bandwidth. A kernel is considered Compute bound if its Arithmetic Intensity is greater than the balance point (ratio of machine peak performance and max bandwidth), otherwise the kernel is considered Bandwidth bound.

Provide only one word as your response, chosen from the set: ['Compute', 'Bandwidth'].
**Examples**'''

end_systemMessage = '''Now, answer the following arithmetic intensity classification question:'''

def make_system_message(examples, msgType='simple'):

  assert msgType in ['simple', 'cot']
  assert len(examples) > 0
  assert len(examples) % 2 == 0

  sysMessage = start_systemMessage + '\n\n'

  for idx, ex in enumerate(examples):
    maxBandwidth, peakPerf, arithmInten, measuredPerf = ex
    sysMessage += f'**Example {idx+1}:**\n' + gen_prompt_str(maxBandwidth, peakPerf, arithmInten, measuredPerf, promptType=msgType) + '\n\n'

  sysMessage += end_systemMessage

  return sysMessage


def make_user_message(target):
  maxBandwidth, peakPerf, arithmInten, measuredPerf = target

  maxBandwidth = float(format(maxBandwidth, '.2f'))
  peakPerf     = float(format(peakPerf,     '.2f'))
  arithmInten  = float(format(arithmInten,  '.2f'))
  measuredPerf = float(format(measuredPerf, '.2f'))

  return questionTemplate.format(maxBandwidth, peakPerf, arithmInten, measuredPerf)


async def make_chat_history(examples, target, msgType='simple'):

    systemMessage = make_system_message(examples, msgType)
    sys_msg = SystemMessage(content=systemMessage)
    usrMessage = make_user_message(target)
    usr_msg = UserMessage(source='User', content=usrMessage)

    context = UnboundedChatCompletionContext(initial_messages=[sys_msg, usr_msg])
    return context



def writeToFile(filename, lines):
    # going to overwrite the whole file each time
    # it's redundant but the file wont be that large
    # so the speed doesn't matter
    with open(filename, 'w') as jsonLFile:
        jsonLFile.write(lines)



def bytes_to_ascii(bytesList):
  if bytesList == None:
    return None
  else:
    bytes_arr = bytes(bytesList)
    # for some reason we get responses back with character codes in the `latin` set instead of `utf-8`
    # `utf-8` kept throwing errors when decoding
    return bytes_arr.decode('latin')

# logprobs [ChatCompletionTokenLogprob(token='Bandwidth', logprob=-0.00010902655776590109, top_logprobs=[TopLogprob(logprob=-0.00010902655776590109, bytes=[66, 97, 110, 100, 119, 105, 100, 116, 104]), TopLogprob(logprob=-9.25010871887207, bytes=[67, 111, 109, 112, 117, 116, 101])]

def parse_toplogprobs(ccTL):
  parsed = []
  for top in ccTL.top_logprobs:
    logprob = top.logprob
    tokens = bytes_to_ascii(top.bytes)

    if not (tokens is None):
      parsed.append((tokens, logprob))

  return parsed


def convert_logprobs_to_json_str(logProbsObj):
    if logProbsObj is None:
        return ''

    #print(logProbsObj)
    toReturn = {}
    for ccTL in logProbsObj:
        #print(ccTL)
        toReturn[ccTL.token] = {'logprob':ccTL.logprob, 'topLogprob':dict(parse_toplogprobs(ccTL))}
        #print(toReturn[ccTL.token])

    return json.dumps(toReturn)


 