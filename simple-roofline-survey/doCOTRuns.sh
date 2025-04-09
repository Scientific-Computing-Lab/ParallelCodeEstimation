#!/bin/bash


# non-reasoning models
python3 runRooflineSurvey.py --yes --modelName=openai/gpt-4o-mini-2024-07-18 --includeLogProbs --numTrials=40 --useCOT --temps 0.1 --topps 0.2
python3 runRooflineSurvey.py --yes --modelName=openai/gpt-4o-mini --includeLogProbs --numTrials=40 --useCOT --temps 0.1 --topps 0.2
python3 runRooflineSurvey.py --yes --modelName=openai/gpt-4o-2024-11-20 --includeLogProbs --numTrials=40 --useCOT --temps 0.1 --topps 0.2

python3 runRooflineSurvey.py --yes --modelName=google/gemini-2.0-flash-001 --includeLogProbs --numTrials=40 --useCOT --temps 0.1 --topps 0.2


# reasoning models
python3 runRooflineSurvey.py --yes --modelName=openai/o3-mini --reasoning --includeLogProbs --numTrials=40 --useCOT
python3 runRooflineSurvey.py --yes --modelName=openai/o3-mini-high --reasoning --includeLogProbs --numTrials=40 --useCOT
python3 runRooflineSurvey.py --yes --modelName=openai/o1-mini-2024-09-12 --reasoning --includeLogProbs --numTrials=40 --useCOT 

python3 runRooflineSurvey.py --yes --modelName=deepseek/deepseek-r1-distill-qwen-32b --reasoning --includeLogProbs --numTrials=40 --useCOT 