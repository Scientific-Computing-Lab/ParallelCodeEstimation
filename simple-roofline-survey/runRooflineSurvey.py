
from roofline_survey_utils import *

import time
import argparse

import openai
import sys
import copy

# please create a file called '.llm-api-key' with your api key and no newline characters
with open('../dataset-gen/.llm-api-key', 'r') as file:
    LLM_API_KEY=file.read().strip()

with open('../dataset-gen/.openrouter-api-key', 'r') as file:
    OPENROUTER_API_KEY=file.read().strip()

IS_REASONING_MODEL = False

async def ask_llm_for_roofline_classification(chatHistory, modelName, useAzure=False, temp=1.0, topp=0.1, timeout=60, storeLogProbs=False):

    model_client = None
    logprob_args = {}
    temp_args = {}

    if not IS_REASONING_MODEL:
        temp_args = {'top_p':topp, 'temperature':temp}
        # reasoning models don't let us save logprobs
        if storeLogProbs:
            logprob_args = {'logprobs': storeLogProbs, 'top_logprobs': 4}

    if useAzure:
        model_client = AzureOpenAIChatCompletionClient(
                # https://galor-m6d0ej1n-eastus2.cognitiveservices.azure.com/openai/deployments/o1/chat/completions?api-version=2024-12-01-preview
                model=modelName,
                azure_endpoint='https://galor-m6d0ej1n-eastus2.cognitiveservices.azure.com',
                #azure_endpoint='https://galor-m6d0ej1n-eastus2.cognitiveservices.azure.com',
                azure_deployment=modelName,
                api_key=LLM_API_KEY,
                timeout=timeout,
                #temperature=temp,
                #top_p = topp,
                #api_version='2024-12-01-preview',
                api_version='2025-01-01-preview',
                **logprob_args,
                **temp_args,
                model_info = {'vision':False, 'function_calling':True, 'json_output':True, 'family':'unknown'}
        )
    else:
        model_client = OpenAIChatCompletionClient(
                #model='openai/gpt-4o-mini',
                #model='openai/gpt-4o-mini-2024-07-18',
                #model='google/gemini-2.0-flash-001',
                #model='openai/o3-mini',
                #model='openai/gpt-4o-2024-11-20',
                #model='deepseek/deepseek-r1',
                #model='openai/o3-mini-high',
                #model='openai/o1-mini-2024-09-12',
                model=modelName,
                base_url='https://openrouter.ai/api/v1',
                api_key=OPENROUTER_API_KEY,
                timeout=timeout,
                #top_p = topp,
                #temperature=temp,
                **logprob_args,
                **temp_args,
                model_info = {'vision':False, 'function_calling':True, 'json_output':True, 'family':'unknown'}
        )

    #result = await model_client.create(messages = await chatHistory.get_messages(), extra_create_args={'logprobs':True, 'top_logprobs':10})
    result = await model_client.create(messages = await chatHistory.get_messages())

    #return agent._model_context
    await chatHistory.add_message(result)
    return chatHistory, result.logprobs



#resultMsgs, logProbs = await run_row_trial(modelName, temp, topp, useAzure, useCOT, storeLogProbs, chatHist)
async def run_row_trial(modelName, temp, topp, useAzure, storeLogProbs, chatHist):

    assert chatHist != None
    resultHist, logProbs = await ask_llm_for_roofline_classification(chatHist, modelName, useAzure=useAzure, temp=temp, topp=topp, timeout=120, storeLogProbs=storeLogProbs)
    assert resultHist != None

    resultMessages = await resultHist.get_messages()
    return resultMessages, logProbs


def is_already_sampled(df, trial, temp, topp, numExamples, answer):

    if df.shape[0] == 0:
        return False

    resultRow = df[(df['temp'] == temp) & (df['topp'] == topp) & (df['trial'] == trial) & (df['numExamples'] == numExamples) & (df['answer'] == answer)]

    if resultRow.shape[0] == 0:
        return False

    assert resultRow.shape[0] == 1, f"resultRow.shape = {resultRow.shape}"

    response = resultRow.iloc[0]['llmResponse']

    return response != ''


#newRow = await run_row_trial_task(resultsDF, modelName, trial, temp, topp, numExamples, useAzure, useCOT, storeLogProbs, chatHist)
async def run_row_trial_task(resultsDF, modelName, trial, temp, topp, numExamples, useAzure, useCOT, storeLogProbs, chatHist, answer):

    sampleName = f'Trial: [{trial}], Temp: [{temp}], Topp: [{topp}], Num Examples: [{numExamples}], Answer: [{answer}]'

    # check if we already sampled this point
    if is_already_sampled(resultsDF, trial, temp, topp, numExamples, answer):
        print(f'Already sampled, skipping... \t{sampleName}')
        return None

    # make a copy of the chatHist object, as it gets re-used between trials
    chatHist = copy.deepcopy(chatHist)

    # make a dictionary with all the trial info
    # this will be turned into a DF and appended to the
    # output df
    row = {'topp':[topp],
           'temp': [temp],
           'chatHistory':[chat_history_to_json_line(await chatHist.get_messages())],
           'trial': trial,
           'answer': answer,
           'useCOT': useCOT,
           'numExamples':numExamples}

    try:
        resultMsgs, logProbs = await run_row_trial(modelName, temp, topp, useAzure, storeLogProbs, chatHist)
    except TypeError as e:
        print(f'Unable to gather sample {sampleName} -- it may exceed input context limits')
        print(e)
        return None
    except openai.BadRequestError as e:
        print(f'Unable to gather sample {sampleName} -- Got a bad server response!')
        print(e)
        return None

    # the last message contains the LLM response and thought message
    resultStr = resultMsgs[-1].content
    thoughtStr = resultMsgs[-1].thought

    if not (resultStr in ['Compute', 'Bandwidth']):
        print(f'{sampleName}: BAD response: [{resultStr}]')

    # we'll re-do the run later
    if (resultStr == ''):
        print('Please re-run this script later to retry this sample.')
        # we will still save the response to the CSV file
        #pbar.update(1)
        #continue 

    if thoughtStr == None:
        thoughtStr = ''

    row['llmResponse'] = resultStr
    row['llmThought'] = thoughtStr
    if storeLogProbs:
        row['logprobs'] = convert_logprobs_to_json_str(logProbs)
    else:
        row['logprobs'] = ''

    return pd.DataFrame(row)


# collect data
    #await run_all_trials(args.outputCSV, args.modelName, args.temps, args.topps, chatHistories, args.postQuerySleep, args.useAzure, args.useCOT , args.includeLogProbs)
async def run_all_trials(resultsCSV, modelName, temps, topPs, chatHistories, postQuerySleepTime, useAzure, useCOT, storeLogProbs):

    dtypes = {}
    # if we already captured some data
    if os.path.isfile(resultsCSV):
        dtypes['topp'] = np.float64
        dtypes['temp'] = np.float64
        dtypes['chatHistory'] = 'string'
        dtypes['llmResponse'] = 'string'
        dtypes['llmThought'] = 'string'
        dtypes['logprobs'] = 'string'
        dtypes['trial'] = np.int64
        dtypes['numExamples'] = np.int64
        dtypes['useCOT'] = np.bool

        resultsDF = pd.read_csv(resultsCSV, quotechar='\"', dtype=dtypes)
        # drop any rows with NA responses, will need to regather
        resultsDF = resultsDF.dropna(subset=['llmResponse'])
    else:
        # setup the resultsDF
        resultsDF = pd.DataFrame()

    # calculate how many total iters
    totalQueries = len(chatHistories) * len(temps) * len(topPs) 

    # each trial will be random -- because the models have caching so they always return the same result
    # add a row to the dataframe keeping track of the returned result
    with tqdm(total=totalQueries) as pbar:
        for chatHistItem in chatHistories:
            numExamples, trial, answer, chatHist = chatHistItem

            for temp in temps:
                for topp in topPs:
                    newRow = await run_row_trial_task(resultsDF, modelName, trial, temp, topp, numExamples, useAzure, useCOT, storeLogProbs, chatHist, answer)

                    if newRow is None:
                        pbar.update(1)
                        continue 

                    resultsDF = pd.concat([resultsDF, newRow], ignore_index=True)

                    # spam save the CSV -- it's a small amount of data so it's not much of a time-sink
                    # it'll also help slow down quickly querying the model so we don't get cloudflare banned
                    resultsDF.to_csv(resultsCSV, quoting=csv.QUOTE_NONNUMERIC, quotechar='\"', index=False, na_rep='NULL')

                    pbar.update(1)
                    time.sleep(postQuerySleepTime)
    return


async def generate_chat_histories(numTrials, exampleCounts, msgType='simple', seed=449493):

    for i in exampleCounts:
        assert i % 2 == 0
        assert i > 0

    assert msgType in ['cot', 'simple']

    # set the seed to get consistent/repeatable generations
    random.seed(seed)

    # the examples will be the same across numExamples and temps and topPs
    maxExamples = max(exampleCounts)
    examples = []

    # each example in the example bank will be different
    for i in range(0, maxExamples, 2):
        maxBandEx, peakPerfEx = gen_random_roofline(50.0, 893.8, 2535.8, 7068.9)
        # these are (ai,perf) tuples
        bbSample = gen_x_samples_from_roofline(maxBandEx, peakPerfEx, 1, 'BB')
        cbSample = gen_x_samples_from_roofline(maxBandEx, peakPerfEx, 1, 'CB')
        examples.append((maxBandEx, peakPerfEx, bbSample[0][0], bbSample[0][1]))
        examples.append((maxBandEx, peakPerfEx, cbSample[0][0], cbSample[0][1]))

    assert len(examples) == maxExamples, f"{len(examples)}, {maxExamples}"

    # set the seed to get consistent/repeatable generations
    # this will allow us to up the number of trials later and keep
    # adding to the same dataframe
    random.seed(seed+1)

    trialValues = []
    # each trial is going to have a BB and CB question
    # thus, we double the number of trials
    for trial in range(numTrials):
        maxBand, peakPerf = gen_random_roofline(50.0, 893.8, 2535.8, 7068.9)
        bbSample = gen_x_samples_from_roofline(maxBand, peakPerf, 1, 'BB')
        cbSample = gen_x_samples_from_roofline(maxBand, peakPerf, 1, 'CB')
        trialValues.append((maxBand, peakPerf, bbSample[0][0], bbSample[0][1]))
        trialValues.append((maxBand, peakPerf, cbSample[0][0], cbSample[0][1]))


    # we're going to loop over this array for each temp and topp value
    histories = []
    for numExamples in exampleCounts:
        for idx, trialVal in enumerate(trialValues):
            trialIdx = idx // 2
            answer = 'Compute' if (idx % 2 == 1) else 'Bandwidth'
            chatData = await make_chat_history(examples[:numExamples], trialVal, msgType)
            histories.append((numExamples, trialIdx, answer, chatData))

    return histories


async def main():
    global LLM_API_KEY
    global OPENROUTER_API_KEY
    global IS_REASONING_MODEL

    parser = argparse.ArgumentParser(description="A script to handle various arguments for model inference")
    
    parser.add_argument('--modelName', type=str, default='openai/o3-mini', help='Name of the model')
    parser.add_argument('--apiKey', type=str, default='', help='User-provided API key')
    parser.add_argument('--useAzure', action='store_true', default=False, help='Flag to use Azure')
    parser.add_argument('--useCOT', action='store_true', default=False, help='Flag to use COT instead of simple prompt')
    parser.add_argument('--reasoning', action='store_true', default=False, help='Indicate if the model uses reasoning, to avoid passing topp and temp args')
    parser.add_argument('--yes', action='store_true', default=False, help='Skip user confirmation prior to running')
    parser.add_argument('--includeLogProbs', action='store_true', default=False, help='Record Lob Probabilities for Tokens')
    parser.add_argument('--postQuerySleep', type=float, default=0.5, help='Sleep time after each query')
    parser.add_argument('--numTrials', type=int, default=1, help='Number of trials -- each trial will use different values due to provider caching. Each trial will do 2 queries.')
    parser.add_argument('--temps', type=float, nargs='+', default=[0.1, 0.5, 1.0], help='List of temperature values')
    parser.add_argument('--topps', type=float, nargs='+', default=[0.2, 0.5, 0.9], help='List of top-p values')
    parser.add_argument('--exampleCounts', type=float, nargs='+', default=[2, 4, 8], help='List of top-p values')
    
    args = parser.parse_args()

    temps = [-1.0]
    topps = [-1.0]
    incLogProbs = False

    if args.reasoning:
        IS_REASONING_MODEL = True

    if args.useCOT:
        outcsvPrefix = 'COT'
    else:
        outcsvPrefix = 'simple'

    if not IS_REASONING_MODEL:
        temps = args.temps
        topps = args.topps
        if args.includeLogProbs:
            incLogProbs = True
            outcsvPrefix += '-withLogProbs'

    parser.add_argument('--outputCSV', type=str, default=f'{outcsvPrefix}-inference-results-{args.modelName.split("/")[-1]}.csv', help='Output CSV file name')

    args = parser.parse_args()
    
    print('\n\nData Collection Parameters:\n------------------------------------------------')
    print(f"Model Name: {args.modelName}")
    print(f"Use Azure: {args.useAzure}")
    print(f"Use COT: {args.useCOT}")
    print(f"Is Reasoning Model: {IS_REASONING_MODEL}")
    print(f"Include Log Probs: {args.includeLogProbs}")
    print(f"Output CSV: {args.outputCSV}")
    print(f"Post Query Sleep: {args.postQuerySleep}")
    print(f"Number of Trials: {args.numTrials}")
    print(f"Temperatures: {temps}")
    print(f"Top-p Values: {topps}")
    print(f"Example Counts: {args.exampleCounts}")

    if args.apiKey != '':
        print(f"User-provided API Key: [{args.apiKey}]")
        LLM_API_KEY = args.apiKey
        OPENROUTER_API_KEY = args.apiKey

    chatHistories = await generate_chat_histories(args.numTrials, args.exampleCounts, msgType='cot' if args.useCOT else 'simple')

    # check with the user before continuing
    print('------------------------------------------------\n\n')
    print('Will start collecting data, press ENTER to confirm settings are correct!')
    print('If output CSV file exists, will be appended to!')
    print('Total Number of LLM queries to be made:', len(chatHistories)*len(temps)*len(topps))

    if not args.yes:
        input()

    print('Starting data collection!')

    await run_all_trials(args.outputCSV, args.modelName, temps, topps, chatHistories, args.postQuerySleep, args.useAzure, args.useCOT, incLogProbs)

    print('Done gathering data! Results saved to:', args.outputCSV)

if __name__ == "__main__":
    asyncio.run(main())