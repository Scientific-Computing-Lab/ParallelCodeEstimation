from roofline_utils import *

import time
import argparse

import openai
import sys

# please create a file called '.llm-api-key' with your api key and no newline characters
with open('./.llm-api-key', 'r') as file:
    LLM_API_KEY=file.read().strip()

with open('./.openrouter-api-key', 'r') as file:
    OPENROUTER_API_KEY=file.read().strip()

# ### Open the Trin/Val Data CSV Files
dtypes['language'] = 'string'
dtypes['numTokens'] = np.int64
dtypes['kernelCode'] = 'string'
dtypes['kernelSASS'] = 'string'
dtypes['isBB'] = np.int64
dtypes['class'] = 'string'
dtypes['answer'] = 'string'



async def ask_llm_for_roofline_classification(chatHistory, modelName, useAzure=False, temp=1.0, topp=0.1, timeout=60, storeLogProbs=False):

    model_client = None
    logprob_args = {}
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
                model_info = {'vision':False, 'function_calling':True, 'json_output':False, 'family':'unknown'}
        )

    #agent = AssistantAgent(
    #    name="assistant",
    #    model_client=model_client,
    #    model_context=chatHistory
    #)

    #response = await agent.run()
    #for msg in response.messages:
    #    print('msg', msg)
    #    print()

    #result = await model_client.create(messages = await chatHistory.get_messages(), extra_create_args={'logprobs':True, 'top_logprobs':10})
    result = await model_client.create(messages = await chatHistory.get_messages())
    #print('type', type(result))
    #print('cached', result.cached)
    #print('thought', result.thought)
    #print('content', result.content)
    #print('usage', result.usage)
    #print('logprobs', result.logprobs)
    #print('finish_reason', result.finish_reason)
    #print('result modeldump', result.model_dump_json())
    #print('result modeldump', result.model_dump())

    #return agent._model_context
    await chatHistory.add_message(result)
    return chatHistory, result.logprobs



async def run_row_trial(dfRow, modelName, temp, topp, useAzure, isZeroShot, useSASS, storeLogProbs):
    #targetName = dfRow['targetName']
    kernelName = dfRow['Kernel Name']
    exeArgs = dfRow['exeArgs']
    blockSz = dfRow['Block Size']
    gridSz = dfRow['Grid Size']
    language = dfRow['language']
    device = dfRow['device']

    if useSASS:
        kernelCode = dfRow['kernelSASS']
    else:
        kernelCode = dfRow['kernelCode']

    infoMsg = make_kernel_info_message(device, exeArgs, kernelName, blockSz, gridSz, language, useSASS)

    if isZeroShot:
        chatHist = await make_chat_history(infoMsg, kernelCode, 0, useSASS=useSASS)
    # only include OMP examples
    elif language == 'OMP':
        chatHist = await make_chat_history(infoMsg, kernelCode, 2, useSASS=useSASS)
    # only include CUDA examples
    else:
        assert language == 'CUDA'
        chatHist = await make_chat_history(infoMsg, kernelCode, 3, useSASS=useSASS)

    assert chatHist != None
    resultHist, logProbs = await ask_llm_for_roofline_classification(chatHist, modelName, useAzure=useAzure, temp=temp, topp=topp, timeout=120, storeLogProbs=storeLogProbs)
    assert resultHist != None

    resultMessages = await resultHist.get_messages()
    return resultMessages, logProbs


async def run_row_trial_task(resultsDF, modelName, inRow, trial, temp, topp, useAzure, isZeroShot, useSASS, storeLogProbs):
    targetName = inRow['targetName']
    kernelName = inRow['Kernel Name']

    # check if we already sampled this point
    if is_already_sampled(resultsDF, inRow, trial, temp, topp):
        print(f'Already sampled, skipping \t{targetName}: [{kernelName}]')
        return None

    try:
        resultMsgs, logProbs = await run_row_trial(inRow, modelName, temp, topp, useAzure, isZeroShot, useSASS, storeLogProbs)
    except TypeError as e:
        print(f'1 Unable to gather sample {targetName} -- [{kernelName}] it may exceed input context limits')
        print(e)
        return None
    except openai.BadRequestError as e:
        print(f'2 Unable to gather sample {targetName} -- [{kernelName}] it may exceed input context limits')
        print(e)
        return None

    # make a copy so we can modify it
    row = inRow.copy().to_frame().T.reset_index(drop=True)

    assert row.shape[0] == 1

    row['trial'] = trial
    row['topp'] = topp
    row['temp'] = temp
    row['llmResponse'] = ''
    row['llmThought'] = ''
    row['logprobs'] = ''

    # get the last message (it's the answer)
    #pprint(resultMsgs)

    # the last message contains the LLM response and thought message
    resultStr = resultMsgs[-1].content
    thoughtStr = resultMsgs[-1].thought

    if not (resultStr in ['Compute', 'Bandwidth']):
        print(f'{targetName}: [{kernelName}] bad response: [{resultStr}]')

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

    return row


# collect data
async def run_all_trials(df, resultsCSV, modelName, temps, topPs, numTrials, postQuerySleepTime, useAzure, isZeroShot, useSASS, storeLogProbs):
    # if we already captured some data
    if os.path.isfile(resultsCSV):
        dtypes['topp'] = np.float64
        dtypes['temp'] = np.float64
        dtypes['llmResponse'] = 'string'
        dtypes['llmThought'] = 'string'
        dtypes['logprobs'] = 'string'
        dtypes['trial'] = np.int64
        dtypes['isTrain'] = np.int64

        resultsDF = pd.read_csv(resultsCSV, quotechar='\"', dtype=dtypes)
        # drop any rows with NA responses, will need to regather
        resultsDF = resultsDF.dropna(subset=['llmResponse'])
    else:
        # setup the resultsDF
        resultsDF = pd.DataFrame()

    # calculate how many total iters
    totalQueries = numTrials * len(temps) * len(topPs) * df.shape[0]

    # add a row to the dataframe keeping track of the returned result
    with tqdm(total=totalQueries) as pbar:
        for trial in range(numTrials):
            for temp in temps:
                for topp in topPs:
                    for index, row in df.iterrows():

                        newRow = await run_row_trial_task(resultsDF, modelName, row, trial, temp, topp, useAzure, isZeroShot, useSASS, storeLogProbs)

                        #print()
                        #print(newRow)
                        #print(newRow['logprobs'][0])

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


async def main():
    global LLM_API_KEY
    global OPENROUTER_API_KEY

    parser = argparse.ArgumentParser(description="A script to handle various arguments for model inference")
    
    parser.add_argument('--modelName', type=str, default='openai/o3-mini', help='Name of the model')
    parser.add_argument('--apiKey', type=str, default='', help='User-provided API key')
    parser.add_argument('--useAzure', action='store_true', default=False, help='Flag to use Azure')
    parser.add_argument('--useSASS', action='store_true', default=False, help='Use SASS code instead of source')
    parser.add_argument('--includeLogProbs', action='store_true', default=False, help='Record Lob Probabilities for Tokens')
    parser.add_argument('--zeroShot', action='store_true', default=False, help='Flag for zero-shot inference')
    parser.add_argument('--postQuerySleep', type=float, default=0.5, help='Sleep time after each query')
    parser.add_argument('--numTrials', type=int, default=1, help='Number of trials')
    parser.add_argument('--temps', type=float, nargs='+', default=[0.1], help='List of temperature values')
    parser.add_argument('--topps', type=float, nargs='+', default=[0.2], help='List of top-p values')
    
    args = parser.parse_args()

    if args.zeroShot:
        outcsvPrefix = 'zero-shot'
    else:
        outcsvPrefix = 'few-shot'

    if args.useSASS:
        outcsvPrefix += '-SASS-only'

    if args.includeLogProbs:
        outcsvPrefix += '-withLogProbs'

    parser.add_argument('--outputCSV', type=str, default=f'{outcsvPrefix}-inference-results-{args.modelName.split("/")[-1]}.csv', help='Output CSV file name')

    args = parser.parse_args()
    
    print('\n\nData Collection Parameters:\n------------------------------------------------')
    print(f"Model Name: {args.modelName}")
    print(f"Use Azure: {args.useAzure}")
    print(f"Use SASS: {args.useSASS}")
    print(f"Zero Shot: {args.zeroShot}")
    print(f"Include Log Probs: {args.includeLogProbs}")
    print(f"Output CSV: {args.outputCSV}")
    print(f"Post Query Sleep: {args.postQuerySleep}")
    print(f"Number of Trials: {args.numTrials}")
    print(f"Temperatures: {args.temps}")
    print(f"Top-p Values: {args.topps}")

    if args.apiKey != '':
        print(f"User-provided API Key: [{args.apiKey}]")
        LLM_API_KEY = args.apiKey
        OPENROUTER_API_KEY = args.apiKey

    # we need to gather more data for this dataset
    trainDF = pd.read_csv('train-dataset-balanced.csv', quotechar='"', dtype=dtypes)
    valDF = pd.read_csv('validation-dataset-balanced.csv', quotechar='"', dtype=dtypes)

    trainDF['isTrain'] = 1
    valDF['isTrain'] = 0

    df = pd.concat([trainDF, valDF], ignore_index=True)

    print('------------------------------------------------\n\n')
    print('Will start collecting data, press ENTER to confirm settings are correct!')
    input()
    print('Starting data collection!')

    await run_all_trials(df, args.outputCSV, args.modelName, args.temps, args.topps, args.numTrials, args.postQuerySleep, args.useAzure, args.zeroShot, args.useSASS, args.includeLogProbs)

if __name__ == "__main__":
    asyncio.run(main())