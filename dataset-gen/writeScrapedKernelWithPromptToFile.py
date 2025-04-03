from roofline_utils import *

import time
import argparse

# ### Open the Trin/Val Data CSV Files
dtypes['language'] = 'string'
dtypes['numTokens'] = np.int64
dtypes['kernelCode'] = 'string'
dtypes['kernelSASS'] = 'string'
dtypes['isBB'] = np.int64
dtypes['class'] = 'string'
dtypes['answer'] = 'string'


async def make_query_chat_hist(dfRow, isZeroShot, useSASS=False):
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

    infoMsg = make_kernel_info_message(device, exeArgs, kernelName, blockSz, gridSz, language, useSASS=useSASS)

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
    return await chatHist.get_messages()


async def create_output_file(df, kernelName, targetName, isZeroShot, outfileName, useSASS=False):
    subdf = df[(df['kernelName'] == kernelName) & (df['targetName'] == targetName)]
    
    assert subdf.shape[0] == 1, f"subdf shape: {subdf.shape}"

    for index, row in subdf.iterrows():
        messages = await make_query_chat_hist(row, isZeroShot, useSASS)

        concat = '\n'.join([a.content for a in messages])

        with open(outfileName, 'w') as file:
            file.write(concat)

        return
    return


async def main():
    parser = argparse.ArgumentParser(description="A script to handle various arguments for model inference")
    
    parser.add_argument('--kernelName', type=str, default='mean_shift', help='Name of the kernel we want. Simplename for CUDA, mangled for OMP.')
    parser.add_argument('--targetName', type=str, default='meanshift-cuda', help='Name of the target executable')
    parser.add_argument('--zeroShot', action='store_true', default=False, help='Flag for zero-shot inference')
    parser.add_argument('--useSASS', action='store_true', default=False, help='Use SASS code instead of source')
    
    args = parser.parse_args()

    if args.zeroShot:
        outfilePrefix = 'zero-shot'
    else:
        outfilePrefix = 'few-shot'

    if args.useSASS:
        outfilePrefix += '-SASS-only'

    parser.add_argument('--outputFile', type=str, default=f'{outfilePrefix}-LLM-query__{args.targetName}__{args.kernelName}.txt', help='Output CSV file name')

    args = parser.parse_args()
    
    print('\n\nInput Arguments:\n------------------------------------------------')
    print(f"Kernel Name: {args.kernelName}")
    print(f"Target Name: {args.targetName}")
    print(f"Zero Shot: {args.zeroShot}")
    print(f"Output File: {args.outputFile}")
    print(f"Use SASS: {args.useSASS}")

    # we need to gather more data for this dataset
    trainDF = pd.read_csv('train-dataset-balanced.csv', quotechar='"', dtype=dtypes)
    valDF = pd.read_csv('validation-dataset-balanced.csv', quotechar='"', dtype=dtypes)

    trainDF['isTrain'] = 1
    valDF['isTrain'] = 0

    df = pd.concat([trainDF, valDF], ignore_index=True)

    print('------------------------------------------------\n\n')

    await create_output_file(df, args.kernelName, args.targetName, args.zeroShot, args.outputFile, args.useSASS)

if __name__ == "__main__":
    asyncio.run(main())
