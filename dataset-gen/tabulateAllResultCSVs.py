from roofline_utils import *
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import re
import argparse 
import glob

# ## Open the CSV file

# 

#resultsCSV = 'few-shot-inference-results.csv'
#resultsCSV = 'few-shot-inference-results-gemini-flash-2.csv'
#resultsCSV = 'few-shot-inference-results-o3-mini.csv'
#resultsCSV = 'few-shot-inference-results-o3-mini-high.csv'
#resultsCSV = 'few-shot-inference-results-o1-mini.csv'
#resultsCSV = 'few-shot-inference-results-o1.csv'
#resultsCSV = 'few-shot-inference-results-gpt-4o-2024-11-20.csv'
#resultsCSV = 'few-shot-inference-results-gpt-4o-mini.csv'
#resultsCSV = 'few-shot-inference-results-gpt-4o-mini-2024-07-18.csv'
#resultsCSV = 'zero-shot-inference-results-gpt-4o-mini-2024-07-18.csv'
#resultsCSV = 'zero-shot-inference-results-gpt-4o-mini.csv'
#resultsCSV = 'zero-shot-inference-results-gemini-2.0-flash-001.csv'
#resultsCSV = 'zero-shot-inference-results-gpt-4o-2024-11-20.csv'
#resultsCSV = 'zero-shot-inference-results-o1-mini-2024-09-12.csv'
#resultsCSV = 'zero-shot-inference-results-o3-mini.csv'
#resultsCSV = 'few-shot-inference-results-deepseek-r1-distill-qwen-32b.csv'
#resultsCSV = 'zero-shot-inference-results-o3-mini-high.csv'
#resultsCSV = 'zero-shot-SASS-only-inference-results-gemini-2.0-flash-001.csv'
#resultsCSV = 'zero-shot-SASS-only-inference-results-o3-mini-high.csv'
#resultsCSV = 'zero-shot-SASS-only-inference-results-o3-mini.csv'
#resultsCSV = 'zero-shot-SASS-only-inference-results-gpt-4o-mini.csv'
#resultsCSV = 'zero-shot-SASS-only-inference-results-gpt-4o-2024-11-20.csv'
#resultsCSV = 'zero-shot-inference-results-gpt-4.5-preview.csv'
#resultsCSV = 'few-shot-inference-results-gpt-4.5-preview.csv'



dtypes['language'] = 'string'
dtypes['numTokens'] = np.int64
dtypes['kernelCode'] = 'string'
dtypes['kernelSASS'] = 'string'
dtypes['isBB'] = np.int64
dtypes['class'] = 'string'
dtypes['answer'] = 'string'
dtypes['topp'] = np.float64
dtypes['temp'] = np.float64
dtypes['llmResponse'] = 'string'
dtypes['llmThought'] = 'string'
dtypes['trial'] = np.int64
dtypes['isTrain'] = np.int64



# 
# need to cleanup the LLM responses
def cleanup_responses(x):
    #print('input:', x)
    if not (str(x) == '<NA>'):
        matches = re.finditer(r'([Bb]andwidth|[Cc]ompute)', x, re.MULTILINE)
        matches = [m for m in matches]
        if len(matches) == 0:
            return '<NA>'
        if len(matches) > 1:
            # just take the last match
            print('\tMore than 1 match, taking last one!')
            matches = [matches[-1]]
        else:
            assert len(matches) == 1
        for match in matches:
            m = match.group()
            return m.title()

    print(f'returning NA for [{x}]')
    assert False, "this should never be reached!"
    return 'NA'

#
## 
#print(f'Accuracy: {100*df[df["isLLMCorrect"] == True]["isLLMCorrect"].count()/df.shape[0]}')
#
#numCUDACorrect = df[(df["isLLMCorrect"] == True) & (df["language"] == 'CUDA')]["isLLMCorrect"].count()
#numCUDA = df[(df["language"] == 'CUDA')]["isLLMCorrect"].count()
#print(f'CUDA Accuracy: {100*numCUDACorrect/numCUDA}')
#
#numOMPCorrect = df[(df["isLLMCorrect"] == True) & (df["language"] == 'OMP')]["isLLMCorrect"].count()
#numOMP = df[(df["language"] == 'OMP')]["isLLMCorrect"].count()
#print(f'OMP Accuracy: {100*numOMPCorrect/numOMP}')
#
## let's look at some of the failed examples
#
#failedCUDA = df[(df['language'] == 'CUDA') & (df['isLLMCorrect'] == False)]
#failedCUDA.to_csv(f'failed-CUDA-{resultsCSV}', quoting=csv.QUOTE_NONNUMERIC, quotechar='\"', index=False, na_rep='NULL')
#print(failedCUDA['targetName'].to_list())
#
#
#failedOMP = df[(df['language'] == 'OMP') & (df['isLLMCorrect'] == False)]
#failedOMP.to_csv(f'failed-OMP-{resultsCSV}', quoting=csv.QUOTE_NONNUMERIC, quotechar='\"', index=False, na_rep='NULL')
#print(failedOMP['targetName'].to_list())
#
#
## 
#def make_confusion_plot(df):
#    cf_matrix = confusion_matrix(df['answer'], df['llmResponse'], labels=['Compute', 'Bandwidth'])
#    print(cf_matrix)
#
#    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
#    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
#    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts,group_percentages)]
#    labels = np.asarray(labels).reshape(2,2)
#
#    return
#
## 
#
#
#ompSubset = df[df['language'] == 'OMP']
#print('Only OMP')
#make_confusion_plot(ompSubset)
#
#cudaSubset = df[df['language'] == 'CUDA']
#print('Only CUDA')
#make_confusion_plot(cudaSubset)
#
#print('whole DF')
#make_confusion_plot(df)
#
## 
#
#class_report = classification_report(df['answer'], df['llmResponse'], labels=['Compute', 'Bandwidth'], target_names=['Compute-Bound', 'Bandwidth-Bound'])
#print(class_report)
#
## precision: Percentage of correct predictions relative to total predictions
#
#
## Should we do Bonferroni correction for this test?
#
#
##  [markdown]
## ## Area Under Curve + ROC Curve
#
## 
## Encode the categorical string labels to integers
#label_encoder = LabelEncoder()
#true_labels_encoded = label_encoder.fit_transform(df['answer'])
#predicted_labels_encoded = label_encoder.transform(df['llmResponse'])
#
## Calculate ROC AUC score
#roc_auc = roc_auc_score(true_labels_encoded, predicted_labels_encoded)
#print(f"ROC AUC Score: {roc_auc}")
#
## Calculate ROC curve
#fpr, tpr, thresholds = roc_curve(true_labels_encoded, predicted_labels_encoded)
#
#print('fpr', fpr)
#print('tpr', tpr)
#print('thesholds', thresholds)


def read_results_csv(csvFile):
    isZeroShot = False
    isSASS = False
    isTrainedModel = False
    if 'zero-shot-' in csvFile:
        isZeroShot = True
        if 'SASS-only' in csvFile:
            isSASS = True
            modelName = csvFile[len('zero-shot-SASS-only-inference-results-'):-4]
        else:
            modelName = csvFile[len('zero-shot-inference-results-'):-4]
    elif 'few-shot-' in csvFile:
        if 'SASS-only' in csvFile:
            isSASS = True
            modelName = csvFile[len('few-shot-SASS-only-inference-results-'):-4]
        else:
            modelName = csvFile[len('few-shot-inference-results-'):-4]
    else:
        assert 'trainedModel-inference-BALANCED-training-results-' in csvFile, "Don't recognize input CSV file"
        isTrainedModel = True
        modelName = csvFile[len('trainedModel-inference-BALANCED-training-results-'):-4]


    df = pd.read_csv(csvFile, quotechar='\"', dtype=dtypes)
    
    #print(df.shape)
    #print(df.columns)
    
    df.drop(['kernelCode'], axis=1, inplace=True)
    
    #print(df.shape)
    #print(df.columns)

    # let's just drop the failed cases for now
    #badCases = df[df['llmResponse'].isna()]
    #print('badCases:', badCases[['Kernel Name', 'llmResponse']])
    df = df.dropna(subset=['llmResponse'])

    # do some response cleanup for returned strings that have more than 1 token
    df['llmResponse'] = df['llmResponse'].apply(cleanup_responses)

    # check if the LLM produced the correct answer
    df['isLLMCorrect'] = df.apply(lambda x: x['answer'] == x['llmResponse'], axis=1)

    # some of the 'llmResponse' columns are '<NA>' after cleanup for no matches
    # make the response the opposite value -- so it's counted as wrong
    df['llmResponse'] = df.apply(lambda x: x['answer'] if x['isLLMCorrect'] else ('Compute' if x['answer'] == 'Bandwidth' else 'Bandwidth'), axis=1)
    
    return (df, modelName, isZeroShot, isSASS, isTrainedModel)


def find_all_csvs_of_type(fewShot, zeroShot, trained, onlySASS):
    # we assume all the data CSVs are in the current directory
    # we're going to amalgamate results from CSVs that share the same model
    # therefore be careful calling this with fewShot and zeroShot enabled
    # since it'll join both results
    csvFiles = []
    if fewShot:
        if onlySASS:
            csvs = list(glob.glob('few-shot-SASS-only-inference-results-*.csv'))
        else:
            csvs = list(glob.glob('few-shot-inference-results-*.csv'))
        csvFiles += csvs
    if zeroShot:
        if onlySASS:
            csvs = list(glob.glob('zero-shot-SASS-only-inference-results-*.csv'))
        else:
            csvs = list(glob.glob('zero-shot-inference-results-*.csv'))
        csvFiles += csvs
    if trained:
        # didn't train with SASS, so we omit the SASS check
        csvs = list(glob.glob('trainedModel-inference-BALANCED-training-results-*.csv'))
        csvFiles += csvs

        assert (not fewShot) and (not zeroShot) and (onlySASS), "We didn't gather SASS inference data on trained models"

    assert len(csvFiles) != 0, "Didn't find any CSV files, aborting"
    return csvFiles


def calc_metrics_of_df(df):

    # Calculate accuracy
    accuracy = accuracy_score(df['actual'], df['predicted'])
    # Calculate precision
    #precision = precision_score(df['actual'], df['predicted'])
    #print('precisionA', precision)
    #print('precisionB-1', precision_score(df['actual'], df['predicted'], pos_label=1))
    #print('precisionB-2', precision_score(df['actual'], df['predicted'], pos_label=2))
    ## Calculate recall
    #recall = recall_score(df['actual'], df['predicted'])

    f1 = f1_score(df['actual'], df['predicted'], average='macro')

    mcc = matthews_corrcoef(df['actual'], df['predicted'])

    # Print the results
    #print(f'Accuracy: {accuracy:.2f}')
    #print(f'Precision: {precision:.2f}')
    #print(f'Recall: {recall:.2f}')

    return (accuracy, f1, mcc)

def calculate_metrics(csvFiles):

    dfStats = pd.DataFrame()

    # each CSV file will get it's own row in the stats table
    for csvName in csvFiles:
        print('Calculating metrics for ', csvName)
        csvDF, modelName, isZeroShot, isSASS, isTrained = read_results_csv(csvName)
        print('Read CSV complete for model:', modelName)

        summDict = {}
        summDict['Model Name'] = [modelName]
        summDict['Is Zero Shot?'] = [isZeroShot]
        summDict['Is SASS?'] = [isSASS]
        summDict['Used Trained Model?'] = [isTrained]
        summDict['Number of Samples'] = csvDF.shape[0]

        summDF = pd.DataFrame.from_dict(summDict)

        # add some columns for ease-of-calculations
        csvDF['actual'] = csvDF['answer'].apply(lambda x: 1 if x == 'Bandwidth' else 2)
        csvDF['predicted'] = csvDF['llmResponse'].apply(lambda x: 1 if x == 'Bandwidth' else 2)

        cudaOnly = csvDF[csvDF['language'] == 'CUDA'].reset_index(drop=True)
        cudaMetrics = calc_metrics_of_df(cudaOnly)

        ompOnly = csvDF[csvDF['language'] == 'OMP'].reset_index(drop=True)
        ompMetrics = calc_metrics_of_df(ompOnly)

        jointMetrics = calc_metrics_of_df(csvDF)

        #assert ompMetrics[0] != cudaMetrics[0], "Are you sure these are supposed to be the same accuracy for CUDA and OMP?"

        summDF['CUDA Acc'] = round(100.0*cudaMetrics[0],2)
        #summDF['CUDA Perc'] = round(100.0*cudaMetrics[1],2)
        #summDF['CUDA Recall'] = round(100.0*cudaMetrics[2],2)
        summDF['CUDA F1'] = round(100.0*cudaMetrics[1],2)
        summDF['CUDA MCC'] = round(100.0*cudaMetrics[2],2)

        summDF['OMP Acc'] = round(100.0*ompMetrics[0],2)
        #summDF['OMP Perc'] = round(100.0*ompMetrics[1],2)
        #summDF['OMP Recall'] = round(100.0*ompMetrics[2],2)
        #summDF['OMP F1'] = round(100.0*ompMetrics[3],2)
        summDF['OMP F1'] = round(100.0*ompMetrics[1],2)
        summDF['OMP MCC'] = round(100.0*ompMetrics[2],2)

        summDF['Joint Acc'] = round(100.0*jointMetrics[0],2)
        #summDF['Joint Perc'] = round(100.0*jointMetrics[1],2)
        #summDF['Joint Recall'] = round(100.0*jointMetrics[2],2)
        #summDF['Joint F1'] = round(100.0*jointMetrics[3],2)
        summDF['Joint F1'] = round(100.0*jointMetrics[1],2)
        summDF['Joint MCC'] = round(100.0*jointMetrics[2],2)

        summDF['Diff Acc'] = round(summDF['CUDA Acc'] - summDF['OMP Acc'], 2)
        summDF['Diff F1'] = round(summDF['CUDA F1'] - summDF['OMP F1'], 2)
        summDF['Diff MCC'] = round(summDF['CUDA MCC'] - summDF['OMP MCC'], 2)

        dfStats = pd.concat([dfStats, summDF], ignore_index=True)


    return dfStats

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fewShot', action='store_true', default=False, help='Make table for few-shot data')
    parser.add_argument('--zeroShot', action='store_true', default=False, help='Make table for zero-shot data')
    parser.add_argument('--trained', action='store_true', default=False, help='Make table for trained-model inference data')
    parser.add_argument('--onlySASS', action='store_true', default=False, help='Only read the data files that used SASS')

    args = parser.parse_args()

    outfileName = './allResultsMetrics'
    if args.zeroShot:
        outfileName += '-zeroShot'
    if args.fewShot:
        outfileName += '-fewShot'
    if args.trained:
        outfileName += '-trained'
    if args.onlySASS:
        outfileName += '-onlySASS'

    outfileName += '.csv'

    print('Input Args:')
    print('Few Shot', args.fewShot)
    print('Zero Shot', args.zeroShot)
    print('Trained', args.trained)
    print('Only SASS', args.onlySASS)
    print('Output File', outfileName)

    csvFiles = find_all_csvs_of_type(args.fewShot, args.zeroShot, args.trained, args.onlySASS) 
    print('Found CSVs:')
    for filename in csvFiles:
        print(f'\n\t {filename}')

    df = calculate_metrics(csvFiles)

    df = df.sort_values(by=['Joint Acc', 'Model Name', 'Number of Samples'], ignore_index=True, ascending=False)
    print(df)

    df.to_csv(outfileName, quoting=csv.QUOTE_NONNUMERIC, quotechar='"', index=False, na_rep='NULL')
    print('Saved Summary Metrics to:', outfileName)



    return


if __name__ == "__main__":
    main()

