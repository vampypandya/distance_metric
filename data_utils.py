import numpy as np
import pandas as pd
def getData(file_name, both = False):
    data = np.load(file_name)
    data = data.reshape(data.shape[0], -1)
    return data

def getHeader(file_name):
    lst_x = open(file_name, "r").read().split('\n')
    lst = [x+'_1' for x in lst_x] + [x+'_2' for x in lst_x]
    return lst

def getDataframes(data,lst):
    data1 = data[[x for x in lst if x[0] == 'i']]
    data2 = data[[x for x in lst if x[0] == 'a']]
    return data1, data2

def findCorrelation(real_train_file, syn_train_file, lst, threshold = 0.5):
    data = getData(real_train_file)
    data_syn = getData(syn_train_file)
    main_lst = lst
    # Update Real Training DataFrame and remove Zero columns
    df = pd.DataFrame(data, columns=main_lst)
    df, lst = removeZero(df)

    # Update Synthetic Training DataFrame and remove Zero columns
    df_syn = pd.DataFrame(data_syn, columns=main_lst)
    df_syn = df_syn[lst]
    df_syn, lst = removeZero(df_syn)

    # Again update real training data to match the column names with Synthetic data
    df = df[lst]

    # Find real data dataframs of meds and diag and find correlation between them
    diag, meds = getDataframes(df, lst)
    corr = np.corrcoef(diag, meds, rowvar=False)

    # Find synthetic data dataframs of meds and diag and find correlation between them
    diag_syn, meds_syn = getDataframes(df_syn, lst)
    corr_syn = np.corrcoef(diag_syn, meds_syn, rowvar=False)
    return corr, corr_syn, lst

def removeZero(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df, list(df.columns)

def generateReport(distances):
    # TODO Generate proper result
    print(distances)