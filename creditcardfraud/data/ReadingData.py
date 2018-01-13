import pandas as pd
from sklearn.model_selection import train_test_split

def readAsPandas(pathOfFileCsv):
    """

    :param pathOfFileCsv: the path of CSV file, valid link required
    :return:
    """
    datapdFrame = pd.read_csv(pathOfFileCsv)
    return datapdFrame

def splitDataToClass(pdframe):
    # skipFirstNColumns = 0 # we skip no columns
    skipFirstNColumns = 1   # we skip the first column which is time
    return pdframe.iloc[:,skipFirstNColumns:-1],pdframe.iloc[:,-1]

def splitDataToTrainTest(pdframeX, pdframeY, dataTestSize):
    """

    :param pdframeX: example set
    :param pdframeY: target set
    :param dataTestSize: between 0 and 1
    :return:
    """
    pdframeXTrain, pdframeXTest, pdframeYTrain, pdframeYTest = train_test_split(
        pdframeX,pdframeY,random_state=0,test_size=dataTestSize
    )
    return pdframeXTrain, pdframeXTest, pdframeYTrain, pdframeYTest