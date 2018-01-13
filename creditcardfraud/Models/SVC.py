from getDataInMLFormat import setupData
from sklearn import svm
from sklearn.model_selection import cross_val_score

def trainClassifierSVC(params):
    """
    Using Sklearn; Params are given for hypertuning
    :param params: [C_value, Gamma_value_rbf]
    :return: ['test_score', 'fit_time', 'score_time']
    """
    CVal = params[0]
    GammaVal = params[1]
    creditCardXTrain, creditCardXTest, creditCardYTrain, creditCardYTest = setupData(testSize=0.2)
    classifier = svm.SVC(C=CVal,kernel='rbf',gamma=GammaVal,verbose=True)
    classifier.fit(creditCardXTrain,creditCardYTrain)
    cvScores = cross_val_score(classifier,n_jobs=-1,X=creditCardXTrain,y=creditCardYTrain,cv=1,verbose=True)
    return cvScores


