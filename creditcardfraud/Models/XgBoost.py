from getDataInMLFormat import setupData
import xgboost as xgb

def defParamForXgBoost(parameters=[]):
    if (len(parameters) > 0):
        params = parameters
    else:
        # default parameters
        params = {'updater':'grow_gpu',
                  'silent':0,
                  'eta':0.01,
                  'eval_metric':'auc'
                  }
    return params



def trainClassifierXGBoost(parameters=[]):
    params = defParamForXgBoost(parameters)
    evalResults = {}
    creditCardXTrain, creditCardXTest, creditCardYTrain, creditCardYTest = setupData(testSize=0.2)
    dTrainData = xgb.DMatrix(creditCardXTrain,creditCardYTrain)
    dTestData = xgb.DMatrix(creditCardXTest,creditCardYTest)
    trainedModel = xgb.train(dtrain=dTrainData,num_boost_round=10,evals=[(dTrainData,'trainData'),
                            (dTestData,'testData')], verbose_eval=True,params=params,
                             evals_result=evalResults)
    return trainedModel,evalResults

