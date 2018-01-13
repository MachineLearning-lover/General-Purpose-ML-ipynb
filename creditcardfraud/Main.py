from Models.SVC import trainClassifierSVC
from Models.XgBoost import trainClassifierXGBoost

def testWhichModelIsBetter():

    modelXGB,evalResults = trainClassifierXGBoost()
    print("XGBModel:")
    print(evalResults)

    # # SVC parameters for the rbf kernel
    # C = 1
    # gamma = 0.01
    #
    # modelSVC = trainClassifierSVC([C,gamma]) #it takes a long long time, sklearn limitation of 10k samples
    # print("SVCModel:")
    # print(modelSVC)

if __name__ == '__main__':
    testWhichModelIsBetter()