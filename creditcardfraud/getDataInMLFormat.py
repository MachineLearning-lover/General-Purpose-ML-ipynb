import data.ReadingData as ReadingData

def setupData(testSize):
    path = "D:\python projects\CreditCardFraud\creditcardfraud\creditcard.csv"
    creditCard = ReadingData.readAsPandas(path)
    creditCardX, creditCardY = ReadingData.splitDataToClass(creditCard)
    creditCardXTrain,creditCardXTest, creditCardYTrain, creditCardYTest = ReadingData.splitDataToTrainTest(
        creditCardX,creditCardY, dataTestSize=testSize
    )
    return creditCardXTrain,creditCardXTest, creditCardYTrain, creditCardYTest
