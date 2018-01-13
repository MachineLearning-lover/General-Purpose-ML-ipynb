import unittest
from data.ReadingData import splitDataToClass
from data.ReadingData import readAsPandas
from data.ReadingData import splitDataToTrainTest

class TestReadingData(unittest.TestCase):

    def test_splitDataToClass_ok(self):
        self.path = r"D:\python projects\CreditCardFraud\creditcardfraud\creditcard.csv"
        date = readAsPandas(self.path)
        X, Y = splitDataToClass(date)
        print(type(Y))
        self.assertEquals(len(date.columns.tolist())-1,len(X.columns.tolist()))


    def test_splitDataToTrainTest_length_validity(self):
        self.path = r"D:\python projects\CreditCardFraud\creditcardfraud\creditcard.csv"
        date = readAsPandas(self.path)
        X, Y = splitDataToClass(date)
        pdframeXTrain, pdframeXTest, pdframeYTrain, pdframeYTest = splitDataToTrainTest(X, Y, dataTestSize=0.5)
        self.assertEquals(len(pdframeXTest),len(pdframeYTest))

if __name__=='main':
    unittest.main()