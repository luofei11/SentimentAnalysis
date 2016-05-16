from frequency_bayes import Bayes_Classifier
from random import shuffle
import os
def main():
    """DO 10 times of 10 fold cross validation"""
    data = []
    pSum, rSum, fSum = 0, 0, 0
    for fFileObj in os.walk("data/"):
        #print fFileObj
        data = fFileObj[2]
        break
    for iter in range(1):
        shuffle(data)
        prec, recall, f_measure = cross_validation(data)
        pSum += prec
        rSum += recall
        fSum += f_measure
    finalP, finalR, finalF = pSum / 10, rSum / 10, fSum / 10
    print "precision is: ", finalP
    print "recall is: ", finalR
    print "f measure is: ", finalF
def cross_validation(data):
    num = len(data)
    chunk = num / 10
    pSum, rSum, fSum = 0, 0, 0
    for i in range(10):
        testing_data = data[(chunk * i) : (chunk * (i + 1))]
        training_data = data[ :(chunk * i)] + data[(chunk * (i + 1)): ]
        bc = Bayes_Classifier(eval = True)
        bc.train(training_data)
        prec, recall, f_measure = do_evaluation(bc, testing_data)
        pSum += prec
        rSum += recall
        fSum += f_measure
    return pSum / 10, rSum / 10, fSum / 10

def do_evaluation(bc, testing_data):
    #TODO
    typeList, resultList = [], []
    for testing_filename in testing_data:
        filePath = "data/" + testing_filename
        fileContent = bc.loadFile(filePath)
        fileType = bc.parseType(testing_filename)
        tResult = bc.classify(fileContent)
        typeList.append(fileType)
        resultList.append(tResult)
    precision = cal_precision(typeList, resultList)
    recall = cal_recall(typeList, resultList)
    f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure

def cal_precision(typeList, resultList):
    #TODO
    resMapper = map(lambda x: 1 if x == "positive" else 0, resultList)
    typePosMapper = map(lambda x, y: 1 if x == "positive" and y == "positive" else 0, typeList, resultList)
    typeNegMapper = map(lambda x, y: 1 if x == "negative" and y == "negative" else 0, typeList, resultList)
    numPos = sum(resMapper)
    numNeg = len(resultList) - numPos
    posPrecision = float(sum(typePosMapper)) / numPos
    negPrecision = float(sum(typeNegMapper)) / numNeg
    return (posPrecision + negPrecision) * 0.5

def cal_recall(typeList, resultList):
    #TODO
    typeMapper = map(lambda x: 1 if x == "positive" else 0, typeList)
    resPosMapper = map(lambda x, y: 1 if x == "positive" and y == "positive" else 0, resultList, typeList)
    resNegMapper = map(lambda x, y: 1 if x == "negative" and y == "negative" else 0, resultList, typeList)
    numPos = sum(typeMapper)
    numNeg = len(typeList) - numPos
    posRecall = float(sum(resPosMapper)) / numPos
    negRecall = float(sum(resNegMapper)) / numNeg
    return (posRecall + negRecall) * 0.5
main()
