from freq_doclen import Bayes_Classifier
from random import shuffle
import os
def main():
    """DO 10 times of 10 fold cross validation"""
    data = []
    ppSum, npSum, prSum, nrSum, pfSum, nfSum = 0, 0, 0, 0, 0, 0
    for fFileObj in os.walk("data/"):
        #print fFileObj
        data = fFileObj[2]
        break
    for iter in range(1):
        shuffle(data)
        pos_prec, neg_prec, pos_recall, neg_recall, pos_f_measure, neg_f_measure = cross_validation(data)
        ppSum += pos_prec
        npSum += neg_prec
        prSum += pos_recall
        nrSum += neg_recall
        pfSum += pos_f_measure
        nfSum += neg_f_measure
    #finalP, finalR, finalF = pSum / 10, rSum / 10, fSum / 10
    print "positive precision is: ", ppSum
    print "negative precision is: ", npSum
    print "positive recall is: ", prSum
    print "negative recall is: ", nrSum
    print "positive f measure is: ", pfSum
    print "negative f measure is: ", nfSum
def cross_validation(data):
    num = len(data)
    chunk = num / 10
    ppSum, npSum, prSum, nrSum, pfSum, nfSum = 0, 0, 0, 0, 0, 0
    for i in range(10):
        testing_data = data[(chunk * i) : (chunk * (i + 1))]
        training_data = data[ :(chunk * i)] + data[(chunk * (i + 1)): ]
        bc = Bayes_Classifier(eval = True)
        bc.train(training_data)
        pos_prec, neg_prec, pos_recall, neg_recall, pos_f_measure, neg_f_measure = do_evaluation(bc, testing_data)
        ppSum += pos_prec
        npSum += neg_prec
        prSum += pos_recall
        nrSum += neg_recall
        pfSum += pos_f_measure
        nfSum += neg_f_measure
    return ppSum / 10, npSum / 10, prSum / 10, nrSum / 10, pfSum / 10, nfSum / 10

def do_evaluation(bc, testing_data):
    #TODO
    typeList, resultList = [], []
    for testing_filename in testing_data:
        filePath = "data/" + testing_filename
        fileContent = bc.loadFile(filePath)
        fileType = bc.parseType(testing_filename)
        tResult = bc.classify(fileContent)
        #print "result: ", tResult
        typeList.append(fileType)
        resultList.append(tResult)
    pos_precision, neg_precision = cal_precision(typeList, resultList)
    pos_recall, neg_recall = cal_recall(typeList, resultList)
    pos_f_measure = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
    neg_f_measure = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
    return pos_precision, neg_precision, pos_recall, neg_recall, pos_f_measure, neg_f_measure

def cal_precision(typeList, resultList):
    #TODO
    resMapper = map(lambda x: 1 if x == "positive" else 0, resultList)
    typePosMapper = map(lambda x, y: 1 if x == "positive" and y == "positive" else 0, typeList, resultList)
    typeNegMapper = map(lambda x, y: 1 if x == "negative" and y == "negative" else 0, typeList, resultList)
    numPos = sum(resMapper)
    #print "numPos:", numPos
    numNeg = len(resultList) - numPos
    posPrecision = float(sum(typePosMapper)) / numPos
    negPrecision = float(sum(typeNegMapper)) / numNeg
    return posPrecision, negPrecision
    # return (posPrecision + negPrecision) * 0.5

def cal_recall(typeList, resultList):
    #TODO
    typeMapper = map(lambda x: 1 if x == "positive" else 0, typeList)
    resPosMapper = map(lambda x, y: 1 if x == "positive" and y == "positive" else 0, resultList, typeList)
    resNegMapper = map(lambda x, y: 1 if x == "negative" and y == "negative" else 0, resultList, typeList)
    numPos = sum(typeMapper)
    numNeg = len(typeList) - numPos
    posRecall = float(sum(resPosMapper)) / numPos
    negRecall = float(sum(resNegMapper)) / numNeg
    return posRecall, negRecall
    # return (posRecall + negRecall) * 0.5
main()
