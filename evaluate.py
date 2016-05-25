# Name: evaluate.py
# Date: 5/23/2016
# Description:
# Evaluates the performance(precision, recall and F1 measure) of a bayes classifier
# Author(s) names AND netid's:
#        -Fei Luo(fla414)
#        -Ye  Xue(yxe836)
# Group work statement: All group members were present and contributing during all work on this project.
#
from freq_bigram import Bayes_Classifier
from random import shuffle
import os

def parseType(name):
   stars = name.split("-")[1]
   return "positive" if stars == "5" else "negative"
def main():
    """Performs 10 trials of 10 fold cross validation"""
    data = []
    ppSum, npSum, prSum, nrSum, pfSum, nfSum = 0, 0, 0, 0, 0, 0
    for fFileObj in os.walk("movies_reviews/"):
        data = fFileObj[2]
        break
    pos_data, neg_data = [], []
    for filename in data:
        if parseType(filename) == "positive":
            pos_data.append(filename)
        else:
            neg_data.append(filename)
    for iter in range(10):
        print iter, "th iteration"
        shuffle(pos_data)
        shuffle(neg_data)
        pos_prec, neg_prec, pos_recall, neg_recall, pos_f_measure, neg_f_measure = cross_validation(pos_data, neg_data)
        ppSum += pos_prec
        npSum += neg_prec
        prSum += pos_recall
        nrSum += neg_recall
        pfSum += pos_f_measure
        nfSum += neg_f_measure
    print "positive precision is: ", ppSum / 10
    print "negative precision is: ", npSum / 10
    print "positive recall is: ", prSum / 10
    print "negative recall is: ", nrSum / 10
    print "positive f measure is: ", pfSum / 10
    print "negative f measure is: ", nfSum / 10
def cross_validation(pos_data, neg_data):
    pos_num = len(pos_data)
    neg_num = len(neg_data)
    pos_chunk = pos_num / 10
    neg_chunk = neg_num / 10
    ppSum, npSum, prSum, nrSum, pfSum, nfSum = 0, 0, 0, 0, 0, 0
    for i in range(10):
        #evenly distributes the positive and negative documents in the testing data
        testing_data = pos_data[(pos_chunk * i) : (pos_chunk * (i + 1))] + neg_data[(neg_chunk * i) : (neg_chunk * (i + 1))]
        training_data = pos_data[ :(pos_chunk * i)] + pos_data[(pos_chunk * (i + 1)): ] + neg_data[ :(neg_chunk * i)] + neg_data[(neg_chunk * (i + 1)): ]
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
    """Given a classifier and testing data, return the performance measurements"""
    typeList, resultList = [], []
    for testing_filename in testing_data:
        filePath = "movies_reviews/" + testing_filename
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
    """Given the file type list and classification result list, return the precision"""
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
    """Given the file type list and classification result list, return the recall"""
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
