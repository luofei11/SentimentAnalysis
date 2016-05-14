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
    for iter in range(10):
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
    testing_data = data[:num / 10]
    training_data = data[num / 10 + 1:]
    bc = Bayes_Classifier(eval = True)
    bc.train(training_data)
    prec, recall, f_measure = do_evaluation(bc, testing_data)
    return prec, recall, f_measure

def do_evaluation(bc, testing_data):
    #TODO
    actualPos, actualNeg, classifiedPos, classifiedNeg = 0, 0, 0, 0
    numPos, numNeg, numResultPos, numResultNeg = 0, 0, 0, 0
    for testing_filename in testing_data:
        filePath = "data/" + testing_filename
        fileContent = bc.loadFile(filePath)
        fileType = bc.parseType(testing_filename)
        tResult = bc.classify(fileContent)
        if tResult == "positive":
            classifiedPos += 1
            if fileType == "positive":
                actualPos += 1
        elif tResult == "negative":
            classifiedNeg += 1
            if fileType == "negative":
                actualNeg += 1
        if fileType == "positive":
            numPos += 1
            if tResult == "positive":
                numResultPos += 1
        elif fileType == "negative":
            numNeg += 1
            if tResult == "negative":
                numResultNeg += 1
    precision = (float(actualPos) / float(classifiedPos) + float(actualNeg) / float(classifiedNeg)) * 0.5
    recall = (float(numResultPos) / numPos + float(numResultNeg) / numNeg) * 0.5
    f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure
main()
