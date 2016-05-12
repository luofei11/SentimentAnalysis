from frequency_bayes import Bayes_Classifier
from random import shuffle
def main():
    """DO 10 times of 10 fold cross validation"""
    data = []
    for fFileObj in os.walk("data/"):
        #print fFileObj
        data = fFileObj[2]
        break
    for iter in range(10):
        shuffle(data)
        prec, recall, f_measure = cross_validation(data)

def cross_validation(data):
    num = len(data)
    testing_data = data[:num / 10]
    training_data = data[num / 10 + 1:]
    bc = Bayes_Classifier(eval = True)
    bc.train(training_data)
    prec = bc.calPrecision(testing_data)
    recall = bc.calRecall(testing_data)
    f_measure = bc.fMeasure(testing_data)
    return prec, recall, f_measure
main()
