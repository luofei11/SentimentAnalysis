# Name:
# Date:
# Description:
#
#

import math, os, pickle, re

class Bayes_Classifier:

   def __init__(self, eval = False):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
      cache of a trained classifier has been stored, it loads this cache.  Otherwise,
      the system will proceed through training.  After running this method, the classifier
      is ready to classify input text."""
      if eval:
          #for evaluation purpose
          self.pos_dic = dict()
          self.neg_dic = dict()
      else:
          try:
              self.pos_dic = self.load("presence_pos_dic")
              self.neg_dic = self.load("presence_neg_dic")
              print "loading cached data: Done"
          except IOError:
              print "no existing trained data"
              self.train()


   def train(self, training_data = None):
      """Trains the Naive Bayes Sentiment Classifier."""
      IFileList = []
      pos_dic, neg_dic = dict(), dict()
      if not training_data:
          for fFileObj in os.walk("data/"):
              IFileList = fFileObj[2]
              break
      else:
          IFileList = training_data
      #print len(IFileList)
      for filename in IFileList:
          visited = set()
          fileType = self.parseType(filename)
          filePath = "data/" + filename
          fileContent = self.loadFile(filePath)
          tokens = self.tokenize(fileContent)
          if fileType == "positive":
              for token in tokens:
                  if token not in visited:
                      pos_dic[token] = pos_dic.get(token, 0) + 1
                      visited.add(token)
          else:
              for token in tokens:
                  if token not in visited:
                      neg_dic[token] = neg_dic.get(token, 0) + 1
                      visited.add(token)
      self.pos_dic = pos_dic
      self.neg_dic = neg_dic
      if not training_data:
          self.save(pos_dic, "presence_pos_dic")
          self.save(neg_dic, "presence_neg_dic")
      print "finish training naive bayes with presence."


   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      tokens = self.tokenize(sText)
      posProbability, negProbability = 0, 0
      posNum, negNum = float(sum(self.pos_dic.values())), float(sum(self.neg_dic.values()))

      for token in tokens:
          posProbability += math.log(float((self.pos_dic.get(token,0) + 1)) / posNum)
          negProbability += math.log(float((self.neg_dic.get(token,0) + 1)) / negNum)
      if posProbability > negProbability:
          return "positive"
      else:
          return "negative"

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt

   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()

   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj
   def parseType(self, name):
      stars = name.split("-")[1]
      return "positive" if stars == "5" else "negative"

   def tokenize(self, sText):
      """Given a string of text sText, returns a list of the individual tokens that
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))

      if sToken != "":
         lTokens.append(sToken)

      return lTokens
