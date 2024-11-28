import numpy as np
import math
import random
import heapq
import logging
from pathlib import Path
from collections import Counter


#  Cat to prevent bugs
#        _                        
#        \`*-.                    
#         )  _`-.                 
#        .  : `. .                
#        : _   '  \               
#        ; *` _.   `*-._          
#        `-.-'          `-.       
#          ;       `       `.     
#          :.       .        \    
#          . \  .   :   .-'   .   
#          '  `+.;  ;  '      :   
#          :  '  |    ;       ;-. 
#          ; '   : :`-:     _.`* ;
# [bug] .*' /  .*' ; .*`- +'  `*' 
#       `*-*   `*-*  `*-*'


SMALL_DATA = Path("small-test-dataset.txt")
BIG_DATA = Path("large-test-dataset.txt")
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    
class Data:
    labels = np.array
    features = np.array
    featureList = []
    
    def __init__ (self):
        pass
       
    def loadTestData(self,testSet=SMALL_DATA): # SMALL_DATA or BIG_DATA
        logging.info(f"Loading {testSet}...")
        data = np.loadtxt(testSet)
        self.labels = data[:,0].astype(int)
        self.features = data[:,1:]
        logging.info(f"Successfully Loaded into Matrix of Size {data.shape}")
        
    def loadFeatureList(self,featureList):
        self.featureList=featureList
        logging.info(f"Successfully Loaded Features Set!")
        
class Classifier: # Calculates distance between every point for NN
    data = Data()
    kNN = 0
    
    def train(self,data, kNN = 1):
        self.data = data
        self.kNN = kNN
        
    def test(self,testIndex) -> int:
        logging.info("Starting Classifier Test...")
        distList = [] #Heap (Dist to testIndex, Index)
        logging.info(f"Calculating the Distance between index {testIndex} and the other datapoints...")
        for R in range(len(self.data.features)):
            if R == testIndex: continue
            heapq.heappush(distList,(self.__calcDistance__(R,testIndex),R))
        logging.info(f"Finding the {self.kNN} Nearest Neighbors...")
        counter = Counter()
        for _ in range(self.kNN): # Get k shorests distances to testIndex
            _ ,index = heapq.heappop(distList)
            counter[self.data.labels[index]] += 1
        
        return counter.most_common(1)[0][0]
    
                
    #Returns euclidian distance between testindex and row R
    def __calcDistance__(self, R, testIndex) -> float:
        currentSum : float = 0
        for C in self.data.featureList:
            currentSum += (self.data.features[testIndex,C]-self.data.features[R,C])**2
        return math.sqrt(currentSum)

class Validator: #Computes classifier's accuracy
    def __init__(self):
        pass
    
    def evaluate(self, data:Data, classifier:Classifier, featureList = None): 
        correct = 0
        accuracy = 0
        if featureList: 
            data.loadFeatureList(featureList) #will load feature list if given one

        for i in range(len(data.features)): #loop through instance id
            if data.labels[i] == classifier.test(i): #check if it got correct for each row
                correct+= 1
        accuracy = (correct / len(data.features)) #divide correct by total instances to get accuracy
        accuracy = round(accuracy, 4)
        return accuracy

class FeatureSearch:
    featureList = []
    
    def __init__(self,featureCount):
        self.featureList = list(range(featureCount))
    
    def evaluate(self): 
        return random.randint(1,100) #stubbed

    def forwardSelection(self) -> list:
        n = len(self.featureList)
        parentAccuracy = -math.inf
        currentFeatures = set()
        depth = 0
        print(Printer.searchStartForward)
        while depth < n:
            bestChildAccuracy = (-math.inf, None) # (EVAL_SCORE, FEATURE_INDEX_TO_ADD)
            
            for i in range(n):
                if self.featureList[i] in currentFeatures: continue
                currentFeatures.add(self.featureList[i]) 
                eval = self.evaluate()
                print(f"{currentFeatures} Evaluated at {eval}")
                
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy = (eval,i)
                    
                currentFeatures.remove(self.featureList[i]) #backtrack
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                print(Printer.searchQuit)
                break
            
            featureChanged = self.featureList[bestChildAccuracy[1]]
            currentFeatures.add(featureChanged) # Add back best branch
            Printer.printFeatureChange(featureChanged,currentFeatures,bestChildAccuracy[0],True)
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
            
        Printer.printFeatureListSelected(currentFeatures,parentAccuracy)
        return currentFeatures
    
    def backwardElimination(self)->list:
        n = len(self.featureList)
        parentAccuracy = -math.inf
        currentFeatures = set(self.featureList)
        depth = 1
        logging.info(Printer.searchStartBackward)
        while depth < n:
            bestChildAccuracy = (-math.inf, 0) # (eval,index of that item)
            
            for i in range(n):
                if self.featureList[i] not in currentFeatures: continue
                currentFeatures.remove(self.featureList[i]) 
                eval = self.evaluate()
                logging.info(f"Evaluated {currentFeatures} at {eval} ")
                
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy = (eval, i)
                    
                currentFeatures.add(self.featureList[i]) 
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                logging.info(Printer.searchQuit)
                break
            
            featureChanged = self.featureList[bestChildAccuracy[1]]
            currentFeatures.remove(featureChanged) #Removing the best child from feature list
            Printer.printFeatureChange(featureChanged,currentFeatures,bestChildAccuracy[0],False)       
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
        Printer.printFeatureListSelected(currentFeatures,parentAccuracy)
        return currentFeatures

        
    
class Printer:
    mainWelcome : str = "\nWelcome to SZIMM011 and LADAM020's Project 2!"
    searchStartForward : str = "Starting Forward Selection Search... "
    searchStartBackward : str = "Starting Backward Elimination Search... "
    searchQuit : str = "All Children Result in Lower Accuracy, Terminating Search..."
    feetAlgPrompt : str ="""Type the number of the algorithm you want to run
1) Forward Selection 
2) Backward Elimination
Choice: """
    datAlgPrompt : str ="""Type the number cooresponding to the data you want
1) Big data 
2) Small data
Choice: """
    
    @staticmethod
    def featureAlgPrompt(feet: FeatureSearch) -> list:
        inny = input(Printer.feetAlgPrompt)
        if inny == 1:
            return feet.forwardSelection()
        else:
            return feet.backwardElimination()

    @staticmethod
    def dataAlgPrompt() -> Path:
        datPick = input(Printer.datAlgPrompt)
        if datPick == 1:
            return BIG_DATA
        else:
            return SMALL_DATA

    @staticmethod
    def featureCountPrompt() -> int:
        prompt = "\nEnter the number of features: "
        valIn = input(prompt)
        return int(valIn)

    @staticmethod
    def printFeatureListSelected(Currentfeatures,accuracy):
        print(f"Best Feature Set Found: {Currentfeatures}")
        print(f"Accuracy: {accuracy}\n")
    
    @staticmethod    
    def printFeatureChange(featureChanged,currentFeatures,accuracy,add=True):
        if add: logging.info(f"Adding Feature {featureChanged}" )
        else: logging.info(f"Removing Feature {featureChanged}" )
        logging.info(f"New Feature Set: {currentFeatures}  Accuracy: {accuracy}")
                 
#MAIN      
if __name__ == "__main__":
    dadi = Data()
    classi=Classifier()
    vally=Validator()
    featureList = []

    print(Printer.mainWelcome)
    featureCount = Printer.featureCountPrompt()
    dadi.loadTestData(Printer.dataAlgPrompt())

    feet = FeatureSearch(featureCount)
    algPick = Printer.featureAlgPrompt(feet)

    dadi.loadFeatureList(featureList)
    classi.train(dadi)
    print(vally.evaluate(dadi, classi, [2, 4, 6]))