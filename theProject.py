import numpy as np
import math
import random
import heapq
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
    
class Data:
    labels = np.array
    features = np.array
    featureList = []
    
    def __init__ (self):
        pass
       
    def loadTestData(self,testSet=SMALL_DATA): # SMALL_DATA or BIG_DATA
        print(f"Loading {testSet}...")
        data = np.loadtxt(testSet)
        self.labels = data[:,0].astype(int)
        self.features = data[:,1:]
        print(f"Successfully Loaded into Matrix of Size {data.shape}")
        
    def loadFeatureList(self,featureList):
        self.featureList=featureList
        print(f"Successfully Loaded Features Set!")
        
class Classifier: # Calculates distance between every point for NN
    data = Data()
    kNN = 0
    
    def train(self,data, kNN = 3):
        self.data = data
        self.kNN = kNN
        
    def test(self,testIndex) -> int:
        print("Starting Classifier Test...")
        distList = [] #Heap (Dist to testIndex, Index)
        print(f"Calculating the Distance between {testIndex} and the other datapoints...")
        for R in range(len(self.data.features)):
            if R == testIndex: continue
            heapq.heappush(distList,(self.__calcDistance__(R,testIndex),R))
        print(f"Finding the {self.kNN} Nearest Neighbors...")
        counter = Counter()
        for _ in range(self.kNN): # Get k shorests distances to testIndex
            _ ,index = heapq.heappop(distList)
            counter[self.data.labels[index]] += 1
        
        return counter.most_common(1)[0][0]
    
                
    #return np.linalg.norm(point1 - point2)
    #https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
    def __calcDistance__(self, R, testIndex) -> float:
        currentSum : float = 0
        for C in self.data.featureList:
            currentSum += (self.data.features[testIndex][C]-self.data.features[R][C])**2
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
        accuracy = (correct / len(data.features)) * 100 #divide correct by total instances to get accuracy
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
        print(Printer.searchStartBackward)
        while depth < n:
            bestChildAccuracy = (-math.inf, 0) # (eval,index of that item)
            
            for i in range(n):
                if self.featureList[i] not in currentFeatures: continue
                currentFeatures.remove(self.featureList[i]) 
                eval = self.evaluate()
                print(f"Evaluated {currentFeatures} at {eval} ")
                
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy = (eval, i)
                    
                currentFeatures.add(self.featureList[i]) 
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                print(Printer.searchQuit)
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
    searchStartForward : str = "\nStarting Forward Selection Search... "
    searchStartBackward : str = "\nStarting Backward Elimination Search... "
    searchQuit : str = "All Children Result in Lower Accuracy, Terminating Search..."
    featureAlgPrompt : str ="""Type the number of the algorithm you want to run
1) Forward Selection 
2) Backward Elimination
Choice: """
    
    @staticmethod
    def featureCountPrompt() -> int:
        prompt = "\nEnter the number of features: "
        valIn = input(prompt)
        return int(valIn)

    @staticmethod
    def printFeatureListSelected(Currentfeatures,accuracy):
        print()
        print(f"Best Feature Set Found: {Currentfeatures}")
        print(f"Accuracy: {accuracy}\n")
    
    @staticmethod    
    def printFeatureChange(featureChanged,currentFeatures,accuracy,add=True):
            if add: print("\nAdd ", end="") 
            else: print("\nRemove ",end="")
            print(f"Feature {featureChanged}" )
            print(f"New Feature Set: {currentFeatures} ~ Accuracy {accuracy}")
                 
#MAIN      
if __name__ == "__main__":
    print(Printer.mainWelcome)
    featureCount = Printer.featureCountPrompt()
    feet = FeatureSearch(featureCount)

    featureList = []
    
    algPick = input(Printer.featureAlgPrompt)
    if algPick == "1":
       featureList = feet.forwardSelection()
    else:
        featureList = feet.backwardElimination()

    dadi = Data()
    dadi.loadTestData()
    dadi.loadFeatureList(featureList)
    classi=Classifier()
    classi.train(dadi)
    vally=Validator()
    print(vally.evaluate(dadi, classi, [3, 5, 7]))
    print(f"Guess: {classi.test(11)} Actual:{dadi.labels[11]}")