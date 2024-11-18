import numpy as np
import math
import random
from pathlib import Path


DEFAULT_FEATURE_COUNT = 5
MAX_FEATURE_COUNT = 20
SMALL_DATA = Path("small-test-dataset.txt")
BIG_DATA = Path("large-test-dataset.txt")

class Data:
    labels = np.array
    features = np.array
    
    def __init__ (self):
        pass
       
    def loadTestData(self,testSet=SMALL_DATA): # SMALL_DATA or BIG_DATA
        print(f"Loading {testSet}...")
        data = np.loadtxt(testSet)
        self.labels = data[:,0].astype(int)
        self.features = data[:,1:]
        print(f"Successfully Loaded into Matrix of Size {data.shape}")
        

class FeatureSearch:
    featureList = []
    
    def __init__(self,featureList = range(DEFAULT_FEATURE_COUNT)):
        self.featureList = featureList
    
    def evaluate(self): 
        return random.randint(1,100) #stubbed

    def forwardSelection(self)->list:
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
        depth = 0
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
    searchStartForward : str = "Starting Forward Selection Search... "
    searchStartBackward : str = "Starting Backward Elimination Search... "
    searchQuit : str = "All Children Result in Lower Accuracy, Terminating Search..."
    
    @staticmethod
    def featureCountPrompt() -> int:
        prompt = "Enter the number of features"
        valIn = input(prompt)
        if valIn.isnumeric() and int(valIn) < MAX_FEATURE_COUNT:
            return int(valIn)
        return DEFAULT_FEATURE_COUNT
    
    @staticmethod
    def printFeatureListSelected(Currentfeatures,accuracy):
        print()
        print(f"Best Feature Set Found: {Currentfeatures}")
        print(f"Acurracy: {accuracy}\n")
    
    @staticmethod    
    def printFeatureChange(featureChanged,currentFeatures,accuracy,add=True):
            if add: print("\nAdd", end="") 
            else: print("\nRemove",end="")
            print(f"Feature {featureChanged}" )
            print(f"New Feature Set: {currentFeatures} ~ Accuracy {accuracy}")
                 
#MAIN      
if __name__ == "__main__":
    print(Printer.mainWelcome)
    feet = FeatureSearch()
    feet.forwardSelection()
    feet.backwardElimination()
    #hey = Data()
    #hey.loadTestData()
    