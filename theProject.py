
import math
import random

MAX_FEATURE_COUNT = 20

class FeatureSearch:
    featureList = []
    
    def __init__(self,featureList = range(5)):
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
            bestChildAccuracy = (-math.inf,None) # (EVAL_SCORE, FEATURE_INDEX_TO_ADD)
            
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
            
            currentFeatures.add(self.featureList[bestChildAccuracy[1]]) # Add back best branch
            print(f"\nAdding Feature {self.featureList[bestChildAccuracy[1]]}" )
            print(f"New Feature Set: {currentFeatures}: Accuracy {bestChildAccuracy[0]}\n")
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
            
        return currentFeatures
    
    def backwardSelection(self)->list:
        n = len(self.featureList)
        parentAccuracy = -math.inf
        currentFeatures=set(self.featureList)
        depth = 0
        while depth < n:
            bestChildAccuracy = (-math.inf,0) #(eval,index of that item)
            
            for i in range(n):
                if self.featureList[i] not in currentFeatures: continue
                currentFeatures.remove(self.featureList[i]) 
                eval = self.evaluate()
                print(f"Evaluated {currentFeatures} at {eval} ")
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy=(eval,i)
                currentFeatures.add(self.featureList[i]) 
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                print(f"No children with better accuracy, returning!")
                break
            
            currentFeatures.remove(self.featureList[bestChildAccuracy[1]]) #Removing the best child from feature list
            print(f"adding feature {self.featureList[bestChildAccuracy[1]]}! new set: {currentFeatures}, New accuracy {bestChildAccuracy[0]}")
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
            
        return currentFeatures

        
    
class Printer:
    mainWelcome : str = "Welcome to SZIMM011 and LADAM020's Project 2!"
    searchStartForward : str = "Starting Forward Selection Search... "
    searchQuit : str = "All Children Result in Lower Accuracy, Terminating Search..."
    
    @staticmethod
    def featureCountPrompt() -> int:
        prompt = "Enter the number of features"
        valIn = input(prompt)
        if valIn.isnumeric() and int(valIn) < MAX_FEATURE_COUNT:
            return int(valIn)
        return 5
            
        
if __name__ == "__main__":
    print(Printer.mainWelcome)
    feet = FeatureSearch()
    print(f"best set: {feet.forwardSelection()}")
    print(f"best set: {feet.backwardSelection()}")