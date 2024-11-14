
import math
import random

class FeatureSearch:
    featureList = []
    
    def __init__(self,featureList = range(5)):
        self.featureList = featureList
    
    def evaluate(self):
        return random.randint(1,100)

    def forwardSelection(self)->list:
        n = len(self.featureList)
        parentAccuracy = -math.inf
        currentFeatures=set()
        depth = 0
        while depth < n:
            bestChildAccuracy = (-math.inf,0)
            
            for i in range(n):
                if self.featureList[i] in currentFeatures: continue
                currentFeatures.add(self.featureList[i]) 
                eval = self.evaluate()
                print(f"Evaluated {currentFeatures} at {eval} ")
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy=(eval,i)
                    
                currentFeatures.remove(self.featureList[i]) #backtrack
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                print(f"No children with better accuracy, returning!")
                break
            
            currentFeatures.add(self.featureList[bestChildAccuracy[1]]) #Add back best branch
            print(f"adding feature {self.featureList[bestChildAccuracy[1]]}! new set: {currentFeatures}, New accuracy {bestChildAccuracy[0]}")
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
            
        return currentFeatures
    
class Printer:
    pass
    
if __name__ == "__main__":
    feet = FeatureSearch()
    print(f"best set: {feet.forwardSelection()}")