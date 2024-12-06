import numpy as np
import math
import random
import heapq
import logging
import time
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
    featureList = np.array
    
    def __init__ (self):
        pass
       
    def loadTestData(self, testSet=SMALL_DATA):
        logging.info(f"Loading {testSet}...")
        data = np.loadtxt(testSet)

        # Extract labels (first column)
        self.labels = data[:, 0].astype(int)

        # Extract features (all columns except first)
        features_to_normalize = data[:, 1:]

        # Calculate min and max for each feature column
        min_vals = np.min(features_to_normalize, axis=0)
        max_vals = np.max(features_to_normalize, axis=0)

        # Min-max normalization - store directly in features
        self.features = (features_to_normalize - min_vals) / (max_vals - min_vals)

        logging.info(f"Data Successfully Loaded into Matrix of Size {data.shape}")
        logging.info("Features have been normalized (excluding labels)")


    def loadFeatureList(self, featureList):
        self.featureList = np.array(featureList)
        logging.debug(f"Feature list loaded: {featureList}")

        
class Classifier: # Calculates distance between every point for NN
    data = Data()
    kNN = 1
    
    def train(self,data, kNN = 3):
        self.data = data
        logging.debug(f"Classifier Training Data Loaded!")
        self.kNN = kNN
        logging.info(f"Set Nearest Neighbor K={self.kNN}")
        
    def test(self, testIndex: int) -> int:
        # left shift indexes (1 -> 0)
        feature_indices = [i-1 for i in self.data.featureList]  
        # isolate test row
        test_point = self.data.features[testIndex, feature_indices]
        # remove our test index from features and label array
        remaining_features = np.delete(self.data.features[:, feature_indices], testIndex, axis=0)
        remaining_labels = np.delete(self.data.labels, testIndex)

        #https://jaykmody.com/blog/distance-matrices-with-numpy/
        # array of euclidan dists only wrt selected features
        # for each row, subtract test point features from current features -> square it -> sum it -> sqrt it
        distances = np.sqrt(np.sum((remaining_features - test_point)**2, axis=1))

        # create a new array of indexes corrisponding to distance list in sorted order
        # slice the kNN smallest distance indexs 
        nearest_indices = np.argsort(distances)[:self.kNN]

        # Count votes
        counter = [0, 0]  # [count for label 1, count for label 2]
        for idx in nearest_indices:
            label = remaining_labels[idx]
            counter[label - 1] += 1

        return 1 if counter[0] > counter[1] else 2

class Validator: #Computes classifier's accuracy
    def __init__(self):
        pass
    
    def evaluate(self, data: Data, classifier: Classifier, featureList=None) -> float:
        correct = 0
        total = data.features.shape[0]

        if featureList:
            data.loadFeatureList(featureList)

        timeStart = time.perf_counter_ns()
        # loop through every row -> guess answer when leaving out that row 
        for R in range(total):
            predicted = classifier.test(R)
            actual = data.labels[R]
            if predicted == actual:
                correct += 1
            logging.debug(f"Instance {R}: Predicted={predicted}, Actual={actual}")

        timeEnd = time.perf_counter_ns()
        accuracy = correct / total
        logging.info(f"Features: {featureList} Accuracy: {accuracy:.4f} Time: {round((timeEnd - timeStart)*10**(-9), 8)}s")
        return accuracy

class FeatureSearch:
    featureList = []
    vally : Validator
    dadi: Data
    classi : Classifier
    
    
    def __init__(self, vally : Validator, data : Data, classi : Classifier):
        self.featureList = list(range(1,data.features.shape[1]))
        self.vally = vally
        self.dadi = data
        self.classi = classi
    
    def evaluate(self): 
        return random.randint(1,100) #stubbed

    def forwardSelection(self) -> list:
        n = len(self.featureList)
        parentAccuracy = -math.inf
        currentFeatures = set()
        depth = 0
        print(Printer.searchStartForward)

        timeStart = time.perf_counter_ns()
        
        while depth < n:
            bestChildAccuracy = (-math.inf, None) # (EVAL_SCORE, FEATURE_INDEX_TO_ADD)
            
            for i in range(n):
                if self.featureList[i] in currentFeatures: continue
                currentFeatures.add(self.featureList[i]) 
                eval = self.vally.evaluate(self.dadi, self.classi, list(currentFeatures))
                logging.debug(f"{currentFeatures} Evaluated at {eval}")
                
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy = (eval,i)
                    
                currentFeatures.remove(self.featureList[i]) #backtrack
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                timeEnd = time.perf_counter_ns()
                logging.info(f"Time: {round((timeEnd - timeStart)*10**(-9), 8)}m")
                print(Printer.searchQuit)
                break
            
            featureChanged = self.featureList[bestChildAccuracy[1]]
            currentFeatures.add(featureChanged) # Add back best branch
            Printer.printFeatureChange(featureChanged,currentFeatures,bestChildAccuracy[0],True)
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
            
        Printer.printFeatureListSelected(currentFeatures,parentAccuracy)
        return list(currentFeatures)
    
    def backwardElimination(self)->list:
        n = len(self.featureList)
        parentAccuracy = -math.inf
        currentFeatures = set(self.featureList)
        depth = 1
        logging.info(Printer.searchStartBackward)

        timeStart = time.perf_counter_ns()
        while depth < n:
            bestChildAccuracy = (-math.inf, 0) # (eval,index of that item)
            
            for i in range(n):
                if self.featureList[i] not in currentFeatures: continue
                currentFeatures.remove(self.featureList[i]) 
                eval = self.vally.evaluate(self.dadi,self.classi,list(currentFeatures))
                logging.debug(f"Evaluated {currentFeatures} at {eval} ")
                
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy = (eval, i)
                    
                currentFeatures.add(self.featureList[i]) 
                
            if bestChildAccuracy[0] < parentAccuracy: # No better options dont add -> exit
                timeEnd = time.perf_counter_ns()
                logging.info(f"Time: {round((timeEnd - timeStart)*10**(-9), 8)}s")
                logging.info(Printer.searchQuit)
                break
            
            featureChanged = self.featureList[bestChildAccuracy[1]]
            currentFeatures.remove(featureChanged) #Removing the best child from feature list
            Printer.printFeatureChange(featureChanged,currentFeatures,bestChildAccuracy[0],False)       
            parentAccuracy = bestChildAccuracy[0]
            depth += 1
        Printer.printFeatureListSelected(currentFeatures,parentAccuracy)
        return list(currentFeatures)

        
    
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
        if inny == '1':
            return feet.forwardSelection()
        else:
            return feet.backwardElimination()

    @staticmethod
    def dataAlgPrompt() -> Path:
        datPick = input(Printer.datAlgPrompt)
        if datPick == '1':
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
        if add: logging.info(f"Adding Best Feature: {featureChanged}" )
        else: logging.info(f"Removing Worst Feature: {featureChanged}" )
        logging.info(f"New Feature Set: {currentFeatures}  Accuracy: {accuracy}")
                 
#MAIN      
if __name__ == "__main__":
    dadi = Data()
    classi=Classifier()
    vally=Validator()

    print(Printer.mainWelcome)
    dadi.loadTestData(Printer.dataAlgPrompt())
    classi.train(dadi)
    feet = FeatureSearch(vally,dadi,classi)
    algPick = Printer.featureAlgPrompt(feet)
    