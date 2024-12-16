import numpy as np
import math
import random
import logging
import time
import argparse
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


SMALL_DATA = Path("./datasets/small-test-dataset.txt")
BIG_DATA = Path("./datasets/large-test-dataset.txt")
DEMO = Path("./datasets/demo.txt")
TITANIC = Path("./datasets/titanic.txt")


# Titanic Feature Key
#Feature 1: Passenger Ticket Class
#Feature 2: Passenger Sex
#Feature 3: Passenger Age in Years
#Feature 4: Number of Passenger's Siblings Aboard the Titanic
#Feature 5: Number of Passenger's Parents/Children Aboard the Titanic
#Feature 6: Passenger Fare (Ticket Cost)
    
class Data:
    labels = np.array
    features = np.array
    featureList = np.array
    normalizationMethod = 0
    dataset : Path
    
    def __init__ (self, normalizeMethod : int = 0):
        self.normalizationMethod = normalizeMethod
       
    def loadTestData(self, testSet=SMALL_DATA):
        self.dataset = testSet
        logging.info(f"Loading {testSet}...")
        data = np.loadtxt(testSet)
        # Extract labels (first column)
        self.labels = data[:, 0].astype(int)
        # Extract features (all columns except first)
        features_to_normalize = data[:, 1:]
    
        #Normalization Method
        match self.normalizationMethod:
            case 0: #Min-Max
                logging.info(f"Normalization Method: Min-Max")
                min_vals = np.min(features_to_normalize, axis=0)
                max_vals = np.max(features_to_normalize, axis=0)
                # Min-max normalization 
                features_to_normalize = (features_to_normalize - min_vals) / (max_vals - min_vals)
            case 1: #Standard Normal
                logging.info(f"Normalization Method: Standard-Normal")
                means = np.mean(features_to_normalize, axis=0)
                stds = np.std(features_to_normalize, axis=0)
                # Normalize features using z-score normalization
                features_to_normalize = (features_to_normalize - means) / stds
            case 2:
                logging.info(f"Normalization Method: Numpy Default")
                np.linalg.norm(features_to_normalize)
            case 3: #NONE
                logging.info(f"Normalization Method: NONE")

        self.features = features_to_normalize

        
        logging.info("Features normalized")
        logging.info(f"Data Successfully Loaded into Matrix of Size {data.shape}")



    def loadFeatureList(self, featureList):
        self.featureList = np.array(featureList)
        logging.debug(f"Feature list loaded: {featureList}")

        
class Classifier: # Calculates distance between every point for NN
    data = Data()
    kNN = 1
    
    def train(self,data, kNN = 1):
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

        # new array of indexes corrisponding to distance list in sorted order
        # slice the kNN smallest distance indexs 
        nearest_indices = np.argsort(distances)[:self.kNN]

        # tally the winner
        voters = Counter()
        for idx in nearest_indices:
            label = remaining_labels[idx]
            voters[label] += 1
        result = voters.most_common(1)[0][0]
        return result

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
        logging.info(f"Features: {featureList} Accuracy: {round(accuracy,4)} Time: {round((timeEnd - timeStart)*10**(-9), 8)}s")
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
    
    def backwardElimination(self) -> list:
        n = len(self.featureList)
        global_best_accuracy = -math.inf
        global_best_features = set()
        current_features = set(self.featureList)
        depth = 1
        logging.info(Printer.searchStartBackward)

        timeStart = time.perf_counter_ns()

        while depth < n:
            bestChildAccuracy = (-math.inf, 0)  # (eval, index of that item)

            # Try removing each feature
            for i in range(n):
                if self.featureList[i] not in current_features:
                    continue
                current_features.remove(self.featureList[i])
                eval = self.vally.evaluate(self.dadi, self.classi, list(current_features))
                logging.debug(f"Evaluated {current_features} at {eval}")

                # Update best child if current evaluation is better
                if eval > bestChildAccuracy[0]:
                    bestChildAccuracy = (eval, i)

                # Update global best if we found a better solution
                if eval > global_best_accuracy:
                    global_best_accuracy = eval
                    global_best_features = set(current_features)

                current_features.add(self.featureList[i])

            # no improvement, but we can try to search deeper
            if bestChildAccuracy[0] < global_best_accuracy and depth < n-1:
                # still remove the worst feature, pray we escape local extrema
                available_features = list(current_features)
                if available_features:  # Make sure we have features to remove
                    feature_to_remove =   random.choice(available_features)
                    current_features.remove(feature_to_remove)
                    logging.info(f"Attempting to escape local maximum by removing feature {feature_to_remove}")
            else:
                # Remove the feature that gave the worst result
                feature_changed = self.featureList[bestChildAccuracy[1]]
                current_features.remove(feature_changed)
                Printer.printFeatureChange(feature_changed, current_features, bestChildAccuracy[0], False)

            depth += 1

        timeEnd = time.perf_counter_ns()
        logging.info(f"Time: {round((timeEnd - timeStart)*10**(-9), 8)}s")

        # Return to the globally best feature set found
        Printer.printFeatureListSelected(global_best_features, global_best_accuracy)
        return list(global_best_features)
    
    def simulatedAnnealing(self) -> list:
        n = len(self.featureList)
        current_features = set(range(1, n+1))  #all features
        current_accuracy = self.vally.evaluate(self.dadi, self.classi, list(current_features))
        best_features = set(current_features)
        best_accuracy = current_accuracy
        visited = set()
        
        # Settings
        initial_temp = 1.0
        final_temp = 0.01
        alpha = 0.99  # cooling rate 
        iterations_per_temp = 20

        current_temp = initial_temp
        logging.info("Starting Simulated Annealing Search...")
        timeStart = time.perf_counter_ns()

        while current_temp > final_temp:
            for _ in range(iterations_per_temp):
                neighbor_features = set(current_features)

                if random.random() < 0.5 and len(neighbor_features) > 1:  # Remove feature
                    feature_to_remove = random.choice(list(neighbor_features))
                    neighbor_features.remove(feature_to_remove)
                else:  # Add feature
                    available_features = set(range(1, n+1)) - neighbor_features
                    if available_features:
                        feature_to_add = random.choice(list(available_features))
                        neighbor_features.add(feature_to_add)
                
                #Don't revisit the same set
                if frozenset(neighbor_features) in visited: continue
                
                # Evaluate neighbor solution
                neighbor_accuracy = self.vally.evaluate(self.dadi, self.classi, list(neighbor_features))

                #acceptance probability
                delta = neighbor_accuracy - current_accuracy
                acceptance_probability = min(1.0, math.exp(delta / current_temp))

                
                # accept or reject random selection
                if delta > 0 or random.random() < acceptance_probability:
                    current_features = neighbor_features
                    current_accuracy = neighbor_accuracy
                    visited.add(frozenset(current_features))

                    # Maybe update best solution!
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_features = set(current_features)
                        logging.info(f"New best solution found: {best_features} with accuracy {best_accuracy}")

                logging.debug(f"Temperature: {current_temp:.4f}, Current Accuracy: {current_accuracy:.4f}")

            # cooling
            current_temp *= alpha

        timeEnd = time.perf_counter_ns()
        logging.info(f"Time: {round((timeEnd - timeStart)*10**(-9), 8)}s")

        Printer.printFeatureListSelected(best_features, best_accuracy)
        return list(best_features)
    
class Printer:
    mainWelcome : str = "\nWelcome to SZIMM011 and LADAM020's Project 2!"
    searchStartForward : str = "Starting Forward Selection Search... "
    searchStartBackward : str = "Starting Backward Elimination Search... "
    searchQuit : str = "All Children Result in Lower Accuracy, Terminating Search..."
    feetAlgPrompt : str ="""Type the number of the algorithm you want to run
1) Forward Selection 
2) Backward Elimination
3) Simulated Annealing
Choice: """
    datAlgPrompt : str ="""Type the number cooresponding to the data you want
1) Big data 
2) Small data
3) Titanic data
Choice: """
    
    @staticmethod
    def featureAlgPrompt(feet: FeatureSearch) -> list:
        inny = input(Printer.feetAlgPrompt)
        if inny == '1':
            return feet.forwardSelection()
        elif inny == '2':
            return feet.backwardElimination()
        else:
            return feet.simulatedAnnealing()

    @staticmethod
    def dataAlgPrompt() -> Path:
        datPick = input(Printer.datAlgPrompt)
        if datPick == '1':
            return BIG_DATA
        elif datPick == '2':
            return SMALL_DATA
        elif datPick == "4":
            return DEMO
        else:
            return TITANIC

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
    parser = argparse.ArgumentParser(
                    prog='Feature Selection with Nearest Neighbor Classifier',
                    description='Given a dataset with a binary label as the first column, this program will attempt to classify the data. Using a leave one out validator it will attempt to find the most accurate feature set for classification.',
                    epilog='Text at the bottom of help')
    parser.add_argument('--customdata', '-d', type=Path, default=None, help='Provide the path to a custom dataset.')
    parser.add_argument('--testdata', default='titanic',choices=["bigdata","smalldata","titanic"],help="Use a provided test/sample dataset")
    parser.add_argument('--search', '-s',choices=["forward","backward","simulated-annealing"],default='forward',help='Pick the feature search method\n 1.[forward] Selection\n2.[backward] Elminiation\n3.simulated-annealing')
    parser.add_argument('--debug', type=bool, default=False, help="Display debug info during run" )
    parser.add_argument('--NN', '-k', type=int, default=3, help="Set the k value for nearest-neighbor. How many neighbors should be considered?")
    parser.add_argument("--normalization",'-norm',action="store",default='min-max', choices=['min-max','std-normal','numpy','none'], help="Set the method of data normalization\nIf your data is already normalize, use 'none'")
    
    print(Printer.mainWelcome)
    args=parser.parse_args()
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    #Data
    dadi = Data(args.normalization)
    if args.customdata:
        dataPath = args.customdata
    else:
        match args.testdata:
            case "smalldata":
                dataPath=SMALL_DATA
            case "bigdata":
                dataPath=BIG_DATA
            case _:
                dataPath=TITANIC
    dadi.loadTestData(dataPath)
    
    classi=Classifier()
    classi.train(dadi,args.NN)
    
    vally=Validator()
    
    feet = FeatureSearch(vally,dadi,classi)
    match args.search:
        case "forward":
            feet.forwardSelection()
        case "backward":
            feet.backwardElimination()
        case "simulated-annealing":
            feet.simulatedAnnealing()


    