
import argparse
import numpy as np
import pickle as pkl
import sys
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import seaborn as sns
import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser(description="RF deletion model training")
    args = parser.parse_args()

    # Load 1) labels: dictionary mapping 557 classes to indices 
    #      2) rev_index: reverse indexing of labels (keys are indices and classes are values)
    #      3) features: microhomology features (total of 2649; keys are feature names and values are indices)
    label,rev_index,features = pkl.load(open('data/feature_index_all.pkl','rb'))
    # Total feature size: add additional 384 sequence one-hot encoded features
    feature_size = len(features) + 384
    # Load training data
    data     = np.loadtxt("data/Lindel_training.txt", delimiter="\t", dtype=str)
    
    Seqs = data[:,0]
    # Features and labels: columns 1 to 3590 (3033 [mh features + sequence features]; the last 557 columns are the classes (probability distribution))
    data = data[:,1:].astype('float32')

    # X = features; y = labels
    X = data[:,:feature_size]
    y = data[:, feature_size:]

    # Select sequences corresponding to shuffled indices
    Seq_train = Seqs
    x_train = []
    y_train = []
    # train_size = int(len(data) * 0.9)
    train_size = len(data)

    # Create training set with
    for i in range(train_size):
        # Check that labels for each sample sum up to at most 1
        if 1> sum(y[i,:536])> 0:
            # Normalize (probability distribution)
            y_train.append(y[i,:536]/sum(y[i,:536]))
            x_train.append(X[i])

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # create random forest model MultiOutputRegressor for soft classification with custom loss function 
    clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=20, max_depth=10, random_state=0, n_jobs=-1, bootstrap=True, oob_score=True), n_jobs=-1)

    

    # Train model
    clf.fit(x_train, y_train)

    # save model
    pkl.dump(clf, open('data/RF_deletion.pkl', 'wb'))


    return

if __name__ == "__main__":
    sys.exit(main())