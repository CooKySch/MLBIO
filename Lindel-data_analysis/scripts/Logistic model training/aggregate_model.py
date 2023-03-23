import sys, os
import pickle as pkl
import numpy as np

# Arguments: test_data, working directory (workdir) and file name (fname)
test_data = "Lindel_test.txt"
workdir = "data/"
fname   = "Lindel_training.txt"


# Load 1) labels: dictionary mapping 557 classes to indices 
#      2) rev_index: reverse indexing of labels (keys are indices and classes are values)
#      3) features: microhomology features (total of 2649; keys are feature names and values are indices)
label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
# Load training data
data = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)

# Load test data
test_data = np.loadtxt(workdir+test_data, delimiter="\t", dtype=str)

# Total feature size: add additional 384 sequence one-hot encoded features
feature_size = len(features) + 384
# Features and labels: columns 1 to 3590 (3033 [mh features + sequence features]; the last 557 columns are the classes (probability distribution))
data = data[:,1:].astype('float32')

# X = features; y = labels
X = np.array(data[:,:feature_size])
y = np.array(data[:, feature_size:])

# Adjust sizes of training and validation sets
train_size = round(len(data) * 0.9) if 'ForeCasT' in fname else 3900
valid_size = round(len(data) * 0.1) if 'ForeCasT' in fname else 450 

# Create predictions with the aggregate model
# "aggregate model, in which the predicted frequencies of 557 indel classes are simply taken from the aggregate frequency at which each is observed in the training and validation datasets"

y = y[:3900+450]
summed_columns = y.sum(axis = 0)
one_prediction = summed_columns / len(y) #summed_columns.sum()

# Aggregate predictions for validation data
aggregate_predictions_test_set = np.tile(one_prediction, (len(test_data), 1))

if os.path.exists(workdir + "aggregate_model_test_predictions.pkl"):
    os.remove(workdir + "aggregate_model_test_predictions.pkl")

# Save predictions to file
with open(workdir + 'aggregate_model_test_predictions.pkl', 'wb') as f:
    pkl.dump(aggregate_predictions_test_set, f)