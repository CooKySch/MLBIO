import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

# Most of the code was taken from LR_deletion.py
def mse(x, y):
    return ((x-y)**2).mean()

# Load aggregate model predictions
aggregate_test_predictions = pkl.load(open("../../../data/aggregate_model_test_predictions.pkl", "rb"))

# Load true predictions
test_data = np.loadtxt("../../../data/Lindel_test.txt", delimiter="\t", dtype=str)
test_data = test_data[:,1:].astype('float32')

_, _, features = pkl.load(open('../../../data/feature_index_all.pkl','rb'))
feature_size = len(features) + 384
y = np.array(test_data[:, feature_size:])

mses = []
for i in range(len(y)):
    mses.append(mse(y[i], aggregate_test_predictions[i]))
print(mse(y, aggregate_test_predictions))
print(len(mses))
plt.hist(mses, bins=20)
plt.show()
