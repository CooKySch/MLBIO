
import sys
import os
import argparse
import pandas as pd
import glob
import datetime as dt
import pickle as pkl
from tensorflow import keras 



def main():
    parser = argparse.ArgumentParser(description="Generate predictions for some encoded feautures")
    parser.add_argument('-f', '--file_encoded_features', dest='file_name', default= "data/Lindel_test.txt", type=str, help="File path of the encoded features")
    parser.add_argument("-m", '--model', dest='model', type=str, help="File path of the model")
    parser.add_argument('-o, --output_file_predictions', dest='output_file', default="data/predictions",type=str, help="File path of the output file")
    args = parser.parse_args()

    # Read the encoded features
    label, rev_index, features = pkl.load(open("data/feature_index_all.pkl", "rb"))
    df = pd.read_csv(args.file_name, sep='\t', header=None)

    feature_size = len(features) + 384
    data = df.iloc[:, 1:].astype('float32')

    #X = features, y = labels
    X = data.iloc[:, :feature_size]
    y = data.iloc[:, feature_size:]

    # Load the model h5 file with keras
    model = keras.models.load_model(args.model)

    # Generate predictions
    predictions = model.predict(X)
    print("Here are the predictions:")
    print(predictions)






    return