
import sys
import os
import argparse
import pandas as pd
import glob
import datetime as dt
import pickle as pkl
from tensorflow import keras 
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for some encoded feautures")
    parser.add_argument('-f', '--file_with_seqs', dest='sequences', default= "Lindel-data_analysis/data/algient_NHEJ_guides_final.txt", type=str, help="file path to the sequences")
    parser.add_argument('-guides', '--guides', dest='guides', default= "data/Lindel_test.txt", type=str, help="file with testdata and guide sequences")
    parser.add_argument('-o, --output_file', dest='output_file', default="data/Lindel_test_with_full_seqs.txt",type=str, help="File path of the output file")
    args = parser.parse_args()


    # read the full sequences, first column is the full sequence
    in_file = pd.read_csv(args.sequences, sep='\t', header=None)

    # filer by second column such that second  column = "70k seq design"
    in_file = in_file[in_file.iloc[:, 1] == "70k seq design"]

    # get the sequences from the first column
    full_seqs = in_file.iloc[:, 0]
    

    # read the guide sequences, first column is the guide sequence

    Lindel_test_file = pd.read_csv(args.guides, sep='\t', header=None)
    
    guides = Lindel_test_file.iloc[:, 0]
    

   
    # match the guide sequences with the full sequences, check if the guide sequence is in the full sequence at positions between 13 to 33
    # if it matches replace the guide sequence with the full sequence
    for i, guide_seq in enumerate(guides):
        count = 0
        for full_seq in full_seqs:
            if guide_seq in full_seq[20:40]:
                count+=1
                # get starting position of the second occurence of the guide sequence in the full sequence
                start = full_seq.rfind(guide_seq)
                start_13 = start-13
                end_47 = start+47

                full_seq = full_seq[start_13:end_47]
                Lindel_test_file.iloc[i, 0] = full_seq
        
            
        if count > 1:
            print("more than one match for guide sequence: ", guide_seq)
            
    
    # Remove file if it already exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    # save the file with the full sequences
    Lindel_test_file.to_csv(args.output_file, sep='\t', header=None, index=None)


    return

if __name__ == "__main__":
    sys.exit(main())