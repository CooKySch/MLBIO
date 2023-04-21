# MLBIO
GitHub repository for the reproduction of figures for the Lindel paper. This was done as part of the course "Machine learning for bioinformatics" (CS4260), at Delft University of technology.

**Students:** Gijs Admiraal, Bianca-Maria Cosma, Ioanna Nika, Connor Schilder.

This project is a reproduction. We extended the code made public in the [Lindel](https://github.com/shendurelab/Lindel) repository, and we credit the following publication:

Wei Chen, Aaron McKenna, Jacob Schreiber, Maximilian Haeussler, Yi Yin, Vikram Agarwal, William Stafford Noble, Jay Shendure, Massively parallel profiling and predictive modeling of the outcomes of CRISPR/Cas9-mediated double-strand break repair, _Nucleic Acids Research_, Volume 47, Issue 15, 05 September 2019, Pages 7989–8003, https://doi.org/10.1093/nar/gkz487

The files used to train the indel, deletion, and insertion logistic regression models can be found in directory `Lindel-data_analysis/scripts/Logistic model training`. We extended `LR_deletion.py` with detailed comments. Since the three files are very similar, we added less detailed documentation to the other two training scripts.

# INSTALL ENVIRONMENT
Environment is given in a yml file. If you do not have this environment installed, run the following command:
`conda env create -f mlbio.yml`

If you do have this environment installed, update it with:
`conda env update --file mlbio.yml --prune`

# REPRODUCTION OF FIGURES
To reproduce figures 6B and 6E from the original paper (shown below), run the following command: `python3 Lindel/plot_6B6E.py`

## Our results

We successfully reproduced figures 6B and 6E from Chen et al., and we corroborate their findings:

![Model](results.png)

# INDIVIDUAL RESEARCH

Each team member conducted individual research and extended the experiments of the original authors. The code can be found in folders named after each student.

## Gijs

## Bianca
I investigated the effect of dimensionality reduction on the Lindel deletion model. I applied multiple correspondence analysis (MCA) with various numbers of components and plotted the performance of these smaller models. All of the code can be found and run with the notebook file `mca.ipynb`.

## Ioanna
I replaced logistic regression models with random forest models. This was done to test and identify whether the performance of a non-linear tree-based classifier is better than the performance of the logistic regression model used which is a linear classifier. The relevant scripts and results can be found in Lindel-data_analysis/scripts/RF_training. Specifically, RF_deletion.py trains a random forest model for the deletion events, RF_insertion.py trains a random forest model for the inserion events and RF_indel.py rains a random forest model for predicting the indel ratio. The file gen_predictions.py uses all the trained models to make and output predictions. The file plot_predictions.py generates the frameshift and histogram plots given the predictions. The rest of the scripts correspond to ablations and analysis. Specifically, the files that end with "_combined" corresponnd to the ablation where the logistic regression and the random forest models is used. The notebook analysis_per_class.ipynb contains the code to plot the MSEs for frameshift annd non frameshift classes. Finally, the folder Ablations contains figures of results when different hyperparameters for the random forest where tested to identify their effect on performance. 

## Connor
