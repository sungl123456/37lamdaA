37lambdaA

***

This code is the basis of our work submitted to *Chemical Engineering Science*, aiming to train and apply the artificial neural network and random forest in the prediction of Vapor-liquid equilibrium data under ambient pressure for binary mixtures with limited descriptors. This is an **alpha version** of the codes used to generate the results published in:

[Vapor-liquid phase equilibria behavior prediction of binary mixtures using machine learning]

Chemical Engineering Science

DOI:

## Prerequisites and dependencies

The code is written in python and can be run from command prompt. This requires the following packages and modules:

* Python 3.8 or higher
* torch==2.0.1
* pandas==1.4.3
* numpy==1.24.3
* scikit_sklearn==1.3.0
* matplotlib==3.7.2
* tensorboardX==2.6.2.2
* openpyxl==3.1.2

## Data

The data that used in the original work is available as three excel documents (.xlsx). Three files are included:

* `dataset.xlsx` All of the data used in the training, the size of data collected was doubled by swapping and rejoining the descriptors of component A and component B.
* `VLE_input.xlsx` The screened descriptors of the substances with the liquid fractions and the pressure (mainly ambient pressure) in vapor-liquid equilibrium.
* `VLE_output_T.xlsx` The bubble point temperature in vapor-liquid equilibrium.
* `VLE_output_Y.xlsx` The vapor fraction in vapor-liquid equilibrium.
* `mixtures.xlsx` All of the mixtures collected.

## Cross-validation
To validate the feasibility of the method as well as to evaluate the training, do:
```commandline

python cross_validation.py

```

The data mentioned above will be disrupted as mixture-based unit for training and evalidation.

## Generate models for prediction
To generate models for the whole dataset with the descriptors contribute the most, do:
```commandline

cd all

python train_all.py

```

## Prediction
To acquire the T-xy data for target mixture, do:

```commandline

python predict.py

```
The name of the components of the target mixture is required to input sequentially, for example,

water
methanol

The name of the substance is also listed in `all//Tc_Tb.csv`.

The prediction results by ANN and RF will be written in `water_methanol_VLE_predict.csv`. 

We suggest the users will further regressed the prediction results with traditional thermodynamic models such as NRTL functions, for a practical use. 

We also suggest the users will expand the data set in the `all` folder for a more accurate prediction. Only the critical temperature and boiling point and the T-xy data are required.
