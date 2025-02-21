# Machine Learning Algorithms and Techniques

This repository contains Python scripts demonstrating fundamental machine learning algorithms and techniques. The code is designed to be modular and readable, enabling easy exploration and experimentation with different approaches.

## Structure

The repository is structured as follows:

*   **`data/`**: This directory contains the datasets used by the scripts. Note: You will need to download the datasets separately (see below) and place them in this directory.
*   **`adaboost.py`**: Demonstrates the AdaBoost algorithm, including experiments on the digits dataset.
*   **`dtree_knn.py`**: Explores nonparametric methods, specifically k-Nearest Neighbors (k-NN) and Decision Trees for classification.
*   **`gmm_km.py`**: Implements and compares Gaussian Mixture Models (GMMs) and k-means clustering.
*   **`kernel_approx.py`**: Explores kernel matrix approximation techniques.
*   **`lda_qda.py`**: Implements and compares Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).
*   **`logistic_regression.py`**: Implements and explores Logistic Regression.
*   **`model_selection.py`**: Demonstrates model selection techniques using GridSearchCV.
*   **`pca_dim_reduction.py`**: Explores dimensionality reduction techniques, including PCA.
*   **`rf_bagging.py`**: Experiments with Bagging Regressor and Random Forest Regressor ensemble techniques.
*   **`svm.py`**: Focuses on Support Vector Machines for both classification and regression tasks.
*   **`utils.py`**: Contains reusable helper functions for data generation, plotting, and other common tasks.

## Datasets

The scripts in this repository use the following datasets. You will need to download these datasets separately and place them in the `data/` directory.

*   **`data/dat.txt`**: Used by `pca_dim_reduction.py`. The structure and meaning of this file are unknown, however the PCA functions will not run if it is not created.
*   **Heart Disease Dataset (`svm.py`):**
    *   Source: Cleveland dataset from the UCI Machine Learning Repository ([https://archive.ics.uci.edu/ml/datasets/Heart+Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)).
    *   File: `data/heart.csv` (This file needs to be downloaded separately and placed in the `data/` folder.)

*   **Golub Gene Expression Dataset (`model_selection.py`, `pca_dim_reduction.py`, `svm.py`):**
    *   Description: Gene expression data used for leukemia classification.
    *   Files: `data/Golub_X` (observations), `data/Golub_y` (classes) (These files need to be downloaded separately and placed in the `data/` folder.)

*   **Breast Cancer Wisconsin (Diagnostic) Dataset (`model_selection.py`, `pca_dim_reduction.py`, `svm.py`):**
    *   Description: Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    *   File: `data/Breast.txt` (This file needs to be downloaded separately and placed in the `data/` folder.)

*   **SPLEX Dataset (`logistic_regression.py`, `rf_bagging.py`, `svm.py`):**
    *   Description: Host and environmental data of obese patients.  Includes environmental (`SPLEX_env.txt`), host (`SPLEX_host.txt`), and microbial (`SPLEX_micro.txt`) data, along with class labels (`classes.csv`).
    *   Files: `data/SPLEX_env.txt`, `data/SPLEX_host.txt`, `data/SPLEX_micro.txt`, `data/classes.csv` (These files need to be downloaded separately and placed in the `data/` folder.)
*   **Mouse Protein Expression Dataset (`gmm_km.py`):**
    *   Description: Data related to protein expression levels in the cerebral cortex of mice.
    *   File: The dataset should be downloaded from : [https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls](https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls)
    *   Save the file as `Data_Cortex_Nuclear.xls` in the `/data` directory.

*   **Diabetes Dataset (`rf_bagging.py`):**
    *   Description: Diabetes-related data for regression tasks.
    *   Source: sklearn.datasets.load_diabetes
    *   *Note: this dataset is loaded directly from scikit-learn and does not require a separate file.*

*   **Dynamic Data for Diabetes Remission (`rf_bagging.py`):**
    *   Description: Dynamic data with HbA1C (glycated hemoglobin), Gly (glycemia), Poids (weight of patients), and Status (remission, non-remission, or partial remission) for time 0, 1 and 5 years after the surgery.
    *   File: `data/dynamic.txt` (This file needs to be downloaded separately and placed in the `data/` folder.)

*   **Swiss Roll Dataset (`pca_dim_reduction.py`):**
    *   Description: Synthetic dataset that forms a 3D manifold resembling a Swiss roll.
    *   Source: sklearn.datasets.make_swiss_roll
    *   *Note: this dataset is generated directly from scikit-learn and does not require a separate file.*

*   **Digits Dataset (`model_selection.py`, `gmm_km.py` ,`adaboost.py`):**
    *   Description: The MNIST dataset is used for recognizing handwritten digits.
    *   Source:  from 'sklearn.datasets import load_digits'
    *    *Note: this dataset is loaded directly from scikit-learn and does not require a separate file.*

## Dependencies

To run the code in this repository, you will need to install the following Python libraries:

```
pip install pandas numpy scikit-learn matplotlib pyAgrum tensorflow scipy

```

## Usage
  
* Clone the Repository:

```
git clone [repository URL]
cd machine_learning
```
* Install Dependencies:
```
pip install -r requirements.txt  # (Create a requirements.txt file based on the libraries in the requirements list.)
```
Download Datasets: Download the dataset files.

Run the Scripts:

To execute a specific script, use the following command:
`python <script_name>.py`

For example:
```
python adaboost.py
```

## Contact
If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [Specify License - e.g., MIT License] - see the LICENSE file for details.

