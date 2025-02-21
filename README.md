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

*   **Swiss Roll Dataset (`pca_dim_reduction.py`):**
    *   Description: Synthetic dataset that forms a 3D manifold resembling a Swiss roll.
    *   Source: sklearn.datasets.make_swiss_roll
    *   *Note: this dataset is generated directly from scikit-learn and does not require a separate file.*

*   **Digits Dataset (`model_selection.py`, `gmm_km.py` ,`adaboost.py`):**
    *   Description: The MNIST dataset is used for recognizing handwritten digits.
    *   Source:  from 'sklearn.datasets import load_digits'
    *    *Note: this dataset is loaded directly from scikit-learn and does not require a separate file.*


## Usage
  
* Clone the Repository:

```
git clone [repository URL]
cd machine_learning
```
* Install Dependencies:
```
pip install -r requirements.txt  
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

