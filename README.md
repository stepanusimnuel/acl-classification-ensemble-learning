# Anterior Cruciate Ligament Injury Classification: A Comparison of Ensemble Learning Algorithms

## Description

This repository implements ACL injury classification using ensemble learning techniques:

- **Random Forest**
- **Stacking SVM** (Linear, Polynomial/Cubic, RBF kernels)
- **Stacking KNN** (Euclidean, Manhattan, Weighted distances)

---

## Dataset

File: `metadata.csv` from [KneeMRI Dataset - Zenodo](https://zenodo.org/records/14789903)  
Contains ROI (Region of Interest) data from knee MRI slices.  
Target column: `aclDiagnosis`  
The dataset is normalized using **Z-score**, and class imbalance is handled using **SMOTE**.

- **0** = Healthy
- **1** = Partially ruptured
- **2** = Completely ruptured

---

## Project Structure

| File / Folder           | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `models/`               | Folder containing trained model and scaler data (.pkl) |
| ├── `Random Forest.pkl` | Random Forest model                                    |
| ├── `Stacking SVM.pkl`  | Stacking SVM model                                     |
| ├── `Stacking KNN.pkl`  | Stacking KNN model                                     |
| └── `scaler_data.pkl`   | Dictionary containing mean & std for scaling           |
| `metadata.csv`          | The main dataset                                       |
| `app.py`                | Main application file (Flask or CLI interface)         |
| `requirements.txt`      | Python dependencies                                    |
| `train-model.ipynb`     | Notebook for EDA, preprocessing, training, testing     |

---

## How to run

### 1. Clone repository

```bash
git clone https://github.com/stepanusimnuel/acl-classification-ensemble-learning.git
cd acl-classification-ensemble-learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain models

Open, edit, and run `train-model.ipynb` if you want to perform EDA, preprocessing, training, and evaluation from scratch.

### 4. Run the application

```bash
python app.py
```

---

#### Notes

If you are using this dataset in your work, please acknowledge the source (Clinical Hospital Centre Rijeka, Croatia) and reference this paper:  
I. Štajduhar, M. Mamula, D. Miletić, G. Unal, Semi-automated detection of anterior cruciate ligament injury from MRI, Computer Methods and Programs in Biomedicine,
Volume 140, 2017, Pages 151–164, https://doi.org/10.1016/j.cmpb.2016.12.006.
