# 🏠 California Housing Price Prediction

A machine learning regression project that predicts **median house values in California** using demographic, geographic, and housing-related features.

The project covers an end-to-end machine learning workflow including data preprocessing, exploratory analysis, model training, model comparison, hyperparameter tuning, evaluation, feature importance analysis, and Streamlit deployment.

---

## 🚀 Live Demo

**Streamlit App:**
https://california-housing-price-prediction-jaqjutb5wt9w8rvygva2hn.streamlit.app/

> The deployed application is currently being improved to ensure that predicted values are displayed in the correct real-world price scale.

---

## 📌 Project Overview

The objective of this project is to build a regression model that can estimate California housing values based on factors such as:

* Geographic location
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Median income
* Ocean proximity

The project compares multiple regression algorithms and uses hyperparameter tuning to improve the best-performing model.

---

## 📊 Dataset

The dataset contains California housing information with geographic, demographic, and housing-related variables.

### Features

| Feature              | Description                                        |
| -------------------- | -------------------------------------------------- |
| `longitude`          | Geographic longitude of the housing block          |
| `latitude`           | Geographic latitude of the housing block           |
| `housing_median_age` | Median age of houses in the housing block          |
| `total_rooms`        | Total number of rooms in the housing block         |
| `total_bedrooms`     | Total number of bedrooms in the housing block      |
| `population`         | Total population in the housing block              |
| `households`         | Total number of households in the housing block    |
| `median_income`      | Median household income                            |
| `<1H OCEAN`          | Housing block is less than one hour from the ocean |
| `INLAND`             | Inland location                                    |
| `ISLAND`             | Island location                                    |
| `NEAR BAY`           | Near the bay                                       |
| `NEAR OCEAN`         | Near the ocean                                     |

### Target

`median_house_value`

The target represents the median house value for the housing block.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle
* Git
* Git LFS

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## 🤖 Models Compared

The following regression models were evaluated:

1. Decision Tree Regressor
2. Gradient Boosting Regressor
3. Random Forest Regressor

Random Forest performed the best among the tested models.

---

## 🎯 Hyperparameter Tuning

Hyperparameter tuning was performed using cross-validation.

The best Random Forest configuration was:

```text
n_estimators = 200
max_depth = None
min_samples_split = 2
min_samples_leaf = 2
```

---

## 📈 Model Performance

### Tuned Random Forest

| Metric   | Score |
| -------- | ----: |
| MAE      | 0.273 |
| RMSE     | 0.423 |
| R² Score | 0.818 |

### What the metrics mean

**MAE — Mean Absolute Error**

Measures the average absolute difference between predicted and actual values.

**RMSE — Root Mean Squared Error**

Measures prediction error while giving more weight to larger errors.

**R² Score**

Measures how much of the variation in the target is explained by the model.

An R² score of approximately **0.818** indicates that the tuned Random Forest explains about **81.8% of the variance** in the target on the test data.

---

## 🔍 Feature Importance

The Random Forest model identified the following features as particularly important:

| Feature              | Importance |
| -------------------- | ---------: |
| `median_income`      |      0.499 |
| `INLAND`             |      0.145 |
| `longitude`          |      0.106 |
| `latitude`           |      0.101 |
| `housing_median_age` |      0.051 |

`median_income` was the most influential feature in the trained model.

Permutation importance was also used to understand how much model performance changes when individual features are randomly shuffled.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can enter housing information and receive a model prediction.

The application provides user-friendly inputs for:

* Longitude
* Latitude
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Median household income
* Ocean proximity

The application also includes input validation and model performance information.

---

## 💾 Model Serialization

The trained Random Forest model is saved using Python's `pickle` module:

```python
with open("california_housing_rf.pkl", "wb") as file:
    pickle.dump(rf_model, file)
```

Because the trained model file is large, **Git LFS** is used to manage the model file.

---

## 📁 Project Structure

```text
california-housing-price-prediction/
│
├── app1.py
├── california_housing_rf.pkl
├── requirements.txt
├── README.md
└── .gitattributes
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/gunaprakash205/california-housing-price-prediction.git
```

Move into the project directory:

```bash
cd california-housing-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app1.py
```

---

## 📦 Requirements

The project uses the following main Python libraries:

```text
streamlit
pandas
scikit-learn
```

---

## 📌 Future Improvements

* Use the original unstandardized target values for direct real-world price prediction
* Improve target-value interpretation and inverse transformation
* Add additional regression algorithms
* Perform more extensive hyperparameter optimization
* Add prediction confidence/range information
* Improve model explainability
* Add interactive visualizations
* Further improve the Streamlit user interface

---

## 💡 Key Learning Outcomes

Through this project, I worked with:

* Regression problems
* Data preprocessing
* Feature engineering
* Train-test splitting
* Decision Trees
* Random Forest
* Gradient Boosting
* Hyperparameter tuning
* Cross-validation
* MAE, RMSE, and R² evaluation
* Feature importance
* Permutation importance
* Model serialization using Pickle
* Git and Git LFS
* Streamlit deployment

---

## 👨‍💻 Author

**Guna Prakash**

GitHub:
https://github.com/gunaprakash205

---

## ⭐ Project Status

**Active — deployed and being refined**

The core machine learning pipeline and deployment are functional. The target-value scaling/display is being refined to ensure predictions are represented correctly in real-world house-price units.
