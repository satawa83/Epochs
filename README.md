# Crop Yield Prediction for Food Security in Africa.

## 1. Business Understanding

### 1.1 Project Overview

Agriculture plays an important role in providing food and supporting the economy in many African countries. However, many farmers find it difficult to estimate how much crop they will harvest before planting, and this can in turn lead to poor planning, low productivity, and financial losses.

The aim of this project is to build a machine learning system that predicts crop yield using soil information such as pH, nitrogen, phosphorus, potassium, soil moisture, and field size. This will help farmers make better farming decisions and improve food security.

### 1.2 Stakeholders.

##### The key statekeholders for this project are:

- Farmers and Farmer groups.
- Government Institutions.
- NGOs and development partners.
- Farm inputs providers and suppliers.
- Market Actors.
- Research Institutions and Academia.
- Food Security and Warning bodies.
- Financial Institutions & insurers.
- Consumers.

### 1.3 Success Metrics

- Highest R2 score.
- Lowest Mean Absolute Error (MAE).

---

## 2. Business Problem

Globally, 2.33 billion people experienced moderate or severe food insecurity in 2023 Africa has the world's highest undernourishment rate, affecting about 20% of its population Agricultural productivity remains low due to climate change, limited technology adoption and inadequate farming information In Kenya, about 98% of crop production depends on rainfall, making farming highly vulnerable to climate variability Food insecurity remains a major challenge in many African countries. Farmers often make planting decisions without knowing how much they are likely to harvest. This can lead to poor planning, inefficient use of fertilizers and other resources, and lower crop production. As a result, farmers may experience financial losses, while governments and agricultural organizations find it difficult to plan for future food needs. Thus creating a need for a a reliable system that can help predict crop yield before planting using soil data.

## 3. Objectives

### 3.1 General Objective

Support farmers in selecting suitable crops and improving agricultural productivity through informed decision-making.

### 3.2 Specific Objectives

- Estimate the expected harvest of different crops .
- Identify crops that are best suited to different farming conditions.
- Provide farmers with alternative crop options to improve production.
- Support farmers in making informed crop selection decisions.
- Promote higher agricultural productivity and sustainable farming practices

---

## 4. Dataset

| Property | Detail |
|---|---|
| Source: Local Soil Dataset
| File | `Editted2.xlsx` |
| Original records | 13,802 rows |
| Final records (after cleaning) | 11,783 rows |
| Crop types | 82 crops |
| Target variable | Target Yield (kg) |

**Feature descriptions:**

| Feature           | Unit     | Description                     |
| ----------------- | -------- | ------------------------------- |
| Target Yield      | kg       | Expected crop yield             |
| Field Size        | acres    | Size of the farm plot           |
| pH (water)        | -        | Soil acidity/alkalinity         |
| Organic Carbon    | g/kg     | Soil organic matter content     |
| Total Nitrogen    | g/kg     | Nitrogen content                |
| Phosphorus (M3)   | mg/kg    | Mehlich-3 phosphorus extraction |
| Potassium (exch.) | mmol+/kg | Exchangeable potassium          |
| Soil moisture     | %        | Soil moisture percentage        |
| Organic Matter    | kg       |
| Lime              | kg       |

## 5. Methodology: CRISP-DM

1. Business Understanding- Problem definition and stakeholder analysis
2. Data Understanding- Exploration of 13,802 rows across 82 crops
3. Data Preparation- Cleaning, imputation, encoding, scaling
4. Modelling- 6 models including deep learning (FT-Transformer)
5. Evaluation- Metric comparison and best model selection
6. Deployment- Interactive crop recommendation system

---

## 6. Data Preparation

| Step                         | Action                                          | Detail                                    |
| ---------------------------- | ----------------------------------------------- | ----------------------------------------- |
| Duplicates                   | Dropped 6 duplicate rows                        | `df.drop_duplicates()`                    |
| Missing Soil moisture (30%) | Compared KNN, Random Forest, XGBoost imputation | **Random Forest selected** (best R²)      |
| Outliers                     | Per-crop IQR method                             | Crop-specific ranges respected            |
| Dropped columns              | Removed `Organic Matter Need` and `Lime Need`   | Not available pre-planting                |
| Log transformation           | Applied `log1p` to Target Yield                 | Reduces skewness, improves model learning |
| Encoding                     | One-hot encoded `Crop Name`                     | 82 new binary columns added               |
| Scaling                      | `StandardScaler` on 7 numeric columns only      | Crop dummy columns left unscaled          |

Three methods were compared for imputing the ~30% missing Soil moisture values:

| Imputation Method | R²        | Selected |
| ----------------- | --------- | -------- |
| KNN               | 0.560     | No       |
| **Random Forest** | **0.796** | **Yes**  |
| XGBoost           | 0.619     | No       |

Based on the model outputs above, the Random Forest model was selected as the best to predict and impute the missing values for the "Soil Moisture" feature as it was the best performing model overall with the folowing metrics:

R² Score : 0.796
MAE      : 2.623
MSE      : 13.196
RMSE     : 3.633
---

## 7. Models

| Model                 | Type              | Configuration                          |
| --------------------- | ----------------- | -------------------------------------- |
| Ridge Regression      | Linear baseline   | alpha=1.0, L2 regularization           |
| Random Forest         | Ensemble          | n_estimators=100                       |
| XGBoost               | Gradient Boosting | n_estimators=300, lr=0.05, max_depth=8 |
| Random Forest (Tuned) | Ensemble          | RandomizedSearchCV, 5-fold CV          |
| XGBoost (Tuned)       | Gradient Boosting | RandomizedSearchCV, 5-fold CV          |
| FT-Transformer        | Deep Learning     | Custom PyTorch implementation          |

## **Random Forest tuning best parameters:**

n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_depth=30

---

## Results

| Model                 | MAE        | MSE        | RMSE       | R² Score   |
| --------------------- | ---------- | ---------- | ---------- | ---------- |
| Linear Regression     | 0.9405     | 1.6830     | 1.2973     | 0.4796     |
| **Random Forest**     | **0.6207** | **1.0681** | **1.0335** | **0.6697** |
| XGBoost               | 0.6598     | 1.0904     | 1.0442     | 0.6629     |
| Random Forest (Tuned) | 0.6348     | 1.0852     | 1.0417     | 0.6645     |
| XGBoost (Tuned)       | 0.6682     | 1.1217     | 1.0591     | 0.6532     |
| FT-Transformer        | 1.4245     | 3.2036     | 1.7899     | 0.0095     |

> All metrics computed on log-transformed Target Yield.

### ✅ Best Model: Random Forest (R² = 0.6697, RMSE = 1.0335)

Random Forest outperformed all other models including XGBoost and the FT-Transformer deep learning model, explaining approximately **67% of yield variance** from soil characteristics and crop type alone.

---

## 8. Key Findings

### Feature Importance (Random Forest)

| Rank | Feature                | Importance (%) |
| ---- | ---------------------- | -------------- |
| 1    | Field Size             | 16.64%         |
| 2    | Crop Name_sunflower    | 7.70%          |
| 3    | Crop Name_maize (corn) | 7.46%          |
| 4    | Soil moisture          | 7.16%          |
| 5    | Potassium (exch.)      | 7.08%          |

Field size is the most influential variable. Crop type and soil chemistry features follow.

### Data Exploration

- Organic Carbon and Total Nitrogen are strongly correlated (r = 0.92)
- Target Yield is heavily right-skewed- log transformation applied before modeling
- Maize had the highest number of records; melon-water and carrot had the fewest

### Model Performance

- Random Forest achieved the best performance (R²=0.67, MAE=0.62, RMSE=1.03)
- FT-Transformer underperformed (R²=0.009)- consistent with research showing transformer models require substantially larger datasets (>100,000 rows) to outperform tree-based ensembles on structured tabular data
- Hyperparameter tuning improved Random Forest marginally but did not significantly improve XGBoost

---

## 9. Crop Recommendation System

An interactive crop recommendation widget was built using `ipywidgets`. The system allows farmers or extension officers to input their soil readings and receive a ranked list of recommended crops with predicted yield estimates.

### How it works

1. Enter your soil test readings (pH, Organic Carbon, Nitrogen, Phosphorus,
   Potassium, Soil moisture) and field size
2. The trained Random Forest model predicts expected yield for every crop
3. Results are ranked from highest to lowest predicted yield
4. Farmer selects the best crop for their soil conditions
5. Farmer gets alternatives if results show higher and better ones.
6. Farmer doesnt get alternatives if the results they get is ranked the highest andbest prediction.

## 10. Findings.

### 10.1 Data Exploration Findings

- The dataset contained over 10,000 crop records from different crop types and soil conditions, providing enough data to build reliable prediction models.
- Removing missing values, duplicates, and crop-specific outliers improved the quality of the dataset.
- Applying a log transformation to the target yield reduced skewness and made the data more suitable for machine learning.

### 10.2 Model Findings

- Among all the models tested, the Random Forest Regressor achieved the best overall performance with an MAE (0.62), MSE(1.07), RMSE (1.03), and R² Score (0.67)
- The Linear Regression (Ridge) model performed the worst among the traditional machine learning models because it could not capture the complex relationships between soil properties and crop yield.
- The XGBoost model produced results that were similar to Random Forest but performed slightly worse on this dataset.
- Hyperparameter tuning did not improve the performance of either the Random Forest or XGBoost models.
- The F-T Transformer performed poorly, achieving an R² score of approximately 0.01, showing that deep learning was not suitable for this dataset due to its size and feature complexity.
- Feature importance analysis showed that Field Size was the most influential variable in predicting crop yield.
- The Random Forest model produced predictions that were generally close to the actual crop yields, although there were a few crops with extremely high or very low yields had larger prediction errors.

### 10.3 Business Findings

- The recommendation system successfully predicted the crop yield of a farmer's selected crop using the provided soil measurements and field size.
- The recommendation system also identified alternative crops that could produce higher yields under the same soil conditions, helping farmers make more informed planting decisions.

## 11. Recommendations

- Farmers should use this prediction model before planting to estimate the expected crop yield based on their soil properties and field size in order to make informed decisions.
- Farmers should also use the recommendation system to compare different crop options and select crops that are predicted to produce higher yields under the same soil conditions.
- Soil testing should be carried out before planting since soil properties such as soil moisture, pH, organic carbon, nitrogen, phosphorus, and potassium were found to have a significant influence on crop yield.
- Agricultural extension officers can use this system to provide farmers with data-driven advice on crop selection instead of relying only on experience.
- Government agencies and agricultural organizations can integrate this model into digital farming platforms to support food security planning and improve crop production.

---

## Limitations

- Dataset reflects expected/attainable yield potential from soil-test records- not literally measured farm harvests
- Climate variables (rainfall, temperature) are absent- adding them would likely improve R² significantly
- FT-Transformer underperformed due to dataset size; transformer architectures are better suited to larger datasets
- Predictions represent a yield estimate, not a guarantee

---
