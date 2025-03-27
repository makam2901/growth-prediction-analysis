# Growth Prediction & Analysis
This project talks about EDA, Data Problems and Model Development for growth prediction of simulated alien data.

## Project Overview

This project aims to develop a machine learning model to predict the growth curves of second-generation subjects (Gen2) from ages 10 to 18 using their height and weight measurements from ages 0 to 9, along with their parents’ complete growth data when available. The dataset spans two generations, with longitudinal growth records from infancy to adulthood. The challenge involves leveraging individual growth trajectories and potential intergenerational influences to forecast future development. Model performance will be evaluated in a Kaggle competition, assessing predictions against actual measurements. This report details the end-to-end process, including data preprocessing, feature engineering, model selection, and evaluation, showcasing the application of machine learning to longitudinal growth prediction.

## Exploratory Data Analysis & Preprocessing

### 1. Data Overview
The dataset consists of two primary components: 
 
* Gen1 data: Growth measurements of parents (first generation) from infancy to adulthood 
(age 20) 
* Gen2 data: Growth measurements of children (second generation), with early 
measurements (ages 0-9) available for model training and later measurements (ages 
10-18) as prediction targets

### 2. Growth Trajectory Visualizations
<img width="991" alt="image" src="https://github.com/user-attachments/assets/776d1f5e-ec43-4dd2-a995-7d2201ca3ff2" />
<img width="1001" alt="image" src="https://github.com/user-attachments/assets/100b261b-38e9-4046-8f33-63059b9df60e" />

Height growth trajectories for both generations (Gen1 and Gen2) follow typical patterns, with rapid early growth, steady middle-phase growth, and a pubertal growth spurt. Gen2 curves are more tightly clustered, suggesting multiple offspring of Gen1 subjects. Males experience their growth spurt between ages 10–15, reaching peak height by 17–18, while females begin earlier (9–14) and plateau by 15–16. Males tend to be taller and show more variability, whereas female growth patterns are more consistent.

### 3. Intergenerational Height Correlation
<img width="1003" alt="image" src="https://github.com/user-attachments/assets/7a969ed4-5615-43bb-a4c4-01c6c0310356" />

Scatter plots indicate a stronger correlation between a child’s height at age 5 and their father’s adult height compared to their mother’s, as shown by tighter clustering of paternal height data points. This suggests that paternal height has a greater influence on early childhood growth, regardless of the child’s gender.

### 4. Data Missingness
<img width="996" alt="image" src="https://github.com/user-attachments/assets/1ec177e6-ce51-4e9c-b126-a33f3082566a" />

Gen1 data is mostly complete in early childhood but becomes inconsistent after age 10, with significant gaps in adolescence (15–20 years). Gen2 data is more consistently recorded but shows a decline in completeness from age 15, dropping sharply by 17. This suggests improved data collection practices in Gen2 compared to Gen1.

#### Addressing Missing Data
* KNN Imputation: Used to estimate missing height values by leveraging similar subjects, improving dataset consistency. Sex-based grouping enhances accuracy.
* Feature-Based Imputation: The Preece-Baines model was applied for biologically plausible estimates, ensuring realistic growth trends.

### 5. Handling Irregular Measurement Intervals
* Fixed Age Grid Creation: Measurements were mapped to standardized age points to ensure consistency across subjects.
* Growth Velocity Calculation: Growth rate was computed between age intervals to accurately capture developmental trends despite irregular measurement intervals.

### 6. Outlier Management
<img width="1015" alt="image" src="https://github.com/user-attachments/assets/07c6ec3a-7bbe-42ab-952f-ad69237c5d42" />

Box plots reveal more outliers in Gen1, particularly during adolescence, while Gen2 exhibits a more consistent height distribution. Differences may be influenced by genetic, environmental, or nutritional factors.

#### Addressing Outliers
* IQR-Based Clipping: Outliers were adjusted within interquartile range (IQR) limits to reduce their impact while preserving data integrity.
* Biological Plausibility Constraints: Growth predictions were bounded within physiological limits to maintain realistic outputs.
* Parent-Informed Initialization: Parental height data was used as an informative prior to refine model accuracy and reduce bias.

### 7. Feature Engineering

A structured feature set was developed by integrating individual (Gen2) and parental (Gen1) height data. Growth velocities were computed to capture rate changes over time. Parental height at key ages (5, 10, 15, 20) and adult height were included to account for genetic influences, improving prediction accuracy of adolescent growth trajectories.


## CatBoost Model Development

### 1. Model Architecture
- **Hyperparameter optimization strategy**: Implemented RandomizedSearchCV to tune key parameters including learning rate (0.01-0.1), tree depth (4-10), L2 regularization (1-7), and number of iterations (1000-3000).
- **Cross-validation approach**: Utilized 5-fold cross-validation with stratified sampling based on gender to ensure balanced representation in each fold.
- **Feature importance analysis**: Generated SHAP values and feature importance plots to identify key predictors, with early childhood heights and parental measurements emerging as top features.
- **Model interpretation methods**: Employed partial dependence plots and SHAP interaction values to understand feature relationships and their impact on predictions.

### 2. Model Evaluation & Selection

#### Evaluation Framework
- **Performance metric selection**: Chose RMSE as primary metric to penalize larger errors more heavily, supplemented by MAE and R² for comprehensive evaluation.
- **Cross-validation strategy**: Implemented time-series cross-validation to maintain temporal order of measurements and prevent data leakage.
- **Bias-variance analysis**: Conducted learning curve analysis to optimize model complexity and prevent overfitting.
- **Model comparison methodology**: Compared performance against baseline models including linear regression and random forests using consistent evaluation metrics.

#### Selection Criteria
- **Prediction accuracy metrics**: Evaluated models using RMSE on validation set, with emphasis on consistent performance across different age groups.
- **Model robustness measures**: Assessed model stability through bootstrap resampling and sensitivity analysis to outliers.
- **Computational efficiency**: Measured training time and prediction speed to ensure practical deployment capabilities.
- **Interpretability considerations**: Prioritized models that provided clear feature importance rankings and interpretable predictions for clinical applications.

## Results
Best hyperparameters for each model of target age is stored in a seperate file 'best_hyperparameters.txt'

## Technical Implementation

### Tech Stack
- **Python**: Core implementation
- **Libraries**: 
  - pandas, numpy: Data manipulation
  - scipy: Growth model fitting
  - CatBoost: Gradient boosting
  - scikit-learn: Model validation
  - matplotlib, seaborn: Visualization
