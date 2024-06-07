# MiPowerProject

## Water Quality Analysis

## Motivation and Background
The analysis of the impact of water quality on potability holds significant real-world importance. Water is a fundamental resource essential for human health. However, the potability of water depends on its safety and suitability for drinking purposes.

Through this analysis, it becomes possible to understand and evaluate the factors affecting the potability of water. For example, the effects of water properties such as pH value, hardness level, and chemical composition on potability can be examined. This information provides crucial data for improving water resource management and water supply processes.

Additionally, determining the pollution levels of water sources and assessing their compliance with potability standards is important for health institutions, organizations responsible for water supply, and decision-makers. Being informed about the potability status of water sources allows for the implementation of measures to minimize health risks and protect public health.

In conclusion, evaluating the impact of water quality on potability offers valuable insights into the protection of water resources, the safety of water supply, and the maintenance of health standards. This analysis supports the sustainable use of water resources, yielding positive effects on human health and the environment.

## Dataset Information
This dataset is a water quality analysis dataset that includes characteristics of water quality and potability status. Below are some details about the dataset:

- The dataset comprises a total of 3,276 observations and 10 variables.
- The variables include physical and chemical properties of water such as pH value, hardness level, chloride, sulfate, and temperature.
- The potability status represents the drinkability of the water. A potability value of 1 indicates that the water is drinkable, while a value of 0 indicates that the water is not drinkable. Therefore, there are 2 classes (Drinkable: 1 and Not Drinkable: 0).
- There are missing values in some variables in the dataset. These missing values need to be taken into account when performing analysis.
- Statistical analyses, visualizations, and machine learning algorithms can be applied to the dataset to examine the relationships and effects of water quality on potability status.

This dataset can be used to understand the potability status of water sources, improve water quality, and take measures for drinking water safety. Analyses performed on this dataset can play a crucial role in the management of water resources, optimization of water supply processes, and provision of safe drinking water.

The dataset contains 491 missing values in the pH column, 781 in the sulfate column, and 162 in the trihalomethanes column. In total, there are 1,434 missing values.

**Data Set:** [Water Potability Dataset on Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## Algorithms Used
- Logistic Regression
- Support Vector Machines (SVM Classification)
- Random Forest

## Libraries Used
The libraries used include Flask, Pandas, NumPy, Matplotlib, and Plotly. The code creates a Flask application and performs data analysis tasks.

## Data Analysis Steps
The data analysis steps include:
- Data loading
- Generating summary statistics
- Missing data analysis
- Outlier detection
- Data visualization

## Algorithm Performance
- Logistic Regression: 63.1% accuracy
- SVC: 63.3% accuracy
- Random Forest: 77.8% accuracy (best performance)

## Algorithm Performance (After the changes(outliers, missing data, categorization))
- Logistic Regression: 57.7% 
- SVC: 57.7% 
- Random Forest: 69.5% 
- XGBoost: 66.2%
- LightGBM: 69.1%
- CatBoost : 72.6%
- AdaBoost: 58.0%

## Prediction
The web application provides users with the ability to make predictions. The prediction results are presented to the user. When tested with sample inputs from the dataset, the prediction results are accurate. This demonstrates that the model can produce correct results with sample inputs from the dataset.

https://github.com/Buse-ng/MiPowerProject/assets/69714956/454732ce-a9be-487b-af52-dd272732ccad

![pred](https://github.com/Buse-ng/MiPowerProject/assets/69714956/853c5d5e-a938-4f6b-929f-fe1fcdddf47f)
