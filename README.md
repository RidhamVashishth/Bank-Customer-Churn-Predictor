# **Customer Churn Prediction App(Based on a Business-Centric Modeling Workflow)**

## **1\. Project Objective and Problem Statement**

In the highly competitive financial services sector, customer attrition, or "churn," represents a significant challenge to sustained profitability and market share. The primary objective of this project is to develop and validate a robust machine learning model capable of accurately identifying customers who are at a high risk of terminating their relationship with the bank.

The project extends beyond simple classification. It aims to deliver a practical, data-driven tool that facilitates strategic business decisions. This is achieved by systematically evaluating model performance not just on technical metrics, but also on custom business-oriented scores. This enables a nuanced understanding of the trade-offs between various retention strategies, from broad-reach campaigns to highly targeted, cost-efficient interventions.

Users can access the app by visiting: https://bank-customer-churn-predictor-ridham.streamlit.app/

## **2\. Initial Data Analysis and Key Findings**

An initial exploratory data analysis was conducted on the Churn\_Modelling.csv dataset to inform our modeling strategy. The key findings were as follows:

* **Significant Class Imbalance:** The dataset exhibits a notable class imbalance, with only **20.4%** of customers having churned. This observation is critical, as a naive model could achieve high accuracy by defaulting to the majority class ("no churn"), rendering it ineffective for our primary goal of identifying at-risk customers. Consequently, accuracy was deemed an insufficient metric for model evaluation.  
* **Identification of Predictive Features:** Correlation analysis indicated that variables such as Age, Balance, NumOfProducts, and IsActiveMember possess considerable predictive power in relation to the churn outcome.  
* **Geographic Disparities:** Churn rates were observed to vary significantly by Geography, with customers in Germany demonstrating a higher attrition rate. This underscored the necessity for appropriate encoding of this categorical feature to capture its full predictive value.

## **3\. The Systematic Modeling and Evaluation Workflow**

A comprehensive, multi-stage workflow was implemented to ensure a rigorous and reproducible approach to model development and selection.

### **Step 1: Foundational Data Preprocessing**

The initial phase focused on preparing the data for machine learning applications, which included:

* **Data Cleansing:** Removal of non-predictive attributes such as RowNumber, CustomerId, and Surname.  
* **Feature Encoding:** Transformation of categorical features (Gender, Geography) into a numerical format suitable for algorithmic processing.  
* **Feature Scaling:** Application of StandardScaler to normalize the feature space. This is a critical step to ensure that features with disparate scales do not disproportionately influence model training.

### **Step 2: Mitigating Class Imbalance via Resampling**

To address the challenge of class imbalance, we employed several industry-standard resampling techniques on the training data. This allowed for a robust evaluation of how data balancing impacts each model's ability to learn the characteristics of the minority class. The methods tested include:

* **RandomOverSampler:** A technique that balances class distribution by randomly duplicating instances from the minority class.  
* **SMOTE (Synthetic Minority Over-sampling Technique):** An advanced method that generates new, synthetic minority class instances to create a more balanced and diverse training set.  
* **SMOTEENN:** A hybrid technique that combines SMOTE's over-sampling with a data cleaning method (Edited Nearest Neighbours) to remove potentially noisy samples.

**Demonstrated Impact:** The application of resampling techniques yielded substantial improvements in model performance. For example, the baseline AdaBoost model's recall for the churn class was **0.45**. After applying RandomOverSampler, its recall increased to **0.75**, representing a **66% improvement** in its capacity to identify at-risk customers.

### **Step 3: Comprehensive Hyperparameter Optimization**

A diverse portfolio of machine learning models was evaluated, ranging from traditional algorithms like Logistic Regression to powerful ensemble methods such as Random Forest, AdaBoost, XGBoost, and LightGBM. For each model and sampler combination, we performed extensive hyperparameter tuning.

* **Optimization Metric (F2-Score):** The tuning process was strategically optimized for the **F2-Score**. This metric assigns **twice the importance to recall as it does to precision**, thereby aligning the model optimization process directly with the primary business objective of identifying the maximum number of potential churners.  
* **Efficient Search Strategy:** For computationally intensive models (Random Forest, XGBoost, Decision Tree), RandomizedSearchCV was employed to efficiently search the vast hyperparameter space.

### **Step 4: Final Calibration via Threshold Analysis**

The output of a classification model is a probability score. The final step of our process involved a detailed analysis of the classification threshold. By adjusting this cutoff value, we could fine-tune the model's sensitivity, allowing us to select an optimal operating point that directly aligns with specific business requirements.

**Demonstrated Impact:** This calibration was instrumental in translating model performance into business value. For a top-performing XGBoost configuration, adjusting the threshold from **0.5 to 0.475** increased recall from **0.66 to 0.70**, enabling the identification of **4% more** at-risk customers.

## **4\. Custom Business Metrics for Strategic Model Selection**

To facilitate a business-centric evaluation, three custom metrics were developed. These scores translate standard machine learning metrics into actionable business insights, allowing stakeholders to select a model based on specific financial or strategic constraints.

1. **Your Score (Recall \* Precision \* 100\)**  
   * **Purpose:** A balanced **performance index** that rewards models proficient in both identifying churners (recall) and maintaining accuracy in those predictions (precision).  
   * **Use Case:** This metric is ideal for a **Project Manager or Data Science Lead** requiring a single, holistic measure of a model's overall effectiveness. A high score signifies a robust and reliable model.  
2. **Value Score (Recall \* Precision^2 \* 100\)**  
   * **Purpose:** A **cost-efficiency score** designed to heavily penalize models with low precision. By squaring the precision term, it places a strong emphasis on minimizing resource wastage.  
   * **Use Case:** This metric is tailored for a **Marketing or Finance Manager operating under strict budget constraints**. It prioritizes models that ensure the highest probability of success for each dollar spent on retention efforts.  
3. **Retained per $100 Spent (Precision \* 100\)**  
   * **Purpose:** A direct **Return on Investment (ROI) metric**. It provides a clear answer to the business question: "Assuming a $1 cost per retention action, how many actual churning customers will we successfully engage for every $100 invested?"  
   * **Use Case:** This metric is intended for **executive leadership and financial stakeholders**, as it translates model performance into a tangible financial forecast, simplifying the justification for the program's investment.

## **5\. Comprehensive Results and Flexible Model Selection**

This project culminated in an exhaustive series of experiments. The complete performance results for every model, sampler, hyperparameter set, and threshold tested are documented in the attached **model\_performance\_results.csv** file.

This file is a valuable asset, empowering stakeholders to select a model that aligns precisely with their unique business goals. For instance, a team focused on aggressive market share retention might prioritize a model with the highest recall, whereas a team focused on profitability might select a model with the highest "Value Score." The final code can be easily adapted to deploy any of the top-performing models documented in this file.

## **6\. Final Model Recommendation**

After a thorough evaluation across all technical and business metrics, the final recommended model is:

* **Model:** **XGBoost**  
* **Data Preprocessing:** **SMOTE**  
* **Hyperparameters:**  
  * subsample: 0.6  
  * reg\_lambda: 3  
  * reg\_alpha: 0.1  
  * n\_estimators: 300  
  * max\_depth: 7  
  * learning\_rate: 0.01  
  * gamma: 0.1  
  * colsample\_bytree: 0.9

### **Justification for Selection:**

This specific configuration was selected because it achieved the **highest F1-Score (0.63)** among all viable candidates. The F1-Score, being the harmonic mean of precision and recall, is an industry-standard metric for identifying a model that has found an optimal and effective balance between its constituent metrics.

* **Optimized Balance (F1-Score):** While some configurations achieved marginally higher recall by significantly sacrificing precision (or vice-versa), this model represents the most effective equilibrium, making it a robust choice for a general business application.  
* **High Recall with Strong Precision:** This configuration delivers a strong **Recall of 0.69** and a solid **Precision of 0.57**. In business terms, this means the model can successfully identify **69% of all customers who are genuinely at risk of churning**, while ensuring that **57% of the retention budget is spent on targeting these at-risk individuals**.  
* **Excellent Overall Predictive Capability:** The model's strong performance is further validated by its high **ROC-AUC score of 0.86**, confirming its superior underlying ability to distinguish between churning and non-churning customers.

This final recommended model represents the most balanced and practically applicable solution from our analysis, providing the business with a reliable and efficient tool to proactively mitigate customer churn.
