# 📌 Ecommerce Customer Spending Analysis

## 📖 Project Overview

This project analyzes customer behavior for an online clothing store that offers in-store styling sessions and allows purchases through a **mobile app** and **website**.

The main objective is:

> To determine whether the company should focus more on improving the mobile app experience or the website experience.

We use **Linear Regression** to model customer yearly spending based on behavioral features.

---

## 📊 Dataset Description

The dataset contains the following features:

- **Avg. Session Length** – Average in-store session duration  
- **Time on App** – Time spent on mobile app  
- **Time on Website** – Time spent on website  
- **Length of Membership** – Number of years as a customer  
- **Yearly Amount Spent** – Target variable (annual spending)

---

## 🧠 Problem Statement

Build a regression model to predict:

> Yearly Amount Spent

And analyze which features influence spending the most.

---

## ⚙️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🔎 Exploratory Data Analysis (EDA)

The following visualizations were performed:

- Pairplot → Feature relationships  
- Heatmap → Correlation matrix  
- Distribution plots → Check data distribution  
- Scatter plots → Actual vs Predicted  
- Residual plot → Assumption checking  

---

## 🤖 Model Used

**Linear Regression**

Steps followed:

1. Train-test split  
2. Model fitting  
3. Prediction  
4. Evaluation using:
   - Mean Squared Error (MSE)
   - R² Score

---

## 📈 Model Performance

- High R² score (~0.97)
- Low MSE
- Residuals approximately normally distributed
- Strong alignment between actual and predicted values

This indicates a strong linear relationship between features and the target variable.

---

## 💡 Key Insights

- **Time on App** shows a stronger impact on yearly spending compared to Time on Website.
- Length of Membership is a significant predictor of spending.
- The model explains most of the variance in customer spending.

---

## 🏢 Business Recommendation

Based on regression analysis:

> The company should prioritize improving the mobile app experience, as it has a stronger relationship with customer spending compared to the website as seen by the coefficient values.


## 📂 Project Structure

```
Ecommerce_Regression_Analysis.ipynb
README.md
dataset.csv
```

---