# 💼 Task 3: Credit Risk Analysis  
### J.P. Morgan Quantitative Research Virtual Experience

---

## 🧩 Overview
This project is part of the **J.P. Morgan Quantitative Research Virtual Experience Program** on *The Forage*.  
The goal of **Task 3** is to develop a predictive model that estimates the **Probability of Default (PD)** for loan borrowers and calculates the **Expected Loss (EL)** assuming a **10% recovery rate**.

The model uses **Logistic Regression**, the same method outlined in the official JPMC example answer, enhanced with:
- 📊 Confusion Matrix visualization for model evaluation  
- 💡 Two sample borrower predictions to demonstrate usage  

---

## 🎯 Objective
Predict whether a borrower will default on a personal loan and estimate the **expected financial loss** to the bank.

**Expected Loss (EL)** = `PD × Exposure at Default (EAD) × (1 – Recovery Rate)`  
*(where Recovery Rate = 10 %)*

---

## 📊 Dataset
**File:** `Task 3 and 4_Loan_Data.csv`

Each row represents an individual borrower profile with the following fields:

| Feature | Description |
|----------|--------------|
| `credit_lines_outstanding` | Number of open credit lines |
| `loan_amt_outstanding` | Current outstanding loan amount |
| `total_debt_outstanding` | Total debt held by the borrower |
| `income` | Annual income |
| `years_employed` | Years of employment |
| `fico_score` | Credit score |
| `default` | Target variable (1 = Default, 0 = Non-default) |

**Derived Ratios:**
- `payment_to_income` = loan_amt_outstanding / income  
- `debt_to_income` = total_debt_outstanding / income  

---

## ⚙️ Methodology

1. **Feature Engineering**
   - Created two key financial ratios: `payment_to_income` and `debt_to_income`.

2. **Model Training**
   - Used `LogisticRegression` (solver = `liblinear`, max_iter = 10000).
   - Trained on borrower data to estimate the **Probability of Default** (PD).

3. **Evaluation**
   - Calculated **AUC** and **error rate**.  
   - Visualized performance with a **confusion matrix heatmap**.

4. **Expected Loss Function**
   - Defined a function to compute **PD** and **Expected Loss (EL)** for new borrowers:
     ```python
     Expected Loss = PD × Loan Amount × (1 - Recovery Rate)
     ```

---

## 📈 Model Evaluation

| Metric | Description |
|---------|-------------|
| **AUC Score** | Measures model’s ability to distinguish defaults from non-defaults |
| **Error Rate** | Proportion of incorrect predictions |
| **Confusion Matrix** | Distribution of true/false positives and negatives |

A confusion matrix heatmap was used for better visualization.

---

## 💰 Example Predictions

**Example 1**
=== Example 1 ===

Probability of Default: 37.51%

Expected Loss: $8,439.54

**Example 2**
=== Example 2 ===

Probability of Default: 100.00%

Expected Loss: $36,000.00


---

## 🧾 Dependencies
Install all required libraries before running:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
