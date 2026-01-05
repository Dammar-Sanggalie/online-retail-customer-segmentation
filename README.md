# Online Retail Customer Segmentation & Temporal Behavior Analysis

## Project Overview
This project focuses on **customer segmentation and behavioral analysis**
for an online retail business using transactional data.

The main objective is to move away from inefficient **mass marketing**
towards **data-driven, segment-based marketing strategies** by:
- Grouping customers based on purchasing behavior (RFM)
- Understanding *when* different customer segments are most active
- Translating analytical results into actionable business recommendations

The project follows a complete **data science lifecycle**:
from raw transaction logs to customer-level insights and visual storytelling.

---

## Business Problem
The retailer owns a large volume of transaction data but applies
the same promotions to all customers.

This results in:
- High marketing costs
- Low conversion efficiency
- Missed opportunities to reward high-value customers
- Weak reactivation strategies for inactive customers

---

## Dataset
**Online Retail II Dataset**

Transactional records containing:
- Invoice
- Product information
- Quantity
- Price
- Invoice date & time
- Customer ID
- Country

The dataset records **transactions**, not customer profiles.
Therefore, customer-level features must be engineered.

---

## Methodology

### 1. Data Cleaning
- Removed transactions with missing CustomerID
- Excluded canceled invoices
- Removed non-positive quantities and prices
- Created transaction-level revenue

Result:
- ~779k clean transactions
- ~5,878 unique customers

---

### 2. Feature Engineering (RFM)
Customer behavior is summarized using **RFM metrics**:
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spending

Due to heavy skew and outliers:
- Log transformation applied
- Standard scaling performed before clustering

---

### 3. Customer Segmentation
- Algorithm: **K-Means Clustering**
- Cluster selection evaluated using:
  - Elbow Method
  - Silhouette Score
- Final choice: **3 clusters** for business interpretability

---

### 4. Cluster Profiling
Clusters were profiled using median RFM values and interpreted as:

| Cluster | Description | Characteristics |
|-------|------------|----------------|
| 0 | Hibernating | Long inactivity, low frequency & spend |
| 1 | Champions | Recent, frequent, high spending |
| 2 | Potential Loyalists | Moderate activity and value |

---

### 5. Temporal Behavior Analysis
To answer *“When is the best time to engage each segment?”*, we analyzed:
- Day of week
- Hour of day
- Month period (Early / Mid / Late)

Metrics:
- Transaction counts
- **Unique active customers** (primary focus)

---

### 6. Visualization & Reporting
Generated visual assets include:
- Cluster size distribution
- RFM profile comparison
- Hour-of-day activity heatmap
- Day-of-week × hour heatmap
- Month-period engagement patterns

All visuals are saved under:
reports/figures/

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- VS Code

---

## Project Structure
online-retail-segmentation/
├── data/
│ ├── raw/
│ ├── interim/
│ └── processed/
├── src/
│ ├── data_cleaning.py
│ ├── rfm.py
│ ├── rfm_features.py
│ ├── kmeans_clustering.py
│ ├── cluster_profiling.py
│ ├── temporal_analysis.py
│ └── visualization.py
├── reports/
│ └── figures/
├── insights.md
├── README.md
└── requirements.txt


---

## Key Takeaways
- RFM-based segmentation reveals clear customer value tiers
- Temporal analysis shows **consistent activity peaks around midday**
- Unique customer analysis avoids bias from power buyers
- Marketing timing can be optimized per segment

---

## Author
**Dammar Sanggalie**  
Machine Learning & Data Science Enthusiast
