# Predicting Used Car Value Retention Ratio Using Multi-Source Data and Machine Learning

## Overview

Build a machine learning pipeline to predict the **value retention ratio** of used vehicles and identify the factors that most strongly influence vehicle depreciation.

Students will collect vehicle listings from multiple online sources, integrate the data into a unified dataset, engineer features, train regression models, and analyze feature importance.

```
Value retention ratio = current listing price / original MSRP
```

---

## 1. Data Collection

- Collect used-car listing data from at least two independent online sources (e.g., Craigslist, Facebook Marketplace, Cars.com) through web scraping.
- Collect MSRP for target makes and models.

**Minimum Requirements**

- [ ] Collect data from at least **2 sources**.
- [ ] Collect MSRP information for each vehicle make, model, and model year from a reliable source and merge it with the used-car dataset.
- [ ] Obtain at least **1,000 vehicle listings** after preprocessing and duplicate removal.
- [ ] Each data source must contribute at least **500 listings** before integration.
- [ ] Include **3–5 vehicle makes**.
- [ ] Include **3–5 models** for each selected make.
- [ ] The integrated dataset must contain at least **10 original attributes** collected from the sources (excluding engineered features).

## 2. Data Preprocessing and Integration

Combine records from multiple sources into a unified dataset by resolving inconsistent attribute names, formats, and units across sources.

**Minimum Requirements**

- [ ] Explore statistics of the collected data attributes.

## 3. Feature Engineering

Create meaningful features that may improve prediction accuracy, such as vehicle age.

**Minimum Requirements**

- [ ] Extract at least **2 meaningful features** from listing descriptions (e.g., one owner).
- [ ] Apply and discuss at least **one encoding technique** for high-cardinality features.

## 4. Model Development

Train and compare at least **3 different regression algorithms**.

**Minimum Requirements**

- [ ] Use an appropriate train/test split or cross-validation strategy.
- [ ] Evaluate models using at least **2 regression metrics** (e.g., MAE, RMSE, R²).

## 5. Depreciation Analysis and Feature Ranking

Discuss how vehicle age, mileage, and other important features contribute to vehicle depreciation, using methods such as regression coefficients or feature importance.

## 6. Visualization

**Minimum Requirements**

- [ ] Visualization illustrating data characteristics (e.g., price distribution, make/model frequencies, mileage distribution).
- [ ] Visualization illustrating feature ranking.
- [ ] Visualization illustrating performance comparison across trained models.

## 7. Discussion and Conclusion

- Most important factors affecting vehicle value
- Major drivers of depreciation
- Challenges in data collection and integration
- Limitations of the dataset and models

---

## Deliverables

| # | Deliverable | Weight |
|---|-------------|--------|
| 1 | GitHub Repository | 10% |
| 2 | Code Quality | 20% |
| 3 | Data | 20% |
| 4 | Final Report | 50% |

### Data (20%)

- Initially parsed data from each source (2 data files if data is collected from 2 sources)
- Preprocessed and integrated data

### Final Report (50%)

| Section | Weight |
|---------|--------|
| Project Description | 5% |
| Data Collection with discussion | 5% |
| Preprocessing and Integration with discussion | 5% |
| Feature Engineering with discussion | 5% |
| Model Development with discussion | 5% |
| Evaluation Results with discussion | 5% |
| Depreciation Analysis | 5% |
| Visualizations with discussion | 5% |
| Conclusion | 6% |
| AI Usage Statement | 4% |

**Conclusion** should cover:
- Summary of project findings
- What you learned through the project
- Challenges

**AI Usage Statement** should clearly describe which AI tools were used and which portions of the work were completed independently.
