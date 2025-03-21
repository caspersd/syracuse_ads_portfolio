# Detecting Fishing Activity in Trawling Vessels Using AIS and Machine Learning

## Project Overview

Illegal, Unreported, and Unregulated (IUU) fishing poses a significant threat to marine ecosystems, depleting fish stocks and causing billions in economic losses annually. This project aims to detect fishing activity using Automatic Identification System (AIS) data from Global Fishing Watch, leveraging machine learning techniques to classify vessel activities.

By analyzing geospatial vessel movements, we classify whether a vessel is actively engaged in fishing or in transit. The project evaluates multiple machine learning models, including Random Forest, Support Vector Machines (SVM), and unsupervised clustering methods (K-Means and DBSCAN), to determine the most effective approach for identifying fishing activity.

## Dataset

- **Source**: Global Fishing Watch (AIS data)
- **Observations**: 4,369,101 AIS signals from 49 trawling vessels
- **Key Features**:
  - **MMSI**: Unique vessel identifier
  - **Timestamp**: Time of AIS transmission
  - **Geolocation**: Latitude & Longitude
  - **Speed & Course**: Vessel movement indicators
  - **Distance from Shore/Port**: Proximity analysis
  - **Fishing Status**: Labels indicating fishing activity

## Methodology

### **Data Processing**
- Removal of sensor errors and outliers
- Handling missing values
- Introduction of lag variables to capture temporal patterns
- Stratification by time for training and test splits to avoid data leakage

### **Model Evaluation**
1. **Clustering Methods**
   - **K-Means**: Found patterns in fishing vs. transit behaviors but struggled with overlapping activities.
   - **DBSCAN**: Performed well in exploratory analysis but misclassified fishing as noise in some cases.
   
2. **Supervised Learning**
   - **Random Forest (RF)**: Achieved **88% accuracy**, with **speed** and **distance from shore** being the most influential features.
   - **Support Vector Machines (SVM)**:
     - **Linear SVM**: Underperformed due to difficulty in defining a clear decision boundary.
     - **Polynomial SVM**: Improved feature representation but prone to overfitting.
     - **RBF SVM**: Performed better than other SVM models but was computationally intensive.

### **Key Findings**
- **Temporal vessel behavior is critical for classification**, with lag variables significantly improving accuracy.
- **Random Forest with lag variables outperformed other models**, achieving 88% accuracy.
- **Speed and distance from shore were the strongest predictors** of fishing activity.
- **Clustering methods provided useful exploratory insights** but were less effective for classification.

## Challenges & Future Improvements
- **Distinguishing fishing from transit** remains challenging due to overlapping movement patterns.
- **AIS sensor errors and missing data** require rigorous cleaning.
- **Temporal modeling** using LSTM or RNN could further enhance classification accuracy.
- **Real-time AIS feed integration** could enable proactive monitoring of potential IUU fishing activities.

## Repository Structure
```
├──  IUUF_Detection_Final_Presentation.pdf # Presentation slides summarizing findings 
├──  IUUF_Detection_Final_Report.pdf # Detailed project report 
├──  Fishing_Activity_Detection_Full_Code.ipynb # Jupyter Notebook with implementation 
├──  README.md # Project documentation
```
## How to Use
1. Clone the repository:
   ```bash
   git clone git clone https://github.com/caspersd/Illegal_Fishing_Detection.git
   cd Illegal_Fishing_Detection```

2. Open Fishing_Activity_Detection_Full_Code.ipynb in Jupyter Notebook to explore the code and experiment with models.


## Acknowledgments
This project was made possible using data from [Global Fishing Watch](https://globalfishingwatch.org/) and was conducted as part of a data science research initiative.
