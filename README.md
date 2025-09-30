# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
🚀 Predictive Analysis using Machine Learning – Iris Dataset

This project is part of my CODTECH Internship (Task-2).
The goal is to build a predictive machine learning model that can classify Iris flowers into species based on their features.

📌 Project Overview

Dataset: Iris Dataset
 (150 samples, 4 features).

Problem Type: Classification.

Objective: Predict the species of Iris flower (Setosa, Versicolor, Virginica) from input features:

Sepal Length

Sepal Width

Petal Length

Petal Width

⚙️ Steps Performed

Data Loading – Loaded the Iris dataset from sklearn.datasets.

Feature Selection – Used all 4 numerical features for classification.

Data Preprocessing – Train-test split and feature scaling using StandardScaler.

Model Training – Applied Logistic Regression for classification.

Evaluation – Measured performance using:

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix (visualized with Seaborn heatmap).

📊 Results

Model Accuracy: ~95% (may vary slightly due to random state).

Confusion Matrix shows high classification performance across all three classes.

🛠️ Tech Stack

Language: Python

Libraries:

pandas, numpy → Data handling

scikit-learn → ML model, preprocessing, evaluation

matplotlib, seaborn → Data visualization

📂 Project Structure
├── README.md           # Project description  
├── iris_prediction.ipynb  # Jupyter Notebook (code + results)  
└── requirements.txt    # Python dependencies  

📥 Installation & Usage

Clone the repository:

git clone https://github.com/your-username/iris-ml-predictive-analysis.git
cd iris-ml-predictive-analysis


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook iris_prediction.ipynb


<img width="530" height="455" alt="Image" src="https://github.com/user-attachments/assets/ac95fc82-13f4-43d9-b8c5-f9309d9c10d8" />

