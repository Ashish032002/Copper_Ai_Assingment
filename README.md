# Cancer Prediction Analysis 🔬 🏥

## Overview 📋
This project implements machine learning models to predict cancer diagnosis based on various clinical features. Using a comprehensive dataset of cancer measurements, we've developed and compared multiple algorithms to achieve high-accuracy predictions.

## Features ⭐
- Multiple Machine Learning Models Implementation
- Advanced Data Preprocessing
- Model Performance Comparison
- Feature Importance Analysis
- Hyperparameter Tuning
- Model Persistence

## Technologies Used 🛠️
- Python 3.8+
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## Models Implemented 🤖
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Neural Network (MLP)

## Installation 💻
1. Clone the repository
```bash
git clone https://github.com/Ashish032002/Copper_Ai_Assingment.git
cd Copper_Ai_Assingment
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Dataset 📊
The dataset contains the following features:
- Radius mean
- Texture mean
- Perimeter mean
- Area mean
- Smoothness mean
- And many more clinical measurements

Target variable: Cancer diagnosis (M = Malignant, B = Benign)

## Usage 🚀
1. Place your data file (Cancer_Data.csv) in the project directory
2. Run the Jupyter notebook
```bash
jupyter notebook Cancer_Prediction_Analysis.ipynb
```

3. To use the saved model for predictions:
```python
import joblib

# Load the model
model = joblib.load('best_cancer_prediction_model.joblib')

# Make predictions
predictions = model.predict(your_data)
```

## Results 📈
- Model performance comparison
- Feature importance visualization
- Confusion matrices for each model
- Detailed classification reports

## Model Performance 🎯
| Model | Accuracy |
|-------|----------|
| Random Forest | ~96% |
| SVM | ~95% |
| Neural Network | ~94% |
| KNN | ~93% |


## Future Improvements 🔮
- [ ] Implement more advanced models
- [ ] Add cross-validation strategies
- [ ] Create web interface for predictions
- [ ] Add more visualization options
- [ ] Implement ensemble methods

