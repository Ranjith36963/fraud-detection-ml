# Machine Learning Techniques for Online Payment System Fraud Detection

**Master's Dissertation Project | University of Hertfordshire**

Comprehensive fraud detection system achieving 99.94% accuracy on 6.3+ million financial transactions using ensemble machine learning techniques.

---

## Project Overview

This research project addresses the critical challenge of detecting fraudulent transactions in online payment systems, where fraud represents only 0.17% of all transactions (highly imbalanced dataset). The system was developed as part of my MSc in Artificial Intelligence and Robotics at the University of Hertfordshire.

**Key Achievement**: Random Forest model achieved **99.94% accuracy** with 99.89% precision and 99.99% recall on a dataset of 6,362,620 transactions.

---

## Problem Statement

Online payment fraud poses significant financial risks to consumers and businesses globally. Traditional rule-based systems struggle with:
- Severe class imbalance (only 0.17% fraudulent transactions)
- Evolving fraud patterns
- High false positive rates
- Real-time processing requirements

This project develops an advanced machine learning system to accurately identify fraudulent transactions while minimizing false positives and false negatives.

---

## Dataset

**Source**: [Kaggle PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

**Size**: 6,362,620 transactions  
**Features**: 11 attributes  
**Class Distribution**: 
- Legitimate transactions: 99.83%
- Fraudulent transactions: 0.17% (severe imbalance)

**Attributes**:
- `step`: Time unit (1 hour)
- `type`: Transaction type (CASH-OUT, PAYMENT, CASH-IN, TRANSFER, DEBIT)
- `amount`: Transaction amount
- `nameOrig`: Customer initiating transaction
- `oldbalanceOrg`: Initial balance before transaction
- `newbalanceOrig`: New balance after transaction
- `nameDest`: Recipient of transaction
- `oldbalanceDest`: Initial recipient balance
- `newbalanceDest`: New recipient balance
- `isFraud`: Target variable (0 = legitimate, 1 = fraud)
- `isFlaggedFraud`: Business rule flagging

---

## Methodology

### Data Preprocessing
1. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of transaction types
   - Correlation analysis between features
   - Feature importance evaluation
   - Visualization of fraud patterns

2. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Addressed 0.17% fraud rate challenge
   - Balanced training data while maintaining test set distribution

3. **Feature Engineering**
   - Selected most predictive features based on importance analysis
   - Removed redundant features
   - Scaled numerical features

### Machine Learning Models Tested

Five state-of-the-art algorithms were evaluated:

1. **Random Forest** (Best performer)
2. **Gradient Boosting**
3. **Multi-Layer Perceptron (MLP)**
4. **Artificial Neural Network (ANN)**
5. **AdaBoost**

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **99.94%** | **99.89%** | **99.99%** | **99.94%** |
| Gradient Boosting | 98.88% | 98.25% | 99.53% | 98.88% |
| Multi-Layer Perceptron | 98.31% | 98.49% | 98.13% | 98.31% |
| AdaBoost | 96.61% | 95.67% | 97.64% | 96.64% |
| Artificial Neural Network | 95.99% | 95.99% | 96.00% | 95.99% |

### Random Forest - Detailed Results

**Best Model Performance**:
- **Accuracy**: 99.94%
- **Precision**: 99.89% (minimal false positives)
- **Recall**: 99.99% (virtually all fraud detected)
- **F1-Score**: 99.94% (excellent balance)

**Confusion Matrix**:
- True Negatives: Correctly identified legitimate transactions
- True Positives: 22 fraudulent transactions correctly detected
- False Positives: 3 (legitimate flagged as fraud)
- False Negatives: Minimal (near-perfect fraud detection)

**Key Insights**:
- Random Forest significantly outperformed other algorithms
- Ensemble approach proved most effective for imbalanced data
- 18% accuracy improvement achieved through hyperparameter tuning
- Model demonstrated excellent generalization on test data

---

## Technical Implementation

### Technologies Used
- **Language**: Python 3.x
- **ML Frameworks**: 
  - Scikit-learn (Random Forest, AdaBoost, Gradient Boosting)
  - TensorFlow/Keras (Neural Networks)
  - XGBoost (Gradient Boosting variants)
- **Data Processing**: 
  - Pandas (data manipulation)
  - NumPy (numerical operations)
- **Imbalance Handling**: 
  - SMOTE (imbalanced-learn)
- **Visualization**: 
  - Matplotlib
  - Seaborn
- **Model Interpretability**: 
  - SHAP (SHapley Additive exPlanations)
  - Feature importance analysis

### Key Techniques
1. **SMOTE for Class Imbalance**: Synthetic sample generation for minority class
2. **Ensemble Methods**: Random Forest decision trees for robust predictions
3. **Hyperparameter Tuning**: Grid search optimization
4. **Cross-Validation**: K-fold validation for model reliability
5. **Feature Selection**: Importance-based feature engineering

---

## Project Structure

```
fraud-detection-ml/
├── data/
│   └── README.md              # Dataset information and download instructions
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb # Data preprocessing and SMOTE
│   ├── 03_Model_Training.ipynb # Model training and evaluation
│   └── 04_Results_Analysis.ipynb # Results visualization
├── src/
│   ├── fraud_detection.py    # Main implementation
│   ├── preprocessing.py      # Data preprocessing utilities
│   ├── models.py            # Model definitions
│   └── evaluation.py        # Evaluation metrics
├── results/
│   ├── figures/             # Visualizations and plots
│   └── metrics/             # Performance metrics
├── thesis.pdf               # Complete master's dissertation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Ranjith36963/fraud-detection-ml.git
cd fraud-detection-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Place in `data/` directory

### Running the Code

**Option 1: Jupyter Notebooks (Recommended)**
```bash
jupyter notebook
# Navigate to notebooks/ directory
# Run notebooks in sequence: 01 → 02 → 03 → 04
```

**Option 2: Python Script**
```bash
python src/fraud_detection.py
```

### Training Models
```python
from src.fraud_detection import FraudDetector

# Initialize detector
detector = FraudDetector()

# Load and preprocess data
detector.load_data('data/paysim.csv')
detector.preprocess()

# Train Random Forest (best model)
detector.train_random_forest()

# Evaluate
results = detector.evaluate()
print(f"Accuracy: {results['accuracy']:.4f}")
```

---

## Research Contributions

1. **Comprehensive Model Comparison**: Evaluated 5 ML algorithms on large-scale financial data
2. **Class Imbalance Solution**: Successfully applied SMOTE to extreme imbalance (0.17% fraud)
3. **High Accuracy Achievement**: 99.94% accuracy on 6.3M+ transactions
4. **Model Interpretability**: SHAP analysis for explainable fraud detection
5. **Production-Ready Insights**: Real-time inference capability demonstrated

---

## Key Findings

### What Worked
- **Random Forest**: Best overall performance due to ensemble approach
- **SMOTE**: Essential for handling severe class imbalance
- **Feature Engineering**: Feature importance analysis critical for performance
- **Hyperparameter Tuning**: 18% accuracy improvement achieved

### Challenges Addressed
- Extreme class imbalance (0.17% fraud rate)
- High computational cost with large dataset
- False positive minimization for business viability
- Real-time prediction requirements

### Feature Importance
Top predictive features identified:
1. `step` (transaction timing)
2. `oldbalanceOrg` (original balance)
3. `newbalanceOrig` (new balance)
4. `type` (transaction type)
5. `amount` (transaction amount)

---

## Future Work

1. **Deep Learning Enhancement**: 
   - LSTM networks for sequential pattern detection
   - Attention mechanisms for temporal analysis

2. **Real-Time Deployment**:
   - Stream processing pipeline
   - Low-latency prediction API
   - Model serving infrastructure

3. **Advanced Techniques**:
   - Graph neural networks for transaction networks
   - Anomaly detection algorithms
   - Federated learning for privacy-preserving fraud detection

4. **Feature Expansion**:
   - Behavioral biometrics
   - Device fingerprinting
   - Geolocation patterns

---

## Academic Context

**Degree**: MSc Artificial Intelligence and Robotics with Advanced Research  
**University**: University of Hertfordshire  
**Course Code**: 7COM1039 - Advanced Computer Science Masters Project  
**Submission Date**: December 4, 2023  
**Supervisor**: Imran Khan  
**Student ID**: 20063331  

**Thesis**: Complete dissertation available in `thesis.pdf`

---

## Dependencies

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
tensorflow>=2.12.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
jupyter>=1.0.0
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{guruprakash2023fraud,
  title={Machine Learning Techniques for Online Payment System Fraud Detection},
  author={Guruprakash, Ranjith Maliga},
  year={2023},
  school={University of Hertfordshire},
  type={MSc Dissertation},
  note={99.94\% accuracy on 6.3M+ transactions}
}
```

---

## Results Visualization

Key visualizations included in the project:

1. **Class Distribution**: Countplot showing imbalance
2. **Feature Correlations**: Heatmap of feature relationships
3. **Feature Importance**: Bar chart of top predictive features
4. **Model Comparison**: Performance metrics across all models
5. **Confusion Matrices**: Detailed classification results
6. **ROC Curves**: Model discrimination capability

---

## Limitations

1. **Dataset Scope**: Synthetic data may not capture all real-world fraud patterns
2. **Temporal Dynamics**: Model trained on static dataset; fraud evolves
3. **Generalization**: Performance on other payment systems may vary
4. **Computational Cost**: Large dataset requires significant resources
5. **Interpretability vs Performance**: Best model (Random Forest) less interpretable than simpler models

---

## Acknowledgments

- **Supervisor**: Imran Khan for invaluable guidance and mentorship
- **University of Hertfordshire**: Resources and facilities
- **Kaggle Community**: For providing the PaySim dataset
- **Research Community**: Prior work in fraud detection that informed this study

---

## License

This project is part of academic research completed at the University of Hertfordshire. The code and methodology are shared for educational and research purposes.

For academic use, please cite the work. For commercial applications, please contact the author.

---

## Contact

**Ranjith Maliga Guruprakash**  
AI/ML Engineer | Generative AI Specialist

- Email: rahulranjith369@gmail.com
- LinkedIn: [linkedin.com/in/ranjith369](https://linkedin.com/in/ranjith369)
- GitHub: [github.com/Ranjith36963](https://github.com/Ranjith36963)
- Portfolio: [ranjith36963.github.io](https://ranjith36963.github.io)

---

## Related Projects

Check out my other AI/ML projects:
- [Multimodal RAG System](https://ranjith36963.github.io#projects) - Production GenAI at Calnex Solutions
- [Cloudflare AI Worker Wizard](https://github.com/Ranjith36963/cf_ai_worker_wizard) - Edge AI code generation
- [Diner Growth Engine](https://diner-growth-engine.lovable.app) - Full-stack SaaS with AI

---

**Status**: ✅ Research Complete | Published December 2023 | 99.94% Accuracy Achieved

*Built with passion for advancing AI-powered fraud detection systems.*
