# MLFromScratch 🤖

A comprehensive implementation of machine learning algorithms from scratch using only NumPy, without relying on high-level ML libraries like TensorFlow, PyTorch, or scikit-learn.

## 🎯 Project Goal

To understand the mathematical foundations of machine learning by implementing algorithms from first principles, focusing on:
- Mathematical derivations and implementations
- Educational clarity and comprehensive explanations
- Validation against industry-standard libraries
- Beautiful visualizations and analysis

## ✨ Current Implementations

### 1. Linear Regression
**File:** `mlr.ipynb`

**Features:**
- **Normal Equation Implementation:** `β = (X^T X)^(-1) X^T y`
- **Comprehensive Model Evaluation:** R², RMSE, MAE metrics
- **Feature Scaling:** Standardization for real-world datasets
- **Train-Test Split:** Proper validation methodology
- **Realistic Dataset:** Student performance prediction with 4 features
- **Visualizations:** 6-panel analysis including residual plots, feature importance
- **Validation:** Perfect match with scikit-learn implementation

**Results:**
- Model explains **42.6%** of score variance on test data
- Average prediction error: **~6.9 points** on exam scores
- Successfully predicts student exam scores based on study habits and academic history

**Mathematical Foundation:**
```
Normal Equation: β = (X^T X)^(-1) X^T y
Model: y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + β₄x₄
Features: StudyHours, PrevGPA, Attendance, SleepHours
```

## 📊 Project Structure

```
MLFromScratch/
├── mlr.ipynb           # Linear Regression implementation
├── pyproject.toml      # Poetry dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file
```

## 🛠️ Tech Stack

- **NumPy** - Core mathematical operations and linear algebra
- **Matplotlib** - Data visualization and plotting
- **Pandas** - Data manipulation and analysis
- **Poetry** - Dependency management and virtual environments
- **Jupyter** - Interactive development and educational notebooks

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mrrahman1517/MLFromScratch.git
cd MLFromScratch
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Activate the environment:**
```bash
poetry shell
```

4. **Launch Jupyter:**
```bash
jupyter notebook mlr.ipynb
```

## 📈 Roadmap

### Upcoming Algorithms
- [ ] **Logistic Regression** - Binary and multiclass classification
- [ ] **Neural Networks** - Feedforward networks with backpropagation
- [ ] **K-Means Clustering** - Unsupervised learning
- [ ] **Decision Trees** - Recursive splitting algorithms
- [ ] **Principal Component Analysis (PCA)** - Dimensionality reduction
- [ ] **Support Vector Machines** - Margin-based classification
- [ ] **Random Forest** - Ensemble methods

### Features to Add
- [ ] Gradient descent implementations
- [ ] Cross-validation techniques
- [ ] Regularization methods (L1/L2)
- [ ] Feature selection algorithms
- [ ] Advanced visualization tools

## 🎓 Educational Approach

Each implementation includes:
- **Mathematical derivations** with step-by-step explanations
- **Code comments** explaining every operation
- **Visualizations** to understand algorithm behavior
- **Comparisons** with established libraries for validation
- **Real-world examples** with practical datasets

## 📝 Key Learning Outcomes

By working through this project, you'll understand:
- The mathematical foundations of ML algorithms
- How to implement algorithms using only basic linear algebra
- Proper model evaluation and validation techniques
- Feature engineering and data preprocessing
- The relationship between theory and practice in machine learning

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new algorithm implementations
- Improve existing code and documentation
- Add more comprehensive examples
- Fix bugs or suggest optimizations

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🌟 Acknowledgments

- Educational inspiration from Andrew Ng's ML courses
- Mathematical foundations from "The Elements of Statistical Learning"
- NumPy and SciPy communities for excellent documentation

---

**Built with ❤️ for learning and understanding machine learning from the ground up**