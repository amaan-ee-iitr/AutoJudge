# ⚡ AutoJudge Pro

An AI-powered machine learning application that predicts the difficulty rating of programming problems based on their descriptions. The system uses natural language processing and ensemble learning to classify problems as Easy, Medium, or Hard, and predicts their numerical rating (800-3500).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Demo video link
https://youtu.be/QrpEH7nq5lg
## ✨ Features

- 🎯 **Dual Prediction System**: Classifies problems into difficulty categories (Easy/Medium/Hard) and predicts numerical ratings (800-3500)
- 🔮 **Advanced ML Models**: Uses Random Forest ensemble methods for robust predictions
- 🎨 **Modern Web Interface**: Beautiful Streamlit app with glassmorphism design and animated backgrounds
- 📊 **TF-IDF Vectorization**: Extracts meaningful features from problem descriptions using Term Frequency-Inverse Document Frequency
- ⚖️ **SMOTE Balancing**: Handles class imbalance in training data using Synthetic Minority Oversampling Technique
- 📈 **Performance Metrics**: Includes confusion matrices and regression plots for model evaluation
- 🚀 **Easy to Use**: Simple web interface where you paste problem descriptions and get instant predictions

## 🖥️ Preview

The application features a modern UI with:
- Animated gradient background
- Glassmorphism design elements
- Interactive difficulty gauge visualization
- Real-time predictions with visual feedback

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amaan-ee-iitr/AutoJudge.git
   cd AutoJudge
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Format

Before training, ensure you have a `problems.jsonl` file with the following structure (JSONL format - one JSON object per line):

```json
{
  "title": "Problem Title",
  "description": "Problem description text...",
  "input_description": "Input specification...",
  "output_description": "Output specification...",
  "problem_class": "Easy"  // or "Medium" or "Hard",
   "problem_score": "7" // 0 to 10
}
```

## 🔧 Usage

### Training the Models

1. **Prepare your data**: Place your `problems.jsonl` file in the `AutoJudge` directory

2. **Train the models**:
   ```bash
   python train_model.py
   ```

   This will:
   - Load and preprocess the data
   - Apply TF-IDF vectorization
   - Balance the dataset using SMOTE
   - Train both classification and regression models
   - Generate evaluation metrics and visualizations
   - Save the trained models (`classifier.pkl`, `regressor.pkl`, `tfidf.pkl`, `scaler.pkl`)

3. **Check the outputs**:
   - `confusion_matrix.png`: Classification performance visualization
   - `regression_plot.png`: Regression model evaluation
   - Model files (`.pkl`) for inference

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**: The app will automatically open at `http://localhost:8501`

3. **Use the interface**:
   - Enter the problem title (optional)
   - Paste the problem statement/description
   - Optionally add input and output specifications
   - Click "🚀 Analyze Difficulty" to get predictions
   - View the predicted difficulty category and numerical rating

## 📁 Project Structure

```
AutoJudge/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── problems.jsonl         # Training data (JSONL format)
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── classifier.pkl        # Trained classification model (generated)
├── regressor.pkl         # Trained regression model (generated)
├── tfidf.pkl            # TF-IDF vectorizer (generated)
├── scaler.pkl           # Feature scaler (generated)
│
├── confusion_matrix.png  # Classification evaluation (generated)
├── regression_plot.png   # Regression evaluation (generated)
│
└── venv/                # Virtual environment (not tracked)
```

## 🔬 Model Details

### Architecture

- **Text Processing**: 
  - HTML tag removal
  - Lowercasing and normalization
  - TF-IDF vectorization (max 1000 features)

- **Feature Engineering**:
  - Text vectors from TF-IDF
  - Word count features (scaled using MinMaxScaler)

- **Models**:
  - **Classifier**: Random Forest (100 estimators) for difficulty category prediction
  - **Regressor**: Random Forest (100 estimators) for numerical rating prediction

- **Data Handling**:
  - Train/test split (80/20)
  - SMOTE oversampling for balanced classification training
  - Ratings mapped: Easy (800-1000), Medium (1100-1500), Hard (1600-3500)

### Performance Metrics

The training script outputs:
- Classification accuracy on test set
- Mean Absolute Error (MAE) for regression
- Confusion matrix visualization
- Regression evaluation plots

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library (Random Forest, TF-IDF, preprocessing)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **imbalanced-learn**: SMOTE for class balancing
- **joblib**: Model serialization

## 📝 Requirements

The project requires the following packages (see `requirements.txt`):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- plotly
- imbalanced-learn (for SMOTE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [scikit-learn](https://scikit-learn.org/)
- Uses [SMOTE](https://imbalanced-learn.org/) for data balancing

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ If you find this project helpful, consider giving it a star!
