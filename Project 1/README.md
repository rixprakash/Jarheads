# 🎵 What Makes a Genre

## 📌 Overview
This project analyzes song lyrics using NLP and machine learning to classify genres, predict success, and explore lyrical evolution over time. We use models such as Logistic Regression, SVM, and DistilBERT to identify trends and patterns in song lyrics.

## 🛠️ Software & Platforms
- **Python** (Pandas, NumPy, Scikit-Learn, TensorFlow, Matplotlib, Seaborn)
- **Jupyter Notebook**
- **Google Colab** (for cloud-based computation)
- **GitHub** (for version control and collaboration)
- **Operating System:** Windows / Mac / Linux

## 📁 Project Structure
```
📂 Project 1
│── 📄 README.md                # Project documentation
│── 📄 LICENSE.md               # MIT License
│── 📄 requirements.txt         # List of project dependencies for easy installation
│
│── 📂 SCRIPTS/                 # All scripts for analysis
│   │── preprocess_data.py       # Data cleaning and preprocessing
│   │── extract_embeddings.py    # Extract BERT embeddings from lyrics
│   │── train_model.py           # Train machine learning models (LogReg, NN)
│   │── predict_genre.py         # Predict genre using trained models
│   │── utils.py                 # Helper functions
│   │── visualization.ipynb      # Data visualization and analysis
│   │── bert_based_classification.ipynb  # Jupyter notebook for BERT classification
│   │── script.ipynb             # Miscellaneous scripts
│
│── 📂 DATA/                    # Data files
│   │── tcc_ceds_music.csv       # Original dataset
│   │── processed_data.csv       # Cleaned dataset ready for analysis
│
│── 📂 OUTPUT/                   # Generated results
│   │── 📂 confusion matrices using BERT/  # Confusion matrices for BERT-based models
│   │   │── LR_confusion_matrix_BERT.png  # Logistic Regression (BERT) confusion matrix
│   │   │── NN_confusion_matrix_BERT.png  # Neural Network (BERT) confusion matrix
│   │
│   │── 📂 confusion matrices using TF-IDF/  # Confusion matrices for TF-IDF models
│   │   │── LR_confusion_matrix_TF-IDF.png  # Logistic Regression (TF-IDF) confusion matrix
│   │   │── NN_confusion_matrix_TF-IDF.png  # Neural Network (TF-IDF) confusion matrix
│   │
│   │── PCA_visualization.png               # PCA visualization of embeddings
│   │── X_embeddings.npy                     # Saved BERT embeddings
│   │── genre_classification_report_using_BERT.pdf   # Analysis report (BERT)
│   │── genre_classification_report_using_TF-IDF.pdf # Analysis report (TF-IDF)
│   │── label_encoder.pkl                    # Label encoder for genre classification
│   │── logistic_regression_bert.pkl         # Trained logistic regression model (BERT)
│   │── neural_network_bert.h5               # Trained neural network model (BERT)
│   │── tfidf_vectorizer.pkl                 # TF-IDF vectorizer for text features
│   │── y.npy                                # Encoded genre labels

```

## 🔄 Reproducibility
To reproduce results:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rixprakash/Jarheads.git
   cd "Project 1"
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run preprocessing script:**
   ```bash
   python SCRIPTS/preprocess_data.py
   ```
4. **Train the model:**
   ```bash
   python SCRIPTS/model_training.py
   ```
5. **Run visualizations and analysis:**
   ```bash
   jupyter notebook SCRIPTS/visualization.ipynb
   ```
6. **View results in the OUTPUT folder.**

## 📜 License
This project is licensed under the MIT License. See the `LICENSE.md` file for details.

## 📚 References
- Research papers on NLP and song lyric analysis
- Machine Learning documentation (Scikit-Learn, TensorFlow)
- GitHub repositories on music genre classification

