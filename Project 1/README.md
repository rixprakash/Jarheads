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
3. **Preprocess Data:**
   ```bash
   python SCRIPTS/preprocess_data.py
   ```
4. **Extract BERT embeddings (This will generate X_embeddings.npy for later use):**
   ```bash
   python SCRIPTS/extract_embeddings.py
   ```
5. **Train the Models:**
   - **BERT-Based Classification (Neural Network & Logistic Regression)**
     ```bash
     python SCRIPTS/train_model.py
     ```
   - **Or run the interactive Jupyter Notebook:**
     ```bash
     jupyter notebook SCRIPTS/bert_based_classification.ipynb
     ```

6. **Predict Genre for a New Song:**
   ```bash
   python SCRIPTS/predict_genre.py
   ```

7. **Generate Reports & Visualizations:**
   - Run the visualization notebook:
     ```bash
     jupyter notebook SCRIPTS/visualization.ipynb
     ```
   - Confusion Matrices, PCA visualization, and Classification Reports can be found in the `OUTPUT/` folder.

## 📜 License
This project is licensed under the MIT License. See the `LICENSE.md` file for details.

## 📚 References
- Research papers on NLP and song lyric analysis
- Scikit-Learn, TensorFlow, and Hugging Face Transformers documentation
- GitHub repositories on music genre classification

