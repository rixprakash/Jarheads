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
📂 project 1
│── 📄 README.md               # Project documentation
│── 📄 LICENSE.md              # MIT License
│── 📄 requirements.txt        # List of project dependencies for easy installation
│── 📂 SCRIPTS/                # All scripts for analysis
│   │── preprocess_data.py     # Data cleaning and preprocessing
│   │── model_training.py      # Machine learning model training
│   │── visualization.ipynb    # Data visualization and analysis
│── 📂 DATA/                   # Data files
│   │── tcc_ceds_music.csv     # Original dataset
│   │── processed_data.csv     # Cleaned dataset ready for analysis
│── 📂 OUTPUT/                 # Generated results
│   │── confusion_matrix.png   # Model performance visualization
│   │── genre_classification_report.pdf  # Analysis report
│   │── PCA_visualization.png  # Dimensionality reduction visualization
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

