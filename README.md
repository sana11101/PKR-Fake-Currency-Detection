# 🇵🇰 PKR Fake Currency Detector

An AI-powered system to detect fake Pakistani Rupee (PKR) banknotes using Convolutional Neural Networks (CNN) with Grad-CAM explainability.

---

## 👩‍💻 Team
- **Sana** — Dataset collection, Augmentation, CNN Model Training, Model Comparison
- **Aqsa** — Grad-CAM Visualization, Streamlit Web App UI, Documentation

---

## 🎯 What This Project Does
1. Takes a photo of a PKR banknote (100, 500, or 1000)
2. Detects whether it is **REAL** or **FAKE**
3. Shows **which part of the note** looks suspicious using a Grad-CAM heatmap
4. Compares CNN accuracy with KNN and Logistic Regression

---

## 🌟 What Makes This Unique
- Pakistani Rupee specific (2025 updated notes)
- Self-collected original dataset
- Grad-CAM explainability — shows *where* the fake region is
- Multi-model comparison (CNN vs KNN vs Logistic Regression)
- Working web app — not just a notebook

---

## 📁 Project Structure
```
PKR-Fake-Currency-Detection/
│
├── dataset/
│   ├── real/          ← real PKR note photos (100, 500, 1000)
│   └── fake/          ← printed fake note photos
│
├── augmentation/
│   └── augment.py     ← expands dataset using image transformations
│
├── model/
│   ├── train_cnn.py        ← builds and trains CNN model
│   ├── train_comparison.py ← trains KNN and Logistic Regression
│   └── evaluate.py         ← shows accuracy, confusion matrix
│
├── gradcam/
│   └── visualize.py   ← generates Grad-CAM heatmap on a note image
│
├── app/
│   └── app.py         ← Streamlit web app (upload note → get result)
│
├── notebooks/
│   └── exploration.ipynb  ← data exploration notebook
│
├── requirements.txt   ← all Python libraries needed
└── README.md
```

---

## ⚙️ How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
- Put real note photos in `dataset/real/`
- Put fake note photos in `dataset/fake/`

### 3. Run augmentation
```bash
python augmentation/augment.py
```

### 4. Train CNN model
```bash
python model/train_cnn.py
```

### 5. Train comparison models
```bash
python model/train_comparison.py
```

### 6. Launch web app
```bash
streamlit run app/app.py
```

---

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.10+ | Main language |
| TensorFlow / Keras | CNN model |
| Scikit-learn | KNN, Logistic Regression |
| Albumentations | Data augmentation |
| OpenCV | Image processing |
| Streamlit | Web UI |
| Matplotlib | Charts and plots |
| NumPy | Numerical operations |

---

## 📊 Results
| Model | Accuracy |
|---|---|
| CNN | TBD after training |
| KNN | TBD after training |
| Logistic Regression | TBD after training |

---

## 📚 References
1. DeepMoney: Counterfeit Money Detection using GANs — https://pmc.ncbi.nlm.nih.gov/articles/PMC7924467/
2. Intelligent System for Pakistani Currency Recognition — https://thesesjournal.com/index.php/1/article/download/1308/1007/2157
3. Counterfeit Currency Detection using CNN — IEEE Xplore https://ieeexplore.ieee.org/document/9105683/
