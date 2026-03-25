# 🐝 Bee vs Wasp Classifier (Computer Vision Project)

A deep learning-based computer vision system that classifies images as **Bee 🐝 or Wasp 🐝** using a pretrained EfficientNet model. The project includes a complete ML pipeline from dataset preparation to deployment via a Streamlit web app.

---

## 🚀 Features

* ✅ Binary image classification (Bee vs Wasp)
* ✅ Transfer learning using EfficientNet-B0
* ✅ Data augmentation for better generalization
* ✅ Model evaluation with classification report
* ✅ Confusion matrix visualization
* ✅ Real-time prediction via Streamlit app
* ✅ Confidence score for predictions

---

## 🧠 Model Details

* **Architecture:** EfficientNet-B0 (Transfer Learning)
* **Framework:** PyTorch
* **Classes:** Bee, Wasp
* **Accuracy:** ~95% on validation dataset

---

## 📁 Project Structure

```
bee-project/
│
├── data/                  # Processed dataset (ignored in Git)
│   ├── train/
│   ├── val/
│
├── data_raw/              # Raw dataset (ignored)
│
├── scripts/
│   ├── build_dataset.py
│   ├── download_data.py
│
├── models/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── app.py
│   ├── utils.py
│   └── bee_model.pth      # Trained model (ignored)
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

This project uses the **Bee vs Wasp dataset from Kaggle**.

> Dataset is not included in the repository due to size constraints.

### 📥 Download Dataset

1. Download from Kaggle:
   https://www.kaggle.com/datasets/jerzydziewierz/bee-vs-wasp

2. Extract into:

```
data_raw/bee_vs_wasp/
```

3. Build dataset:

```bash
python scripts/build_dataset.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bee-vs-wasp-classifier.git
cd bee-vs-wasp-classifier
```

### 2. Install dependencies

```bash
pip install torch torchvision pillow matplotlib scikit-learn streamlit
```

---

## 🏋️ Training the Model

```bash
python models/train.py
```

---

## 📈 Evaluation

```bash
python models/evaluate.py
```

---

## 🔍 Inference (Single Image)

```bash
python models/inference.py
```

---

## 🌐 Run Web App (Streamlit)

```bash
streamlit run models/app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🖥️ Sample Output

* Prediction: **Bee**
* Confidence: **0.95**

---

## 📊 Results

* Accuracy: **~95%**
* Strong performance on real-world images
* Robust to lighting and background variations

---

## 🧠 Key Learnings

* Transfer learning significantly improves performance on small datasets
* Data preprocessing and augmentation are critical for generalization
* Model evaluation (confusion matrix, F1-score) is essential for reliability
* End-to-end ML pipelines improve reproducibility

---

## 🚀 Future Improvements

* 🔍 Multi-class classification (Honeybee, Bumblebee, Wasp, Hornet)
* 🎥 Video-based detection
* 📦 YOLO-based object detection
* ☁️ Cloud deployment

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Dhruv Choudhary**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
