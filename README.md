
This project builds an Artificial Neural Network (ANN) model to classify breast cancer tumors as **benign** or **malignant** using the **Wisconsin Breast Cancer Dataset**.

The aim is to demonstrate how deep learning can support early and accurate diagnosis in medical applications.

---

## 🔍 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- The dataset contains **30 numerical features** derived from digitized images of breast mass samples.
- Target variable: `diagnosis`  
  - `M` = Malignant  
  - `B` = Benign

---

## 💻 Technologies Used

- Python 3.x  
- Keras (TensorFlow backend)  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn

---

## 🧠 Model Architecture

- Input Layer: 30 neurons (1 per feature)
- Hidden Layers:  
  - Dense (16 units) + ReLU + Dropout  
  - Dense (16 units) + ReLU + Dropout  
- Output Layer:  
  - Dense (1 unit) + Sigmoid (for binary classification)

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

model = Sequential()
model.add(Input(shape=(30,)))
model.add(Dense(16, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
````

---

## 🧪 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision / Recall (optional)

**Sample Confusion Matrix Result:**

```
[[65  2]
 [ 3 44]]
```

* **True Negatives:** 65
* **True Positives:** 44
* **False Positives:** 2
* **False Negatives:** 3

---

## 📊 How to Run

1. Clone the repository:

   ```
   git clone https://github.com/your-username/breast-cancer-ann.git
   cd breast-cancer-ann
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```
   Open Classification_ANN.ipynb in Jupyter or Google Colab
   ```

---

## 📂 File Structure

```
├── data.csv                  # Dataset file
├── Classification_ANN.ipynb # Main notebook
├── README.md                 # This file
├── download.png              # Confusion matrix image
└── requirements.txt          # (Optional) Python dependencies
```

---

## ✅ Key Learnings

* Applied neural networks to a real-world medical dataset
* Understood the importance of reducing false negatives in healthcare
* Gained experience with dropout regularization and evaluation metrics

---

## 📬 Contact

For feedback or collaboration, feel free to reach out via LinkedIn or GitHub.
