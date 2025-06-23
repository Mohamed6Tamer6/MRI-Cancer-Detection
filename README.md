
# MRI Cancer Detection 🎯

This project aims to **detect brain tumors in MRI scans** using deep learning techniques.  
It includes training, evaluation, and deployment (via a simple web interface using `app.py`).

---

## 📂 Contents

- `MRI_Detection.ipynb` – Jupyter notebook containing the training and evaluation code.
- `app.py` – Flask app to run the trained model and detect tumors via a user interface.
- `requirements.txt` – Dependencies to set up the environment.

---

## 🧠 Model Weights

The final trained model is not included in this repository due to size limits.  
You must download it manually and place it **in the same folder as `app.py`**.

📦 **Download the trained model (`.pth`) from this link**:  
➡️ [Trained Model on Google Drive](https://drive.google.com/file/d/1jUB7r_3Ar2WXg0yPyLFv-nhmquHc6dLf/view?usp=sharing)

---

## 📊 Dataset Used

The dataset used for training and evaluation is available on Kaggle:  
➡️ [Brain Tumor MultiModal Image (CT and MRI)](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)

---

## ⚙️ How to Run

1. Create a new virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure the downloaded `.pth` file is in the same folder as `app.py`.

4. Run the app:
    ```bash
    python app.py
    ```

5. Access the web interface at `http://127.0.0.1:5000`

---

## 📬 Contact

If you have questions, feel free to reach out:  
**Mohamed Tamer** – [GitHub](https://github.com/Mohamed6Tamer6)

---

> 🚨 Note: This model is a research prototype and should not be used for real medical diagnosis.
