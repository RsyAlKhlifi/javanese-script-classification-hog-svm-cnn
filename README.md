# Klasifikasi Tulisan Tangan Aksara Jawa Menggunakan HOG+SVM dan CNN

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem klasifikasi karakter tulisan tangan **Aksara Jawa (Hanacaraka)** berbasis **Pengolahan Citra Digital (PCD)** dan **Machine Learning**. Penelitian ini membandingkan dua pendekatan klasifikasi:

1. **Histogram of Oriented Gradients (HOG) + Support Vector Machine (SVM)**
2. **Convolutional Neural Network (CNN)**

Tujuan utama penelitian adalah mengidentifikasi model yang paling efektif dalam mengenali karakter tulisan tangan Aksara Jawa berdasarkan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1-Score.

---

## Tujuan Penelitian

* Mengembangkan sistem klasifikasi karakter tulisan tangan Aksara Jawa.
* Menerapkan metode ekstraksi fitur HOG dan klasifikasi SVM.
* Membangun model CNN sebagai metode pembanding.
* Membandingkan performa HOG+SVM dan CNN untuk menentukan model terbaik.

---

## Dataset

Dataset yang digunakan adalah **[Hanacaraka Dataset](https://www.kaggle.com/datasets/vzrenggamani/hanacaraka)** dari Kaggle, terdiri dari citra tulisan tangan karakter Aksara Jawa.

### Statistik Dataset

| Keterangan      | Jumlah |
| --------------- | ------ |
| Jumlah Kelas    | 20     |
| Total Citra     | 1580   |
| Data per Kelas  | 79     |
| Data Training   | 1093   |
| Data Validation | 234    |
| Data Testing    | 235    |

### Kelas Karakter

```text
ha  na  ca  ra  ka
da  ta  sa  wa  la
pa  dha ja  ya  nya
ma  ga  ba  tha nga
```

---

## Alur Penelitian

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Data Splitting
   │
   ▼
Data Augmentation
   │
   ├─────────────┐
   ▼             ▼
 HOG          CNN
   │             │
   ▼             ▼
 SVM       Deep Learning
   │             │
   └──────┬──────┘
          ▼
      Evaluation
```

---

## Preprocessing

Tahapan preprocessing yang digunakan:

1. Grayscale Conversion
2. Gaussian Blur
3. Otsu Thresholding
4. Cropping
5. Padding
6. Resize (64 × 64)
7. Normalization

Tujuan preprocessing adalah menyeragamkan citra sehingga model dapat mempelajari pola karakter dengan lebih baik.

---

## Data Augmentation

Augmentasi diterapkan **hanya pada data training** untuk meningkatkan variasi data dan mengurangi overfitting.

Metode augmentasi yang digunakan:

* Rotasi
* Translasi
* Pergeseran posisi

Jumlah data training meningkat secara signifikan setelah proses augmentasi.

---

## Ekstraksi Fitur HOG

Parameter HOG:

```python
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'
```

Hasil ekstraksi menghasilkan feature vector yang digunakan sebagai input model SVM.

---

## Model HOG + SVM

### Konfigurasi

```python
SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)
```

### Hasil Evaluasi

| Metrik              | Nilai  |
| -------------------- | ------ |
| Train Accuracy       | 98.96% |
| Validation Accuracy  | 83.33% |
| Test Accuracy        | 86.81% |
| Macro Precision      | 0.88   |
| Macro Recall         | 0.87   |
| Macro F1-Score       | 0.87   |

> Gap yang cukup besar antara Train Accuracy (98.96%) dan Validation/Test Accuracy (±83–87%) mengindikasikan model HOG+SVM mengalami **overfitting** pada data training.

---

## Model CNN

CNN digunakan sebagai metode pembanding dengan kemampuan ekstraksi fitur otomatis.

### Arsitektur CNN

```text
Input Image
      │
      ▼
Conv2D
      │
      ▼
MaxPooling
      │
      ▼
Conv2D
      │
      ▼
MaxPooling
      │
      ▼
Flatten
      │
      ▼
Dense
      │
      ▼
Softmax
```

### Hasil Evaluasi

| Metrik              | Nilai  |
| -------------------- | ------ |
| Train Accuracy       | 99.69% |
| Validation Accuracy  | 90.60% |
| Test Accuracy        | 90.21% |
| Macro Precision      | 0.91   |
| Macro Recall         | 0.90   |
| Macro F1-Score       | 0.90   |

---

## Perbandingan Model

| Metrik    | HOG+SVM |      CNN |
| --------- | ------: | -------: |
| Accuracy  |    0.87 | **0.90** |
| Precision |    0.88 | **0.91** |
| Recall    |    0.87 | **0.90** |
| F1-Score  |    0.87 | **0.90** |

### Kesimpulan Perbandingan

CNN memberikan performa yang lebih baik dibandingkan HOG+SVM pada seluruh metrik evaluasi. Selain memperoleh akurasi yang lebih tinggi, CNN juga memiliki kemampuan generalisasi yang lebih baik dengan gap train-validation yang lebih kecil (99.69% → 90.60%) dibandingkan HOG+SVM (98.96% → 83.33%).

---

## Struktur Repository

```
.
├── javanese-script-classification-hog-svm-cnn_source code.ipynb   # Notebook utama (EDA, preprocessing, HOG+SVM, CNN, evaluasi)
└── README.md
```

Isi notebook:
1. Import Library
2. Load Dataset dari Kaggle
3. EDA (Eksplorasi Data)
4. Preprocessing
5. Data Augmentation
6. Model 1 — HOG + SVM
7. Model 2 — CNN
8. Evaluasi & Perbandingan Model

---

## Cara Menjalankan

1. **Clone repo**
```bash
   git clone https://github.com/<username>/javanese-script-classification-hog-svm-cnn.git
   cd aksara-jawa-hog-svm-cnn
```

2. **Install dependensi**
```bash
   pip install numpy pandas opencv-python scikit-learn scikit-image tensorflow matplotlib seaborn kagglehub
```

3. **Unduh dataset**
   Notebook mengunduh dataset otomatis lewat `kagglehub`:
```python
   import kagglehub
   path = kagglehub.dataset_download("vzrenggamani/hanacaraka")
```
   Pastikan sudah memiliki akun Kaggle dan API token (`kaggle.json`) terkonfigurasi jika dijalankan di luar Kaggle/Colab.

4. **Jalankan notebook**
```bash
   javanese-script-classification-hog-svm-cnn_source code.ipynb
```
   GPU disarankan untuk mempercepat training CNN.

---

## Tech Stack

* Python
* OpenCV
* NumPy
* Pandas
* Scikit-Learn
* Scikit-Image
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Kagglehub (unduh dataset)

---

## Saran Pengembangan

* Menerapkan *hyperparameter tuning* (mis. `GridSearchCV`) pada SVM untuk mengurangi overfitting, misalnya dengan menyesuaikan `C` dan `gamma`, atau menambah regularisasi
* Menambah variasi augmentasi (elastic distortion, noise) untuk memperkuat generalisasi kedua model
* Eksperimen arsitektur CNN yang lebih dalam atau *transfer learning* dari model pretrained
* Menambah jumlah data per kelas untuk mengurangi risiko *class imbalance* pada evaluasi per karakter

---

## Referensi

* Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
* Cortes, C., & Vapnik, V. (1995). Support Vector Machines.
* LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning.

---

## Author

**Kelompok 2 Kelas 24B**
Muhammad Geralldo Agatha Saputra (24031554091)
Moh. Rasya Al Khalifi (24031554132)
Najmu Tsaqib Arsalan (24031554026)

**Program Studi Sains Data**
**Universitas Negeri Surabaya**

**2025/2026**
