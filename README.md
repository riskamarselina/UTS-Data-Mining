# UTS Data Mining: Klasifikasi Kualitas Anggur (Wine Quality Classification)

**Nama:** Riska Marselina

**NIM:** 2304020008

**Program Studi:** Pendidikan Matematika

**Rombel:** 1

**Mata Kuliah:** Data Mining  
**Topik:** Klasifikasi  
**Dataset:** Wine Quality  
**Tools:** Python, Google Colab, Scikit-learn

---

## Deskripsi Proyek

Proyek ini merupakan bagian dari Ujian Tengah Semester (UTS) mata kuliah Data Mining. Tujuannya adalah membangun model klasifikasi untuk memprediksi kualitas anggur berdasarkan fitur-fitur kimiawi yang terukur dari setiap sampel anggur. Dataset yang diberikan terdiri dari dua bagian: data training yang memiliki label kualitas (`quality`) dan data testing yang tidak memiliki label. Tugas utama adalah melatih model menggunakan data training, lalu menggunakan model tersebut untuk memprediksi kualitas pada data testing.

---

## Struktur Repository

```
.
├── wine_quality_uts.ipynb          # Notebook Google Colab utama
├── hasilprediksi_3digitNIMterakhir.csv  # File hasil prediksi
├── eda_distribusi.png              # Visualisasi distribusi data
├── heatmap_korelasi.png            # Heatmap korelasi fitur
├── confusion_matrix.png            # Confusion matrix validasi
├── feature_importance.png          # Grafik feature importance
├── distribusi_prediksi.png         # Distribusi prediksi testing
└── README.md                       # Dokumentasi proyek
```

---

## Alur Pengerjaan

### Step 1: Eksplorasi Data Awal (EDA)

Sebelum membangun model, dilakukan eksplorasi terhadap data training untuk memahami karakteristik data.

**Temuan dari EDA:**
- Dataset training terdiri dari **857 baris** dan **13 kolom** (11 fitur kimiawi, 1 kolom `quality`, 1 kolom `Id`).
- Dataset testing terdiri dari **286 baris** dan **12 kolom** (11 fitur kimiawi dan `Id`, tanpa kolom `quality`).
- Variabel target `quality` memiliki rentang nilai 3 hingga 8, dengan distribusi sebagai berikut:

| Quality | Jumlah Sampel |
|---------|--------------|
| 3       | 6            |
| 4       | 26           |
| 5       | 362          |
| 6       | 341          |
| 7       | 109          |
| 8       | 13           |

Kelas 5 dan 6 mendominasi data (82.38% dari total), sementara kelas 3 dan 8 sangat sedikit. Kondisi ini menunjukkan adanya **class imbalance** yang perlu diperhatikan dalam pemilihan model dan evaluasi.

---

### Step 2: Pembersihan Data (Data Cleaning)

Dilakukan pemeriksaan terhadap kualitas data sebelum digunakan untuk pelatihan model.

**Hasil pemeriksaan:**
- **Missing values:** Tidak ditemukan nilai yang hilang pada dataset training maupun testing (seluruh kolom bernilai 0 missing values).
- **Data duplikat:** Tidak ditemukan baris duplikat pada kedua dataset.
- **Kesimpulan:** Dataset dalam kondisi bersih dan siap untuk diproses lebih lanjut tanpa memerlukan imputasi atau penghapusan baris.

---

### Step 3: Feature Scaling

Setelah data dipastikan bersih, dilakukan normalisasi fitur menggunakan **StandardScaler** dari scikit-learn. Proses ini mengubah setiap fitur sehingga memiliki rata-rata 0 dan standar deviasi 1.

**Alasan scaling diperlukan:**
Nilai fitur dalam dataset memiliki rentang yang sangat bervariasi. Misalnya, `total sulfur dioxide` bisa mencapai nilai ratusan, sementara `chlorides` berada di bawah 1. Jika scaling tidak dilakukan, fitur dengan rentang nilai besar akan mendominasi perhitungan jarak dalam beberapa algoritma. Meskipun Random Forest secara teknis tidak sensitif terhadap skala fitur, scaling tetap diterapkan untuk menjaga konsistensi pipeline.

Proses fitting scaler dilakukan **hanya pada data training**, kemudian transformasi yang sama diterapkan ke data testing untuk menghindari data leakage.

---

### Step 4: Pemilihan dan Pelatihan Model

Model yang dipilih adalah **Random Forest Classifier** dengan konfigurasi sebagai berikut:

| Parameter          | Nilai   |
|--------------------|---------|
| n_estimators       | 500     |
| max_depth          | None    |
| min_samples_split  | 2       |
| min_samples_leaf   | 1       |
| random_state       | 42      |

**Alasan pemilihan Random Forest:**
1. Random Forest merupakan metode ensemble yang menggabungkan banyak decision tree sehingga lebih tahan terhadap overfitting dibanding single decision tree.
2. Mampu menangani distribusi kelas yang tidak seimbang dengan cukup baik.
3. Memberikan informasi feature importance yang berguna untuk interpretasi.
4. Tidak memerlukan asumsi distribusi data normal.

Data training dibagi menjadi dua bagian: 80% untuk melatih model dan 20% sebagai validation set. Pembagian dilakukan secara stratified untuk menjaga proporsi kelas tetap seimbang di kedua subset.

---

### Step 5: Evaluasi Model

**Hasil evaluasi pada Validation Set (20% data training):**

| Metrik     | Nilai  |
|------------|--------|
| Accuracy   | ~66%   |

**Hasil evaluasi Cross-Validation (5-Fold Stratified):**

| Fold | Akurasi |
|------|---------|
| 1    | 0.5756  |
| 2    | 0.6512  |
| 3    | 0.6374  |
| 4    | 0.6608  |
| 5    | 0.6784  |
| **Rata-rata** | **0.6407** |
| **Std. Deviasi** | **0.0352** |

**Interpretasi:**
- Rata-rata akurasi cross-validation sebesar **64.07%** dengan standar deviasi **3.52%** menunjukkan performa model cukup stabil dan tidak terlalu bervariasi antar fold.
- Akurasi yang tidak mencapai 100% adalah hal yang wajar pada dataset ini karena adanya class imbalance yang signifikan, serta perbedaan kualitas anggur yang berdekatan (misal kualitas 5 vs 6) memang sulit dibedakan hanya dari fitur kimiawi.
- Confusion matrix menunjukkan bahwa model paling baik dalam memprediksi kelas 5 dan 6 (kelas mayoritas), sedangkan kelas 3, 4, dan 8 lebih jarang diprediksi dengan benar karena minimnya sampel pelatihan.

---

### Step 6: Analisis Feature Importance

Berdasarkan output Random Forest, berikut tingkat kepentingan setiap fitur terhadap prediksi kualitas anggur:

| Peringkat | Fitur                  | Importance Score |
|-----------|------------------------|-----------------|
| 1         | alcohol                | 0.1418          |
| 2         | sulphates              | 0.1172          |
| 3         | volatile acidity       | 0.1109          |
| 4         | total sulfur dioxide   | 0.0983          |
| 5         | density                | 0.0886          |
| 6         | chlorides              | 0.0818          |
| 7         | citric acid            | 0.0768          |
| 8         | fixed acidity          | 0.0758          |
| 9         | pH                     | 0.0749          |
| 10        | residual sugar         | 0.0674          |
| 11        | free sulfur dioxide    | 0.0664          |

**Interpretasi:**
- **Alcohol (0.1418)** menjadi fitur paling berpengaruh. Anggur dengan kadar alkohol lebih tinggi cenderung mendapat penilaian kualitas yang lebih baik karena berkaitan dengan proses fermentasi yang optimal.
- **Sulphates (0.1172)** berperan penting sebagai agen antimikroba dan antioksidan yang membantu menjaga kesegaran anggur.
- **Volatile acidity (0.1109)** yang tinggi menandakan kandungan asam asetat berlebih, yang dapat memberikan rasa tidak enak seperti cuka dan menurunkan kualitas.
- **Free sulfur dioxide (0.0664)** memiliki pengaruh paling kecil di antara semua fitur.

---

### Step 7: Prediksi Data Testing

Setelah model dilatih menggunakan seluruh data training, model digunakan untuk memprediksi kualitas pada 286 sampel data testing.

**Distribusi hasil prediksi:**

| Quality | Jumlah Prediksi | Persentase |
|---------|----------------|------------|
| 5       | 132            | 46.15%     |
| 6       | 124            | 43.36%     |
| 7       | 30             | 10.49%     |

**Interpretasi:**
Pola distribusi prediksi pada data testing (didominasi kelas 5 dan 6) konsisten dengan distribusi kelas pada data training. Tidak ada prediksi untuk kelas 3, 4, dan 8, yang dapat dijelaskan oleh sangat sedikitnya representasi kelas-kelas tersebut dalam data training sehingga model tidak memiliki cukup informasi untuk memprediksi kelas tersebut.

---

### Step 8: Penyimpanan Hasil Prediksi

Hasil prediksi disimpan dalam file CSV dengan format yang sesuai ketentuan soal, yaitu hanya memuat dua kolom: `Id` dan `quality`.

Format file output: `hasilprediksi_008.csv`

Contoh isi file:

```
Id,quality
222,5
1514,6
417,5
754,5
516,5
```

---

## Kesimpulan

1. Dataset Wine Quality dalam kondisi bersih tanpa missing values sehingga tidak diperlukan imputasi data.
2. Model Random Forest dengan 500 pohon berhasil dilatih dan dievaluasi dengan rata-rata akurasi cross-validation sebesar **64.07%**.
3. Fitur **alcohol**, **sulphates**, dan **volatile acidity** merupakan tiga fitur paling berpengaruh dalam menentukan kualitas anggur.
4. Model berhasil menghasilkan prediksi untuk 286 sampel data testing dengan distribusi yang konsisten terhadap data training.
5. Keterbatasan utama adalah adanya class imbalance pada kelas 3, 4, dan 8 yang menyebabkan model kurang optimal dalam memprediksi kelas-kelas tersebut. Penggunaan teknik oversampling seperti SMOTE dapat menjadi arah pengembangan selanjutnya.

---

## Cara Menjalankan

1. Buka Google Colab di [colab.research.google.com](https://colab.research.google.com).
2. Upload file `wine_quality_uts.ipynb` ke Colab.
3. Upload `data_training.csv` dan `data_testing.csv` ke sesi Colab (menggunakan panel Files di sidebar kiri atau melalui `files.upload()`).
4. Jalankan seluruh cell secara berurutan dari atas ke bawah.
5. File hasil prediksi `hasilprediksi_008.csv` akan otomatis terunduh ke komputer setelah cell terakhir dijalankan.

---

## Referensi

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org)
