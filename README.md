# Laporan Proyek Machine Learning - Gibran Faktian Anwar

## Domain Proyek

Latar Belakang:

    Penyakit pada organ hati masih menjadi perhatian dalam kesehatan dunia. Sekitar 4.5 juta orang dewasa di dunia didiagnosis dengan penyakit hati (CDC, 2022). Kondisi tersebut mampu menjadi beban biaya bagi negara. Adanya kerusakan pada hati mampu menyebabkan terbentuknya jaringan parut atau fibrosis. Awalnya, fibrosis tidak akan menimbulkan kerusakan fungsi pada hati, namun dengan meluasnya kerusakan mampu menyebabkan hati kehilangan fungsi.

**Mengapa masalah ini harus diselesaikan?**

- **Deteksi Dini dan Akurat**: Model machine learning dapat membantu memprediksi penyakit hati secara lebih dini dan akurat, sehingga memungkinkan penanganan lebih cepat dan tepat sasaran.
- **Pencegahan Perkembangan Penyakit:** Dengan mengenali pola risiko secara otomatis, model ini dapat membantu mencegah perkembangan fibrosis menjadi kerusakan hati yang lebih serius, mengurangi risiko kehilangan fungsi hati dan beban ekonomi yang ditimbulkan.
- Source : [Liver Cirrhosis: Pathophysiology, Diagnosis, and Management](https://jurnalfkip.unram.ac.id/index.php/JBT/article/download/5763/3582/29907)

## Business Understanding

Pengembangan model prediksi penyakit hati ini memiliki potensi untuk memberikan manfaat bagi berbagai pihak, termasuk dokter dan tenaga medis. Model ini dapat membantu mereka dalam menentukan seseorang terkena penyakit hati atau tidak.

### Problem Statements

Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:

- Bagaimana cara mengidentifikasi pasien yang berisiko tinggi memiliki penyakit hati dari data medis mereka?
- Model machine learning seperti apa yang dapat memberikan prediksi terbaik?

### Goals

Tujuan dibentuknya proyek ini adalah:

- Membangun model machine learning yang baik dan dapat mengidentifikasi seseorang terkenya penyakit hati atau tidak.
- Mendapatkan performa model terbaik dengan memanfaatkan melakukan tuning hyperparameter dan algoritma machine learning.
- ### Solution statements

  - Mempertimbangkan lima algoritma klasifikasi yaitu Support Vector Machine, Logistic Regression, Random Forest dan K-Nearest Neighbor.
  - Melakukan hyperparameter tuning untu mencari parameter terbaik untuk model.

## Data Understanding

Dataset yang digunakan adalah [Liver Disease: 1700 Records](https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset) dari Kaggle, yang memiliki **usability score** sempurna sebesar **10.00** . Usability score ini merupakan indikator kualitas dataset berdasarkan sistem penilaian Kaggle, yang menilai kemudahan penggunaan dan kelayakan dataset. Dengan skor ini, dapat dipastikan bahwa dataset tersebut memiliki kualitas yang tinggi dan cocok digunakan untuk proyek machine learning. Dataset ini mencakup informasi medis dari **1.700 records** dengan total **11 variabel** , yang relevan untuk analisis dan pemodelan machine learning.

### Variabel-variabel pada Liver Disease: 1700 Record Dataset adalah sebagai berikut:

* **Age** : Usia individu, dengan rentang antara 20 hingga 80 tahun.
* **Gender** : Jenis kelamin individu, yang bisa berupa Male (Laki-laki) atau Female (Perempuan).
* **BMI** : Indeks Massa Tubuh (Body Mass Index), dengan rentang antara 15 hingga 40.
* **Alcohol Consumption** : Konsumsi alkohol per minggu, dengan rentang antara 0 hingga 20 unit per minggu.
* **Smoking** : Status merokok, yang dapat berupa No (0) atau Yes (1).
* **Genetic Risk** : Predisposisi genetik terhadap penyakit, dengan nilai Low (0), Medium (1), atau High (2).
* **Physical Activity** : Durasi aktivitas fisik per minggu, dengan rentang antara 0 hingga 10 jam per minggu.
* **Diabetes** : Status diabetes, yang dapat berupa No (0) atau Yes (1).
* **Hypertension** : Status hipertensi, yang dapat berupa No (0) atau Yes (1).
* **Liver Function Test** : Hasil tes enzim hati, dengan rentang antara 20 hingga 100.
* **Diagnosis** : Indikator adanya penyakit hati, dengan No (0) atau Yes (1).

### Data Visualization

- Pie chart : memperlihatkan sebaran data dari fitur diagnosis

![image](https://github.com/user-attachments/assets/4e8bd838-d22c-4e4b-9c7c-cf42333fbe96)

  - Insight

    - Sebaran data dalam fitur diagnosis tidak merata
    - No : 44.9%, Yes : 55.1%
- Matrix Correlation : memperlihatkan korelasi antar fiturnya

![image](https://github.com/user-attachments/assets/44da85c1-86ca-436f-83b8-1be1b9fc45df)


  - Insight :
    - Hanya ada sedikit kotak yang memiliki relasi yang kuat anat fiturnya
    - Alcoholconsumption & liverfunctiontest memiliki korelasi yang sangat kuat terhadap diagnosis

## Data Preparation

Proses Data Preparation penting untuk memastikan data yang bersih dan siap digunakan dalam model machine learning. Langkah-langkah dalam persiapan data mencakup berbagai tahapan berikut:

- One hot encoding untuk kategorikal feature menjadi Numerical Feature
- Splitting dataset, Membagi data menjadi train = 75% dan Test = 25%.

## Modeling

Algoritma pada proyek ini melakukan pemodelan dengan 4 algoritma, yaitu:

1. Support Vector Machine
2. Logistic Regression
3. Random Forest
4. K-Nearest Neighbors (KNN)

Keempat model machine learning tadi dibangun dengan **parameter default** untuk melakukan klasifikasi:

```
    svm = SVC()
    svm.fit(X_train, y_train)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
```

Penjelasan parameter default tiap algoritmanya:

1. SVM adalah algoritma klasifikasi yang mencoba untuk menemukan hyperplane (garis atau bidang) yang memisahkan data ke dalam kelas-kelas yang berbeda dengan margin yang terbesar.:

* **C** : 1.0 – Mengontrol seberapa keras model mencoba untuk menghindari kesalahan. Nilai lebih kecil, lebih toleran terhadap kesalahan.
* **Kernel** : 'rbf' – Fungsi untuk mengubah data ke dimensi yang lebih tinggi agar lebih mudah dipisahkan.
* **Gamma** : 'scale' – Menentukan seberapa besar pengaruh tiap titik data terhadap keputusan model.
* **Shrinking** : True – Mempercepat pelatihan dengan mengurangi ukuran model saat optimasi.

2. Logistic Regression adalah metode klasifikasi yang digunakan untuk memprediksi probabilitas suatu kelas. Ini mengasumsikan hubungan logistik antara input dan output.

* **Penalty** : 'l2' – Menggunakan regularisasi untuk mencegah overfitting.
* **C** : 1.0 – Menyempurnakan keseimbangan antara kesalahan dan kompleksitas model.
* **Solver** : 'lbfgs' – Metode untuk mencari solusi terbaik saat pelatihan.
* **Max_iter** : 100 – Maksimum iterasi untuk melatih model.

3. Random Forest adalah algoritma ensemble learning yang terdiri dari banyak pohon keputusan (decision trees). Setiap pohon dalam hutan ini dibangun secara acak dengan subset data dan subset fitur, dan keputusan akhir diambil berdasarkan voting mayoritas (untuk klasifikasi) atau rata-rata (untuk regresi).

* **n_estimators** : 100 – Jumlah pohon dalam hutan. Semakin banyak pohon, semakin baik hasilnya.
* **Criterion** : 'gini' – Mengukur seberapa baik pohon memisahkan data.
* **Max_depth** : None – Pohon bisa tumbuh tak terbatas hingga data terpisah sempurna.
* **Bootstrap** : True – Memungkinkan pengambilan sampel data berulang saat membangun pohon.

4. KNN adalah algoritma klasifikasi yang sederhana yang mengklasifikasikan data berdasarkan kedekatannya dengan data lain dalam ruang fitur.

* **n_neighbors** : 5 – Jumlah tetangga yang diperhitungkan untuk keputusan klasifikasi.
* **Weights** : 'uniform' – Semua tetangga dianggap sama pentingnya.
* **Algorithm** : 'auto' – Memilih algoritma terbaik untuk menemukan tetangga.
* **Metric** : 'minkowski' – Cara mengukur jarak antar titik data.



## Hyperparameter Tuning

Function yang kita buat adalah sebagai berikut:

1. `StratifiedKFold` adalah metode pembagian data untuk cross-validation yang digunakan untuk memastikan bahwa distribusi label di setiap lipatan (fold) adalah proporsional atau serupa dengan distribusi di dataset keseluruhan. Teknik ini sangat baik digunakan karena dataset kita tidak seimbang.               No = 44,9% dan Yes = 55,1%

Berikut codenya & penjelasan parameternya:

```
fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
```

* **`n_splits=2`** : Membagi data menjadi 2 bagian (fold).
* **`shuffle=True`** : Mengacak data sebelum membagi, agar lebih acak dan adil.
* **`random_state=42`** : Menetapkan seed untuk memastikan pembagian data yang konsisten setiap kali dijalankan.

---


2. `GridSearchCV`e metode untuk mencari kombinasi parameter terbaik bagi sebuah model machine learning. Metode ini mencoba berbagai kombinasi dari parameter yang telah ditentukan dan memilih yang memberikan hasil terbaik berdasarkan kriteria yang diinginkan, seperti akurasi.

Berikut codenya & penjelasan parameternya:

```
def grid_search(model, folds, params, scoring):
    grid_search = GridSearchCV(model,
                               cv=folds,
                               param_grid=params,
                               scoring=scoring,
                               n_jobs=1,
                               verbose=1)
    return grid_search
```

* **`model`** : Model yang ingin diuji (misalnya, Random Forest, KNN, dll.).
* **`cv (folds)`** : Jumlah pembagian data untuk validasi silang (misalnya, `cv=5` berarti data dibagi menjadi 5 bagian).
* **`param_grid (params)`** : Kombinasi pengaturan (hyperparameter) yang ingin dicoba untuk model.
* **`scoring`** : Cara untuk menilai performa model (misalnya, `'accuracy'` untuk akurasi).
* **`n_jobs`** : Jumlah pekerjaan paralel yang dapat dijalankan (misalnya, `n_jobs=1` untuk satu pekerjaa).
* **`verbose`** : Tingkat informasi yang ditampilkan selama pencarian grid (misalnya, `verbose=1` untuk informasi dasar).

---



3. `best_score_params` function ini digunakan untuk menampilkan score dari parameter yang ditest.

   Berikut codenya :

   ```
   def best_score_param(model):
       print('Best score: ', model.best_score_)
       print('Best parameter: ', model.best_params_)
   ```


## Evaluation

Setelah melakukan pemodelan, dari keempat model belum menunjukan performa terbaiknya.

```
                    Model  Accuracy  Precision    Recall
0  Support Vector Machine  0.748235   0.747832  0.741517
1     Logistic Regression  0.814118   0.813119  0.810768
2           Random Forest  0.875294   0.876415  0.871602
3                     KNN  0.736471   0.734270  0.735370
```

 Untuk mengoptimalkan kinerja model kita melakukan beberapa hyperparameter tuning di tiap modelnya. 

maka didapat tiap modelnya sebagai berikut :

1. SVM

   Accuracy:  0.8188235294117647,
   Precision:  0.8277310924369747,
   Recall:  0.8454935622317596
2. Logistic Regression

   Accuracy:  0.8188235294117647,
   Precision:  0.825,
   Recall:  0.8497854077253219
3. Random Forest

   Accuracy:  0.88,
   Precision:  0.8760330578512396,
   Recall:  0.9098712446351931
4. K-Nearest Neighbors

   Accuracy:  0.7905882352941176,
   Precision:  0.8103448275862069,
   Recall:  0.8068669527896996

Dalam kasus ini, dapat disimpulkan bawha :

* Random Forest memiliki Accuracy, Precision, Recall terbaik dari 4 algrtima yang ada
* Random forest akan dipilih sebagai metode terbaik karena performanya dalam case ini

### Metri Evaluasi

Metrik evaluasi yang digunakan dalam proyek ini adalah Accuracy, Precision, dan Recall. Hasil evaluasi menunjukkan perbedaan kinerja di antara model, terutama sebelum dan sesudah hyperparameter tuning:

1. **Accuracy**: Mengukur persentase total prediksi yang benar dari seluruh data.
2. **Precision**: Mengukur ketepatan dari prediksi positif, yaitu berapa banyak dari prediksi positif yang benar-benar positif.
3. **Recall**: Mengukur kemampuan model dalam mendeteksi semua kasus positif, yaitu berapa banyak kasus positif yang berhasil dikenali dari keseluruhan kasus positif.

## Kesimpulan

Dengan membangun keempat algoritma machine learning sekaligus dan melakukan hyperparameter tuning untuk meningkatkan kinerjanya dan kita menemukan model machine learning yang memiliki performa terbaik diantara keempatnya. Kini kita dapat mengidentifikasi pasien yang memiliki penyakit hati menggunakan data medis mereka, dengan menggunakan model algoritma Random Forest. Model ini dipilih karena memiliki performa prediksi yang baik, tuning hyperparameter yang mudah, dan alternatif yang baus ketika dibutuhkan tingkat presisi yang lebih tinggi pada masalah yang lebih kompleks. 


dari kesimpulan diatas berarti kita sudah memecahkan Problem Statement kita.
