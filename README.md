<details>
   <summary>Supervised & Unsupervised Learning</summary>
<br>
   
### Algoritma Supervised Learning:

1. **Regresi Linear:**
   - **Penjelasan Singkat:** Digunakan untuk memprediksi nilai kontinu berdasarkan satu atau lebih variabel prediktor.
   - **Contoh:** Prediksi harga rumah berdasarkan jumlah kamar tidur.

2. **Regresi Logistik:**
   - **Penjelasan Singkat:** Digunakan untuk masalah klasifikasi biner, memprediksi probabilitas sebuah instance termasuk ke dalam suatu kelas.
   - **Contoh:** Prediksi apakah email adalah spam atau bukan.

3. **Decision Trees:**
   - **Penjelasan Singkat:** Membangun struktur pohon untuk membuat keputusan berdasarkan nilai-nilai fitur.
   - **Contoh:** Prediksi apakah seseorang akan membeli produk berdasarkan faktor seperti usia, pendapatan, dan preferensi.

4. **Random Forest:**
   - **Penjelasan Singkat:** Metode ensemble yang menggunakan beberapa pohon keputusan untuk meningkatkan akurasi dan generalisasi.
   - **Contoh:** Klasifikasi gambar berdasarkan fitur-fitur visual.

5. **SVM (Support Vector Machines):**
   - **Penjelasan Singkat:** Digunakan untuk klasifikasi dan regresi, memisahkan titik data ke dalam kelas-kelas yang berbeda.
   - **Contoh:** Klasifikasi dokumen ke dalam kategori berdasarkan isinya.

### Algoritma Unsupervised Learning:

1. **K-Means Clustering:**
   - **Penjelasan Singkat:** Membagi titik data ke dalam k kluster berdasarkan kesamaan.
   - **Contoh:** Pengelompokan pelanggan berdasarkan pola pembelian.

2. **Hierarchical Clustering:**
   - **Penjelasan Singkat:** Membuat struktur pohon kluster untuk memahami hubungan hierarki dalam data.
   - **Contoh:** Pengelompokan spesies tumbuhan berdasarkan kesamaan morfologi.

3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   - **Penjelasan Singkat:** Mengelompokkan titik data berdasarkan kerapatan, cocok untuk data dengan bentuk yang tidak teratur.
   - **Contoh:** Mengidentifikasi anomali dalam pola kredit pelanggan.

4. **PCA (Principal Component Analysis):**
   - **Penjelasan Singkat:** Mengurangi dimensi data sambil mempertahankan sebagian besar variabilitas.
   - **Contoh:** Reduksi dimensi pada citra wajah untuk analisis ekspresi.

5. **Apriori Algorithm:**
   - **Penjelasan Singkat:** Digunakan untuk pembelajaran aturan asosiasi dalam data mining.
   - **Contoh:** Mengidentifikasi pola pembelian yang sering muncul, seperti pembelian roti jika pembeli juga membeli mentega.

Ini hanyalah gambaran singkat, dan setiap algoritma memiliki lebih banyak nuansa dan penerapan yang lebih mendalam. Pilihan algoritma tergantung pada jenis data dan masalah yang dihadapi.
</details>

# Scikit-Learn Essentials: A Handy Cheatsheet for Data Scientists
This is collection of machine learning cheat sheet 
Scikit-learn (sklearn) is a popular Python library for machine learning and data analysis. It provides a wide range of functions and classes for various machine learning tasks, including classification, regression, clustering, dimensionality reduction, and more. Here's a list of some commonly used functions and classes in scikit-learn:

1. **Model Selection:**
   - `train_test_split`: Splits the dataset into training and testing sets.
   - `cross_val_score`: Performs k-fold cross-validation for model evaluation.
   - `GridSearchCV`: Conducts hyperparameter tuning using grid search.
   - `RandomizedSearchCV`: Conducts hyperparameter tuning using randomized search.

2. **Supervised Learning:**
   - `LinearRegression`: Linear regression model for regression tasks.
   - `LogisticRegression`: Logistic regression model for classification tasks.
   - `DecisionTreeClassifier` and `DecisionTreeRegressor`: Decision tree models.
   - `RandomForestClassifier` and `RandomForestRegressor`: Random forest models.
   - `SVC` and `SVR`: Support Vector Classifier and Regressor.
   - `KNeighborsClassifier` and `KNeighborsRegressor`: k-Nearest Neighbors models.
   - `GradientBoostingClassifier` and `GradientBoostingRegressor`: Gradient Boosting models.

3. **Unsupervised Learning:**
   - `KMeans`: K-Means clustering.
   - `PCA`: Principal Component Analysis for dimensionality reduction.
   - `DBSCAN`: Density-Based Spatial Clustering of Applications with Noise.
   - `IsolationForest`: Isolation Forest for anomaly detection.

4. **Preprocessing:**
   - `StandardScaler`: Standardizes features to have mean 0 and variance 1.
   - `MinMaxScaler`: Scales features to a specified range, usually [0, 1].
   - `OneHotEncoder`: Encodes categorical variables into one-hot vectors.
   - `LabelEncoder`: Encodes categorical labels into numerical values.
   - `Imputer`: Imputes missing values in datasets.

5. **Evaluation:**
   - `accuracy_score`: Computes classification accuracy.
   - `mean_squared_error`: Calculates mean squared error for regression.
   - `confusion_matrix`: Generates a confusion matrix for classification.
   - `classification_report`: Generates a classification report with various metrics.
   - `roc_curve` and `roc_auc_score`: Receiver Operating Characteristic (ROC) analysis.
<details>
      
### Underfitting and Overfitting Explanation with Bias and Variance:

### Underfitting:

1. **Brief Explanation:**
   - Occurs when the model is too simple to understand patterns in the training data.
   - The model fails to capture the complexity of the data effectively.

2. **Characteristics:**
   - Low performance on training data.
   - Low performance on testing data (poor generalization).
   - The model cannot predict new data well.

3. **Handling:**
   - Increase the complexity of the model.
   - Add more relevant features.
   - Use a more sophisticated model.

4. **Bias:**
   - High bias indicates that the model is too simple and cannot represent the underlying patterns in the data.

### Overfitting:

1. **Brief Explanation:**
   - Occurs when the model is too complex and "memorizes" the training data, including irrelevant noise and variability.
   - The model over-adapts to the training data.

2. **Characteristics:**
   - High performance on training data.
   - Low performance on testing data (poor generalization).
   - The model is too sensitive to random variability in the data.

3. **Handling:**
   - Reduce the complexity of the model.
   - Use fewer or more relevant features.
   - Apply regularization techniques.
   - Collect more training data if possible.

4. **Variance:**
   - High variance indicates that the model is too sensitive to the training data and does not generalize well to new data.

### Comparison:

- **Underfitting vs. Overfitting:**
  - Underfitting occurs when the model is too simple.
  - Overfitting occurs when the model is too complex.
  - The goal is to strike a balance to create a model that can provide good predictions on new data (generalization).

- **Bias and Variance Trade-off:**
  - The bias-variance trade-off is a fundamental concept in machine learning, emphasizing the need to balance simplicity (bias) and flexibility (variance) for optimal model performance.

- **Learning Curve:**
  - The learning curve is used to understand whether the model tends to experience underfitting or overfitting.
  - By monitoring the model's performance on both training and testing data over time, we can identify the point where the model achieves the best balance.

Understanding and addressing bias and variance issues are crucial in developing reliable and robust models. The best approach varies depending on the type of data and the machine learning task at hand.

---

### Terjemahan ke Bahasa Inggris:

### Underfitting and Overfitting Explanation with Bias and Variance:

### Underfitting:

1. **Penjelasan Singkat:**
   - Terjadi ketika model terlalu sederhana untuk memahami pola dalam data pelatihan.
   - Model gagal menangkap kompleksitas data dengan efektif.

2. **Ciri-ciri:**
   - Performa rendah pada data pelatihan.
   - Performa rendah pada data pengujian (generalisasi buruk).
   - Model tidak dapat memprediksi data baru dengan baik.

3. **Penanganan:**
   - Tingkatkan kompleksitas model.
   - Tambahkan fitur-fitur yang lebih relevan.
   - Gunakan model yang lebih canggih.

4. **Bias:**
   - Bias tinggi menunjukkan bahwa model terlalu sederhana dan tidak dapat merepresentasikan pola yang mendasari dalam data.

### Overfitting:

1. **Penjelasan Singkat:**
   - Terjadi ketika model terlalu kompleks dan "menghafal" data pelatihan, termasuk noise dan variabilitas yang tidak relevan.
   - Model terlalu beradaptasi dengan data pelatihan.

2. **Ciri-ciri:**
   - Performa tinggi pada data pelatihan.
   - Performa rendah pada data pengujian (generalisasi buruk).
   - Model terlalu peka terhadap variabilitas acak dalam data.

3. **Penanganan:**
   - Kurangi kompleksitas model.
   - Gunakan lebih sedikit atau lebih banyak fitur yang relevan.
   - Terapkan teknik regularisasi.
   - Kumpulkan lebih banyak data pelatihan jika memungkinkan.

4. **Varians:**
   - Varian tinggi menunjukkan bahwa model terlalu peka terhadap data pelatihan dan tidak generalisasi dengan baik ke data baru.

### Perbandingan:

- **Underfitting vs. Overfitting:**
  - Underfitting terjadi ketika model terlalu sederhana.
  - Overfitting terjadi ketika model terlalu kompleks.
  - Tujuan adalah mencapai keseimbangan untuk menciptakan model yang dapat memberikan prediksi yang baik pada data baru (generalisasi).

- **Keseimbangan Bias dan Varian:**
  - Keseimbangan bias-variansa adalah konsep dasar dalam pembelajaran mesin, menekankan perlunya seimbang antara kesederhanaan (bias) dan fleksibilitas (varians) untuk kinerja model yang optimal.

- **Kurva Pembelajaran:**
  - Kurva pembelajaran digunakan untuk memahami apakah model cenderung mengalami underfitting atau overfitting.
  - Dengan memonitor performa model pada data pelatihan dan pengujian seiring waktu, kita dapat mengidentifikasi titik di mana model mencapai keseimbangan terbaik.

Memahami dan menangani masalah bias dan varian sangat penting dalam pengembangan model yang andal dan kokoh. Pendekatan terbaik bervariasi tergantung pada jenis data dan tugas pembelajaran mesin yang dihadapi.
</details>

6. **Feature Selection and Extraction:**
   - `SelectKBest`: Selects the top k features based on statistical tests.
   - `PCA`: Principal Component Analysis for feature extraction.
   - `RFE` (Recursive Feature Elimination): Selects features recursively based on model performance.

7. **Ensemble Methods:**
   - `VotingClassifier` and `VotingRegressor`: Combines multiple models for ensemble learning.
   - `BaggingClassifier` and `BaggingRegressor`: Bagging ensemble methods.
   - `AdaBoostClassifier` and `AdaBoostRegressor`: Adaptive Boosting ensemble methods.

8. **Neural Network Interface:**
   - `MLPClassifier` and `MLPRegressor`: Multi-layer Perceptron models.

This is not an exhaustive list, as scikit-learn offers many more functions and classes for various machine learning tasks. You can find more details and documentation on the official scikit-learn website: https://scikit-learn.org/stable/.
