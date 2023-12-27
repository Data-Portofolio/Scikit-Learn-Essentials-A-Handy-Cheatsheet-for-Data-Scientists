# Scikit-Learn Essentials: A Handy Cheatsheet for Data Scientists

<details>
   <summary>📈 Supervised & Unsupervised Learning </summary>
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
      <details align="center">
        <summary>📐 Feature Scaling</summary>
        <br>
         
       ![feature scaling](https://github.com/Data-Portofolio/Scikit-Learn-Essentials-A-Handy-Cheatsheet-for-Data-Scientists/assets/133883292/1f4caa61-d53e-458b-89df-246404b81dc2)

   - `OneHotEncoder`: Encodes categorical variables into one-hot vectors.
   - `LabelEncoder`: Encodes categorical labels into numerical values.
   - `Imputer`: Imputes missing values in datasets.

5. **Evaluation:**
   - `accuracy_score`: Computes classification accuracy.
   - `mean_squared_error`: Calculates mean squared error for regression.
   - `confusion_matrix`: Generates a confusion matrix for classification.
   - `classification_report`: Generates a classification report with various metrics.
   - `roc_curve` and `roc_auc_score`: Receiver Operating Characteristic (ROC) analysis.
<details align='center'>
      <summary>🔥 Underfitting and Overfitting </summary>
   
### Underfitting and Overfitting Explanation with Bias and Variance

## Underfitting:

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

## Overfitting

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

## Comparison

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

 <details align='center'>
    <summary>🏹 Example:
       </summary>
       <br>
    
 ```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load iris dataset as an example
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can be beneficial for some methods)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. SelectKBest: Select the top k features based on univariate statistical tests
k_best_selector = SelectKBest(k=2)
X_train_kbest = k_best_selector.fit_transform(X_train, y_train)
X_test_kbest = k_best_selector.transform(X_test)

# 2. SelectPercentile: Select the top features based on a percentile of the highest scores
percentile_selector = SelectPercentile(percentile=50)
X_train_percentile = percentile_selector.fit_transform(X_train, y_train)
X_test_percentile = percentile_selector.transform(X_test)

# 3. VarianceThreshold: Remove low-variance features
variance_threshold_selector = VarianceThreshold(threshold=0.1)
X_train_variance = variance_threshold_selector.fit_transform(X_train)
X_test_variance = variance_threshold_selector.transform(X_test)

# 4. RFE (Recursive Feature Elimination): Recursively removes the least important features
# Using a support vector machine (SVM) as the base estimator
svm = SVC(kernel="linear")
rfe_selector = RFE(estimator=svm, n_features_to_select=2)
X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
X_test_rfe = rfe_selector.transform(X_test)

# 5. SelectFromModel: Select features based on importance weights from a fitted model
# Using a random forest classifier as the base estimator
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
sfm_selector = SelectFromModel(estimator=rf_model, threshold="mean")
X_train_sfm = sfm_selector.fit_transform(X_train, y_train)
X_test_sfm = sfm_selector.transform(X_test)

# Example: Train and evaluate a classifier on the selected features
def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Evaluate classifiers on different feature sets
accuracy_kbest = train_and_evaluate(X_train_kbest, X_test_kbest, y_train, y_test)
accuracy_percentile = train_and_evaluate(X_train_percentile, X_test_percentile, y_train, y_test)
accuracy_variance = train_and_evaluate(X_train_variance, X_test_variance, y_train, y_test)
accuracy_rfe = train_and_evaluate(X_train_rfe, X_test_rfe, y_train, y_test)
accuracy_sfm = train_and_evaluate(X_train_sfm, X_test_sfm, y_train, y_test)

# Print accuracies
print("Accuracy (SelectKBest):", accuracy_kbest)
print("Accuracy (SelectPercentile):", accuracy_percentile)
print("Accuracy (VarianceThreshold):", accuracy_variance)
print("Accuracy (RFE):", accuracy_rfe)
print("Accuracy (SelectFromModel):", accuracy_sfm)
```

   ![image](https://github.com/Data-Portofolio/Scikit-Learn-Essentials-A-Handy-Cheatsheet-for-Data-Scientists/assets/133883292/310324e8-ee8c-4462-8d49-815de8fda267)
</details>

7. **Ensemble Methods:**
   - `VotingClassifier` and `VotingRegressor`: Combines multiple models for ensemble learning.
   - `BaggingClassifier` and `BaggingRegressor`: Bagging ensemble methods.
   - `AdaBoostClassifier` and `AdaBoostRegressor`: Adaptive Boosting ensemble methods.

8. **Neural Network Interface:**
   - `MLPClassifier` and `MLPRegressor`: Multi-layer Perceptron models.


