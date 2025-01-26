import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# Train datasetini Dask ile yükle
train_dataset = dd.read_csv('2encoded_trainSimplified_noMolecules++_merged.csv')
# Feature'ları ve target'ı ayır
X_train = train_dataset.iloc[:, [1, 2, 3, 5, 6, 7]]
y_train = train_dataset.iloc[:, 4].astype(int)

print("Train loaded")

# Test datasetini Dask ile yükle
test_dataset = dd.read_csv('2encoded_testSimplified+.csv')
# Feature'ları ayır
X_test = test_dataset.iloc[:, [1, 2, 3, 4, 5, 6]]

print("Test loaded")

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling done.")

# Convert Dask arrays to NumPy arrays
X_train = X_train.compute()
X_test = X_test.compute()
y_train = y_train.compute()

# Random Forest Classifier modelini Scikit-learn ile eğit
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)

print("Model trained")

# Test verileriyle tahmin yap
y_pred = classifier.predict(X_test)

print("Prediction completed")

# Test setindeki id'leri al
ids = test_dataset.iloc[:, 0].compute()

# Tahmin edilen bağlanma değerlerini ve id'leri içeren bir DataFrame oluştur
results_df = pd.DataFrame({'id': ids, 'binds': y_pred})

# Sonuçları CSV dosyasına yaz
results_df.to_csv('submission_1.csv', index=False)