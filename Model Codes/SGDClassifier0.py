import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Train ve test datasetlerini Dask ile yükle
train_path = 'C:\\Users\\Erncl\\OneDrive\\Masaüstü\\Projects\\BELKA Discovery\\Clean Data\\train.csv'
test_path = 'C:\\Users\\Erncl\\OneDrive\\Masaüstü\\Projects\\BELKA Discovery\\Clean Data\\test.csv'

# Chunk boyutu
chunk_size = 100000

# Modeli ve scaler'ı tanımla
scaler = StandardScaler()
classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)

# OneHotEncoder'ı tanımla
onehot_encoder_molecule = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoder_protein = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# İlk chunk'ı yükleyerek encoder'ları fit et
first_chunk = pd.read_csv(train_path, chunksize=chunk_size).__next__()
molecule_columns = ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles']
onehot_encoder_molecule.fit(first_chunk[molecule_columns])
onehot_encoder_protein.fit(first_chunk[['protein_name']])

# İlk chunk'ı kullanarak scaler'ı fit et
X_train_chunk_molecule = onehot_encoder_molecule.transform(first_chunk[molecule_columns])
X_train_chunk_protein = onehot_encoder_protein.transform(first_chunk[['protein_name']])
X_train_chunk = pd.concat([pd.DataFrame(X_train_chunk_molecule), pd.DataFrame(X_train_chunk_protein)], axis=1)
scaler.fit(X_train_chunk)

# Chunk bazında veri işleme
counter = 0
for chunk in pd.read_csv(train_path, chunksize=chunk_size):
    # Feature'ları ve target'ı ayır
    print(counter)
    counter += 1
    X_train_chunk_molecule = onehot_encoder_molecule.transform(chunk[molecule_columns])
    X_train_chunk_protein = onehot_encoder_protein.transform(chunk[['protein_name']])
    X_train_chunk = pd.concat([pd.DataFrame(X_train_chunk_molecule), pd.DataFrame(X_train_chunk_protein)], axis=1)
    y_train_chunk = chunk['binds'].astype(int)

    # Transform the chunk with the fitted scaler
    X_train_chunk = scaler.transform(X_train_chunk)
    
    # Modeli her chunk ile güncelle
    classifier.partial_fit(X_train_chunk, y_train_chunk, classes=[0, 1])
print("Model trained incrementally")

# Test veri setini yükle ve işleme
test_dataset = dd.read_csv(test_path)
X_test_molecule = onehot_encoder_molecule.transform(test_dataset[molecule_columns].compute())
X_test_protein = onehot_encoder_protein.transform(test_dataset[['protein_name']].compute())
X_test = pd.concat([pd.DataFrame(X_test_molecule), pd.DataFrame(X_test_protein)], axis=1)
print("1")

# Feature scaling
X_test = scaler.transform(X_test)
print("2")

# Tahmin yap
y_pred = classifier.predict(X_test)
print("3")
print("Prediction completed")

# Test setindeki id'leri al
ids = test_dataset['id'].compute()
print("4")

# Tahmin edilen bağlanma değerlerini ve id'leri içeren bir DataFrame oluştur
results_df = pd.DataFrame({'id': ids, 'binds': y_pred})
print("5")

# Sonuçları CSV dosyasına yaz
results_df.to_csv('submission_incremental.csv', index=False)
print("6")
print("Results saved to submission_incremental.csv")
