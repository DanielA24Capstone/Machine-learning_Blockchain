import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import hashlib

# Descargar recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el dataset
data = pd.read_csv('data_emails_517k_prueba.csv', sep=';')  # Ajusta el separador si es necesario
data_sample = data.sample(n=20000, random_state=42)  # Tomar muestra de 20,000 correos

# Preprocesar texto
def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Convertir a minúsculas
    words = [word for word in words if word.isalnum()]  # Eliminar caracteres no alfanuméricos
    words = [word for word in words if word not in stop_words]  # Eliminar stopwords
    return ' '.join(words)

data_sample['processed_text'] = data_sample['Column2'].apply(preprocess_text)

# Etiquetado: Crear la columna 'is_smishing'
data_sample['is_smishing'] = data_sample['Column2'].apply(
    lambda x: 1 if 'urgent' in x.lower() or 'verify' in x.lower() else 0
)

# Verificar si hay al menos dos clases
if data_sample['is_smishing'].nunique() == 1:
    raise ValueError("El conjunto de datos tiene solo una clase. Necesitas al menos dos clases para entrenar el modelo.")

# Dividir datos en entrenamiento y prueba
X = data_sample['processed_text']
y = data_sample['is_smishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir texto a matriz TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Manejo del desbalanceo de clases con ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_tfidf, y_train)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train_resampled, y_train_resampled)

# Predicción para todo el dataset
X_tfidf = vectorizer.transform(data_sample['processed_text'])
y_pred_proba = svm_model.predict_proba(X_tfidf)[:, 1]

# Ajustar el umbral de clasificación (0.1 en este caso)
threshold = 0.1
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Crear un DataFrame con los resultados
results_df = data_sample[['Column2']].copy()
results_df['Predicted_Label'] = y_pred_adjusted

# Generar hashes para cada registro
results_df['Data_Hash'] = results_df.apply(
    lambda row: hashlib.sha256(str(row['Column2']).encode()).hexdigest(), axis=1
)

# Guardar los resultados en un archivo CSV
results_df[['Data_Hash', 'Predicted_Label']].to_csv('hashed_results_full.csv', index=False)

print("Resultados guardados en 'hashed_results_full.csv'.")
