import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import ADASYN  # Reemplazamos SMOTE por ADASYN
from sklearn.metrics import classification_report
import hashlib
import matplotlib.pyplot as plt

# Descargar recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el dataset
data = pd.read_csv('data_emails_517k_prueba.csv', sep=';')  # Asegúrate de que el separador sea ';'

# Tomar una muestra aleatoria de 20,000 correos
data_sample = data.sample(n=20000, random_state=42)

# Preprocesar el texto
def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    # Tokenizar el texto
    words = word_tokenize(text.lower())  # Convertir a minúsculas
    words = [word for word in words if word.isalnum()]  # Eliminar caracteres no alfanuméricos
    words = [word for word in words if word not in stop_words]  # Eliminar stopwords
    return ' '.join(words)

# Aplicar la función de preprocesamiento al mensaje
data_sample['processed_text'] = data_sample['Column2'].apply(preprocess_text)

# Etiquetado: Asegúrate de que las etiquetas (is_smishing) existan en los datos
data_sample['is_smishing'] = data_sample['Column2'].apply(lambda x: 1 if 'urgent' in x.lower() or 'verify' in x.lower() else 0)

# Dividir los datos en entrenamiento y prueba
X = data_sample['processed_text']
y = data_sample['is_smishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir el texto a una matriz de características utilizando TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Manejo del desbalance de clases: Balancear las clases con ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_tfidf, y_train)

# Entrenar el modelo de clasificación SVM
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train_resampled, y_train_resampled)

# Calcular las probabilidades de predicción
y_pred_proba = svm_model.predict_proba(X_test_tfidf)[:, 1]

# Ajustar el umbral de clasificación
threshold = 0.1
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Evaluar el modelo con el umbral ajustado
print(classification_report(y_test, y_pred_adjusted))

# Crear un DataFrame con resultados y hashes
results_df = pd.DataFrame({
    'Original_Text': X_test,
    'Processed_Text': vectorizer.inverse_transform(X_test_tfidf),
    'Actual_Label': y_test.values,
    'Predicted_Probability': y_pred_proba,
    'Predicted_Label': y_pred_adjusted
})

# Crear un hash para cada registro
results_df['Data_Hash'] = results_df.apply(
    lambda row: hashlib.sha256(str(row.to_dict()).encode()).hexdigest(),
    axis=1
)

# Guardar solo hashes y datos clave para el blockchain
results_df[['Data_Hash', 'Predicted_Label']].to_csv('hashed_results.csv', index=False)

print("Archivo con hashes guardado en 'hashed_results.csv'")
