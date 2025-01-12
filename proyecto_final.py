'''
# Condiciones de la asignación principal

Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

### Servicios de Interconnect

Interconnect proporciona principalmente dos tipos de servicios:

1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.

Algunos otros servicios que ofrece la empresa incluyen:

- Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
- Una línea de soporte técnico (*SoporteTécnico*).
- Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
- Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)

La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.

### Descripción de los datos

Los datos consisten en archivos obtenidos de diferentes fuentes:

- `contract.csv` — información del contrato;
- `personal.csv` — datos personales del cliente;
- `internet.csv` — información sobre los servicios de Internet;
- `phone.csv` — información sobre los servicios telefónicos.

En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente. La información del contrato es válida a partir del 1 de febrero de 2020.

### Datos

[final_provider.zip](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/94210e31-fd3d-451b-a350-4a8476756413/final_provider.zip)

Los datos también se encuentran en la plataforma, en la carpeta `/datasets/final_provider/`.


# 1. Plan de trabajo

Deberás realizar un análisis exploratorio de datos. Al final de *Jupyter Notebook*, escribe:

- Una lista de preguntas aclaratorias.
- Un plan aproximado para resolver la tarea, que especifica de 3 a 5 pasos básicos y los explica en uno o dos enunciados

El líder del equipo revisará tus preguntas y plan de trabajo. Las preguntas serán respondidas durante una videollamada. El código será revisado por el líder del equipo solo si hay algunas dudas.

'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score)
from sklearn.preprocessing import StandardScaler


# Ruta al archivo zip cargado y al directorio de extracción
zip_file_path = 'datasets/final_provider.zip'
extraction_dir = 'datasets/final_provider/'

# Extrae el archivo zip
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# Enumere los archivos extraídos para confirmar la extracción
os.listdir(extraction_dir)

# Ruta al directorio principal de datos
data_dir = os.path.join(extraction_dir, 'final_provider')

# Listar los archivos en el directorio principal.
os.listdir(data_dir)

# Cargar y mostrar las primeras filas de cada archivo
files = ['personal.csv', 'contract.csv', 'phone.csv', 'internet.csv']
dataframes = {}

for file in files:
    file_path = os.path.join(data_dir, file)
    dataframes[file] = pd.read_csv(file_path)
    print(f"Preview of {file}:")
    print(dataframes[file].head(), dataframes[file].info())
    print("\n")


    # Verificar valores nulos y tipos de datos
for file, df in dataframes.items():
    print(f"Análisis de {file}:")
    print(df.isnull().sum())
    print("\n")


# Limpieza preliminar de datos
# Convertir 'TotalCharges' a numérico y manejar errores
contract_df = dataframes['contract.csv']
contract_df['TotalCharges'] = pd.to_numeric(contract_df['TotalCharges'], errors='coerce')


# Reemplazar nulos resultantes de la conversión
contract_df['TotalCharges'].fillna(contract_df['MonthlyCharges'], inplace=True)


# Comprobar valores únicos en columnas categóricas clave
for col in ['gender', 'Partner', 'Dependents']:
    print(f"Valores únicos en {col}:", dataframes['personal.csv'][col].unique())


# Unificar los conjuntos de datos en un único DataFrame
merged_data = dataframes['personal.csv']
for file in ['contract.csv', 'phone.csv', 'internet.csv']:
    merged_data = merged_data.merge(dataframes[file], on='customerID', how='left')


# Verificar si hay valores nulos tras la unificación
print("Valores nulos tras la unificación:")
print(merged_data.isnull().sum())


# Análisis exploratorio
# Estadísticas descriptivas para variables numéricas
print("Estadísticas descriptivas de variables numéricas:")
print(merged_data.describe())


# Distribución de la variable objetivo (EndDate)
print("Distribución de 'EndDate':")
print(merged_data['EndDate'].value_counts())


# Distribución de clientes según tipo de servicio de internet
print("Distribución de 'InternetService':")
print(merged_data['InternetService'].value_counts())


'''
Lista de Preguntas Aclaratorias

¿Cuál es la interpretación específica de "EndDate" como característica objetivo? ¿Deberíamos codificarla como una variable binaria para modelado?

Algunos clientes no tienen servicios de Internet o teléfono. ¿Debemos tratarlos de manera especial en el análisis o eliminar esas observaciones?

¿Qué significa un valor faltante en columnas como InternetService o MultipleLines tras la unificación?

¿Se debe realizar algún tratamiento particular para las columnas categóricas de múltiples niveles como PaymentMethod?

'''


'''
Plan Aproximado

1.- Limpieza y Preprocesamiento de Datos:

Convertir columnas numéricas mal tipificadas (como TotalCharges).
Manejar valores nulos de forma adecuada según su naturaleza.

2.- Análisis Exploratorio de Datos (EDA):

Analizar las distribuciones de las variables principales.
Explorar la relación entre EndDate y otras variables.
Identificar correlaciones y posibles características relevantes.

3.- Codificación y Transformación de Datos:

Convertir variables categóricas a formato numérico (one-hot encoding o label encoding).
Escalar las variables numéricas si es necesario para los modelos.

4.- Preparación del Conjunto de Datos para Modelado:

Dividir los datos en conjuntos de entrenamiento, validación y prueba.
Garantizar un balance adecuado en la variable objetivo si está desbalanceada.

5.- Entrenamiento y Validación de Modelos:

Probar modelos básicos (logística, árboles de decisión).
Ajustar hiperparámetros y optimizar para maximizar el AUC-ROC.

'''


# 1. Limpieza y tratamiento de valores nulos
# Asegurarnos de que no queden valores nulos en 'TotalCharges' y reemplazarlos
merged_data['TotalCharges'] = pd.to_numeric(merged_data['TotalCharges'], errors='coerce')
merged_data['TotalCharges'].fillna(merged_data['MonthlyCharges'], inplace=True)

# Rellenar valores faltantes en columnas categóricas con "No"
categorical_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']

for col in categorical_columns:
    merged_data[col].fillna('No', inplace=True)

# Verificar si quedan valores nulos
print("Valores nulos restantes:")
print(merged_data.isnull().sum())

# 2. Conversión y codificación de datos categóricos
# Convertir columnas categóricas binarias a formato numérico
binary_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling']
binary_mapping = {'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1}
for col in binary_columns:
    merged_data[col] = merged_data[col].map(binary_mapping)

# Codificar variables categóricas con múltiples niveles usando OneHotEncoder
multi_level_columns = ['PaymentMethod', 'Type', 'InternetService']
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = pd.DataFrame(
    encoder.fit_transform(merged_data[multi_level_columns]),
    columns=encoder.get_feature_names_out(multi_level_columns)
)

# Agregar las características codificadas y eliminar las originales
merged_data = pd.concat([merged_data.drop(columns=multi_level_columns), encoded_features], axis=1)

# 3. Escalado de variables numéricas
# Escalar 'MonthlyCharges' y 'TotalCharges'
scaler = StandardScaler()
merged_data[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(merged_data[['MonthlyCharges', 'TotalCharges']])

# Verificar el DataFrame procesado
print("Vista previa de los datos procesados:")
print(merged_data.head())


'''
Limpieza de Datos:

Se corrigieron los valores no numéricos en TotalCharges y se manejaron los valores nulos con valores predeterminados.
Se rellenaron columnas categóricas con valores faltantes asignándoles "No", lo cual es razonable para estas variables de servicio.
Codificación de Datos:

Las columnas binarias se mapearon directamente a valores 0 y 1.
Las columnas con múltiples niveles se codificaron con OneHotEncoder, eliminando una categoría para evitar multicolinealidad.
Escalado:

Las variables numéricas MonthlyCharges y TotalCharges se escalaron usando StandardScaler para que tengan una media de 0 y una desviación estándar de 1, lo cual beneficia a ciertos modelos que son sensibles a la escala.
'''


'''
Siguientes pasos:

Dividir los datos en conjuntos de entrenamiento, validación y prueba.  
Realizar un análisis inicial de las características para determinar su correlación con el objetivo (churn).
Entrenar un modelo base para evaluar métricas preliminares.
'''


# 1. Preparar la columna objetivo
# Transformar 'EndDate' en una variable binaria
merged_data['Churn'] = (merged_data['EndDate'] == 'No').astype(int)


# Verificar la distribución del objetivo
print("\nDistribución de la variable objetivo (Churn):")
print(merged_data['Churn'].value_counts(normalize=True))


# Definir la característica objetivo
target = 'Churn'
X = merged_data.drop(columns=['EndDate', target])  # Eliminamos 'EndDate' ya que se convierte en el objetivo
y = merged_data[target]


# 2. Dividir datos en entrenamiento, validación y prueba
# Dividimos el conjunto en entrenamiento+validación (80%) y prueba (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Dividimos el conjunto de entrenamiento+validación en entrenamiento (70%) y validación (30%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val)


#Idenftificar columnas no numéricas
non_numeric_columns = X_train.select_dtypes(include=['object']).columns
print("Columnas no numéricas:", non_numeric_columns)


# Revisar los valores unicos de las columnas problematicas
for column in non_numeric_columns:
    print(f"Valores únicos en {column}: {X_train[column].unique()}")


encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[non_numeric_columns]))
X_train_encoded.index = X_train.index  # Ajustar el índice


# Combinar columnas codificadas con las demás numéricas
X_train = X_train.drop(non_numeric_columns, axis=1)
X_train = pd.concat([X_train, X_train_encoded], axis=1)


# Verificar tamaños de los conjuntos
print(f"\nTamaños de los conjuntos:")
print(f"Entrenamiento: {X_train.shape}, Validación: {X_val.shape}, Prueba: {X_test.shape}")


'''
Es necesario identificar las columnas catégoricas y excluirlas antes de calcular la matriz de correlación.
'''


# 3. Análisis inicial de características
# Convertir la columna objetivo a numérica si es necesario
merged_data[target] = merged_data[target].astype(int)


# Identificar columnas categóricas y eliminarlas para la correlación
categorical_columns = merged_data.select_dtypes(include=['object']).columns
merged_data_numeric = merged_data.drop(columns=categorical_columns)


# Asegurar que 'EndDate' no está en los datos
if 'EndDate' in merged_data_numeric.columns:
    merged_data_numeric.drop(columns=['EndDate'], inplace=True)


# Calcular la correlación entre las variables numéricas
correlation = merged_data_numeric.corr()


# Mostrar correlación con el objetivo
print("\nCorrelación con el objetivo (Churn):")
print(correlation[target].sort_values(ascending=False))


# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de correlación de las características")
plt.show()


# Distribución del objetivo
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="viridis", hue=y, legend=False)
#sns.countplot(x=y, palette="viridis")
plt.title("Distribución de la variable objetivo (Churn)")
plt.xlabel("Churn")
plt.ylabel("Número de clientes")
plt.show()


'''
Se procede con el entrenamiento de un modelo base. Utilizando una regresión logística, ya que es simple, interpretable y adecuada para establecer métricas iniciales. 
Luego se evaluará el desempeño del modelo utilizando AUC-ROC y precisión.
'''


# 4. Escalado de las características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# 5. Entrenar el modelo base (Regresión Logística)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)


# 6. Predicciones y evaluación
y_val_pred = model.predict(X_val_scaled)
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]


# Cálculo de métricas
auc_roc = roc_auc_score(y_val, y_val_proba)
accuracy = accuracy_score(y_val, y_val_pred)


# Mostrar resultados
print(f"\nAUC-ROC en conjunto de validación: {auc_roc:.3f}")
print(f"Precisión en conjunto de validación: {accuracy:.3f}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_val, y_val_pred))
print("\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred))


'''
La distribución de la variable objetivo muestra un claro desbalance de clases, con el 73.46% de los datos pertenecientes a la clase 1 (probablemente clientes que cancelaron) y solo el 26.54% a la clase 0. 
Este desequilibrio debe abordarse antes de entrenar el modelo para mejorar su desempeño.
'''

'''
Se implementa la ponderación de clases para manejar el desbalance de las clases. 
La librería LogisticRegression de scikit-learn permite configurar el parámetro class_weight en 'balanced' para ajustar automáticamente los pesos de las clases en función de su frecuencia.
'''


# 5. Entrenar el modelo base con ponderación de clases (Regresión Logística)
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)


# 6. Predicciones y evaluación
y_val_pred = model.predict(X_val_scaled)
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]


# Cálculo de métricas
auc_roc = roc_auc_score(y_val, y_val_proba)
accuracy = accuracy_score(y_val, y_val_pred)


# Mostrar resultados
print(f"\nAUC-ROC en conjunto de validación: {auc_roc:.3f}")
print(f"Precisión en conjunto de validación: {accuracy:.3f}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_val, y_val_pred))
print("\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred))


