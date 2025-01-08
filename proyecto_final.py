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

import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


