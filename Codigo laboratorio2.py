# Importar librerías

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Carga de Datos
data = pd.read_csv('train.csv')  # Asegúrate de ajustar la ruta y el nombre del archivo

# Visualizar las primeras filas del DataFrame
print(data.head())

# 2. Exploración inicial de datos
print("Información general del conjunto de datos:")
print(data.info())

# 2. Split de los datos
# División de datos en 80% entrenamiento y 20% prueba
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Imprimir información sobre las formas de los conjuntos de entrenamiento y prueba
print("Forma del conjunto de entrenamiento:", train_data.shape)
print("Forma del conjunto de prueba:", test_data.shape)

# EDA básico
# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(data.describe())

# Histograma de la variable objetivo (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], bins=30, kde=True, color='blue')
plt.title('Distribución de SalePrice')
plt.xlabel('SalePrice')
plt.show()

# Relación entre variables (ajusta las variables según tus necesidades)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=data, color='coral')
plt.title('Relación entre GrLivArea y SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# Matriz de correlación
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Pipeline para la ingeniería de características
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Imputación de variables numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Imputación de variables categóricas y codificación
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Modelo para selección de características basado en Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
feature_selector = SelectFromModel(model)

# Eliminación de outliers con Isolation Forest
outlier_detector = IsolationForest(contamination=0.05)

# Crear pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('outlier_detector', outlier_detector),
    ('feature_selector', feature_selector)
])

# Aplicar el pipeline a los datos de entrenamiento
X_train_processed = pipeline.fit_transform(X_train, y_train)

# Visualizar las características seleccionadas (opcional)
selected_features = X_train.columns[feature_selector.get_support()]
print("Características seleccionadas:", selected_features)

# 6. Entrenamiento y selección de modelos automática. 

# Configuración de PyCaret
exp1 = setup(data, target='SalePrice', session_id=42, numeric_features=['feature1', 'feature2'], ordinal_features={'feature3': ['low', 'medium', 'high']})

# Comparación de modelos
best_models = compare_models()

# Seleccionar los tres mejores modelos
top3_models = [tune_model(model) for model in best_models[:3]]

# Visualizar la importancia de características y métricas para los tres modelos
for model in top3_models:
    # Gráfica de importancia de características
    plot_model(model, plot='feature')

    # Métricas resultantes
    evaluate_model(model)

# R2 (Coeficiente de determinación)
r2 = r2_score(y_true, y_pred)

# RMSE (Error cuadrático medio)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MSE (Error cuadrático medio)
mse = mean_squared_error(y_true, y_pred)

# MAPE (Error porcentual absoluto medio)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Imprimir las métricas
print(f'R2: {r2}')
print(f'RMSE: {rmse}')
print(f'MSE: {mse}')
print(f'MAPE: {mape}%')