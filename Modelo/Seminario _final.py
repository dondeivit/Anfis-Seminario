# ================================================
# 1. Importar librerías necesarias
# ================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from anfis import ANFIS, predict
from membership import membershipfunction

# ================================================
# 2. Cargar datasets desde CSVs
# ================================================
# - Entrenamiento: datos reales (70%)
# - Validación: datos reales (30%)
# - Sintéticos: generados con GMM solo desde el 70%
df_train = pd.read_csv("datos_reales_entrenamiento.csv")
df_test = pd.read_csv("datos_reales_validacion.csv")
df_synth = pd.read_csv("datos_sinteticos_filtrados.csv")
#df_synth = pd.read_csv("datos_sinteticos_sin_filtrar.csv")

# ================================================
# 3. Separar inputs (X) y output (y)
# ================================================
X_train_real = df_train[['tiempo', 'voltaje', 'catalizador']].values
y_train_real = df_train[['hidrogeno']].values

X_test_real = df_test[['tiempo', 'voltaje', 'catalizador']].values
y_test_real = df_test[['hidrogeno']].values

X_synth = df_synth[['tiempo', 'voltaje', 'catalizador']].values
y_synth = df_synth[['hidrogeno']].values

# ================================================
# 4. Normalizar datos con escaladores basados SOLO en el 70%
# ================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_real)
y_train_scaled = scaler_y.fit_transform(y_train_real)

X_test_scaled = scaler_X.transform(X_test_real)
y_test_scaled = scaler_y.transform(y_test_real)

X_synth_scaled = scaler_X.transform(X_synth)
y_synth_scaled = scaler_y.transform(y_synth)

# ================================================
# 5. Combinar datos reales de entrenamiento con los sintéticos
# ================================================
X_augmented = np.vstack((X_train_scaled, X_synth_scaled))
y_augmented = np.vstack((y_train_scaled, y_synth_scaled))

print("Tamaño del conjunto de entrenamiento aumentado:", X_augmented.shape)

# ================================================
# 6. Definir funciones de pertenencia para ANFIS
# ================================================
# - 3 para 'tiempo': corto, medio, largo
# - 2 para 'voltaje': bajo, alto
# - 3 para 'catalizador': bajo, medio, alto
mf = [
    [['gaussmf', {'mean': 10, 'sigma': 5}],
     ['gaussmf', {'mean': 20, 'sigma': 5}],
     ['gaussmf', {'mean': 30, 'sigma': 5}]],
    
    [['gaussmf', {'mean': 2.4, 'sigma': 0.2}],
     ['gaussmf', {'mean': 3.0, 'sigma': 0.2}]],
    
    [['gaussmf', {'mean': 5, 'sigma': 3}],
     ['gaussmf', {'mean': 10, 'sigma': 3}],
     ['gaussmf', {'mean': 15, 'sigma': 3}]]
]
mfc = membershipfunction.MemFuncs(mf)

# ================================================
# 7. Entrenar el modelo ANFIS con los datos aumentados
# ================================================
anfis_model = ANFIS(X_augmented, y_augmented.flatten(), mfc)
anfis_model.trainHybridJangOffLine(epochs=20)

# ================================================
# 8. Validar el modelo con el 30% original (no visto)
# ================================================
# Predicciones
y_pred_scaled = predict(anfis_model, X_test_scaled)

# Desnormalizar resultados
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_real = scaler_y.inverse_transform(y_test_scaled).flatten()

# Calcular métricas de evaluación
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)

print("\n=== Validación con 30% de datos reales NO vistos ===")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Mostrar comparaciones entre predicho y real
print("\nValores reales vs. predichos:")
for real, pred in zip(y_real, y_pred):
    print(f"Real: {real:.2f} - Predicho: {pred:.2f}")

# ================================================
# 9. (Opcional) Graficar resultados
# ================================================
plt.figure(figsize=(10, 5))
plt.plot(y_real, label='Real', marker='o')
plt.plot(y_pred, label='Predicho', marker='x')
plt.title("Comparación: Real vs Predicho")
plt.xlabel("Muestra")
plt.ylabel("Hidrógeno")
plt.legend()
plt.grid(True)
plt.show()
