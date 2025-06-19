from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Datos reales
real_data = np.array([
    [30, 2.7, 15, 79.2],
    [20, 2.7, 10, 53.8],
    [30, 2.7, 5, 78.4],
    [20, 2.7, 10, 53.8],
    [30, 3.0, 10, 104.1],
    [30, 2.4, 10, 58.8],
    [20, 2.7, 10, 53.8],
    [20, 3.0, 5, 68.6],
    [20, 2.7, 10, 53.8],
    [10, 2.7, 5, 26.1],
    [20, 2.7, 10, 53.8],
    [10, 3.0, 10, 34.9],
    [20, 3.0, 15, 69.3],
    [20, 2.4, 15, 39.2],
    [10, 2.7, 15, 26.5],
    [20, 2.4, 5, 39.1],
    [10, 2.4, 10, 19.8]
])

# Split reproducible
train_data, val_data = train_test_split(real_data, test_size=0.3, random_state=42)

# Guardar
pd.DataFrame(train_data, columns=['tiempo', 'voltaje', 'catalizador', 'hidrogeno']).to_csv("datos_reales_entrenamiento.csv", index=False)
pd.DataFrame(val_data, columns=['tiempo', 'voltaje', 'catalizador', 'hidrogeno']).to_csv("datos_reales_validacion.csv", index=False)
