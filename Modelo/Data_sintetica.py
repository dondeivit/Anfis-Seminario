import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

df_real_train = pd.read_csv("datos_reales_entrenamiento.csv")  # <- nombre del archivo de entrenamiento
data_real = df_real_train.to_numpy()
# Entrenar modelo GMM (puedes ajustar el número de componentes si deseas)
gmm = GaussianMixture(n_components=1, random_state=0)
gmm.fit(data_real)

# Generar muchas muestras sintéticas (por ejemplo, 1000)
synthetic_samples, _ = gmm.sample(1000)

# Convertir a DataFrame
df_synth = pd.DataFrame(synthetic_samples, columns=['tiempo', 'voltaje', 'catalizador', 'hidrogeno'])

# Filtrar por rangos físicos realistas
df_synth = df_synth[
    (df_synth['tiempo'].between(10, 30)) &
    (df_synth['voltaje'].between(2.4, 3.0)) &
    (df_synth['catalizador'].between(5, 15))
]

# Redondear un poco los valores para evitar decimales excesivos
df_synth = df_synth.round({'tiempo': 2, 'voltaje': 3, 'catalizador': 2, 'hidrogeno': 2})

# Guardar a CSV si lo deseas
df_synth.to_csv("datos_sinteticos_filtrados.csv", index=False)

print(f"Se generaron {len(df_synth)} muestras sintéticas válidas.")

# Cargar datos reales como DataFrame
df_real = pd.DataFrame(data_real, columns=['tiempo', 'voltaje', 'catalizador', 'hidrogeno'])

# Comparar histogramas
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df_synth['tiempo'], bins=20, color='skyblue', ax=axs[0, 0], kde=True)
sns.histplot(df_real['tiempo'], color='red', kde=True, ax=axs[0, 0], label="Real", alpha=0.5)
axs[0, 0].set_title("Distribución del Tiempo")

sns.histplot(df_synth['voltaje'], bins=20, color='skyblue', ax=axs[0, 1], kde=True)
sns.histplot(df_real['voltaje'], color='red', kde=True, ax=axs[0, 1], label="Real", alpha=0.5)
axs[0, 1].set_title("Distribución del Voltaje")

sns.histplot(df_synth['catalizador'], bins=20, color='skyblue', ax=axs[1, 0], kde=True)
sns.histplot(df_real['catalizador'], color='red', kde=True, ax=axs[1, 0], label="Real", alpha=0.5)
axs[1, 0].set_title("Distribución del Catalizador")

sns.histplot(df_synth['hidrogeno'], bins=20, color='skyblue', ax=axs[1, 1], kde=True)
sns.histplot(df_real['hidrogeno'], color='red', kde=True, ax=axs[1, 1], label="Real", alpha=0.5)
axs[1, 1].set_title("Distribución del Hidrógeno")

plt.tight_layout()
plt.legend()
plt.show()


# Matriz de correlación
sns.heatmap(df_synth.corr(), annot=True, cmap="coolwarm")
plt.title("Correlación entre variables (datos sintéticos)")
plt.show()

# Scatter matrix
sns.pairplot(df_synth)
plt.suptitle("Relaciones entre variables (datos sintéticos)", y=1.02)
plt.show()