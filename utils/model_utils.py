# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Rutas
ruta = "../"
archivo_glucosa = ruta + "GLU_L.xpt"
archivo_insulina = ruta + "INS_L.xpt"
archivo_demo = ruta + "DEMO_L.xpt"
archivo_bmx = ruta + "BMX_L.xpt"

# Cargar datos
glucosa = pd.read_sas(archivo_glucosa)
insulina = pd.read_sas(archivo_insulina)
demografia = pd.read_sas(archivo_demo)
bmx = pd.read_sas(archivo_bmx)

# Merge datasets en base a SEQN
df = demografia.merge(glucosa, on="SEQN").merge(insulina, on="SEQN").merge(bmx, on="SEQN")

# Seleccionar columnas relevantes
df = df[[
    'SEQN', 'RIAGENDR', 'RIDAGEYR', 'LBXGLU', 'LBXIN', 'BMXBMI', 'BMXWAIST'
]]

# Eliminar filas con datos faltantes
df = df.dropna()

# Calcular HOMA-IR
df['HOMA_IR'] = (df['LBXGLU'] * df['LBXIN']) / 405

# Etiqueta de resistencia a la insulina
df['resistencia_insulina'] = df['HOMA_IR'].apply(lambda x: 1 if x >= 2.5 else 0)

# Variables predictoras y target
X = df[['RIAGENDR', 'RIDAGEYR', 'LBXGLU', 'LBXIN', 'BMXBMI', 'BMXWAIST']]
y = df['resistencia_insulina']

# Separar en entrenamiento y calibración (80%-20%)
X_train, X_calib, y_train, y_calib = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_calib_scaled = scaler.transform(X_calib)

# Entrenar modelo (Random Forest)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train_scaled, y_train)

# Guardar modelo y scaler para usar en la web
joblib.dump(modelo, ruta + '/models/modelo_resistencia.pkl')
joblib.dump(scaler, ruta + '/models/scaler_resistencia.pkl')

# Guardar set de calibración (escalado)
df_calib = pd.DataFrame(X_calib_scaled, columns=X.columns)
df_calib['resistencia_insulina'] = y_calib.values
df_calib.to_csv(ruta + "datos_calibracion.csv", index=False)

print("Modelo entrenado, scaler y datos de calibración guardados correctamente.")
