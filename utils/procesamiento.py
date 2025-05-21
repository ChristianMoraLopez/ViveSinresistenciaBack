# -*- coding: utf-8 -*-
import pandas as pd

# Rutas a los archivos (ajusta según tu estructura)
archivo_glucosa = "../GLU_L.xpt"
archivo_insulina = "../INS_L.xpt"
archivo_demo = "../DEMO_L.xpt"
archivo_bmx = "../BMX_L.xpt"

# Leer los archivos xpt
glucosa = pd.read_sas(archivo_glucosa)
insulina = pd.read_sas(archivo_insulina)
demografia = pd.read_sas(archivo_demo)
bmx = pd.read_sas(archivo_bmx)

# Unir DataFrames por SEQN (ID)
df = demografia.merge(glucosa, on="SEQN")\
               .merge(insulina, on="SEQN")\
               .merge(bmx, on="SEQN")

# Seleccionar solo columnas relevantes para el análisis
df = df[[
    'SEQN',
    'RIAGENDR',    # Género
    'RIDAGEYR',    # Edad en años
    'LBXGLU',      # Glucosa (mg/dL)
    'LBXIN',       # Insulina (uU/mL)
    'BMXBMI',      # Índice de masa corporal (IMC)
    'BMXWAIST'     # Circunferencia de cintura (cm)
]]

# Eliminar filas con datos faltantes
df = df.dropna()

# Calcular HOMA-IR = (Glucosa * Insulina) / 405
df["HOMA_IR"] = (df["LBXGLU"] * df["LBXIN"]) / 405

# Etiqueta binaria: 1 = resistencia a la insulina (HOMA-IR >= 2.5), 0 = no resistencia
df["resistencia_insulina"] = df["HOMA_IR"].apply(lambda x: 1 if x >= 2.5 else 0)

# Guardar CSV listo para modelar
df.to_csv("../datos_calibrados.csv", index=False, encoding="utf-8")

# Mostrar primeras filas para verificar
print(df.head())
