# -*- coding: utf-8 -*-
import pandas as pd

# Rutas a los archivos
archivo_glucosa = "../GLU_L.xpt"
archivo_insulina = "../INS_L.xpt"
archivo_demo = "../DEMO_L.xpt"
archivo_bmx = "../BMX_L.xpt"

# Leer los archivos xpt
glucosa = pd.read_sas(archivo_glucosa)
insulina = pd.read_sas(archivo_insulina)
demografia = pd.read_sas(archivo_demo)
bmx = pd.read_sas(archivo_bmx)

# Mostrar columnas disponibles en cada dataset
print("\nColumnas disponibles:")
print("Glucosa:", glucosa.columns)
print("Insulina:", insulina.columns)
print("Demografía:", demografia.columns)
print("Body Measures:", bmx.columns)

# Mostrar algunas filas para inspección visual
print("\nPrimeras filas de cada dataset:")

print("\nGlucosa:")
print(glucosa.head(3))

print("\nInsulina:")
print(insulina.head(3))

print("\nDemografía:")
print(demografia[['SEQN', 'RIAGENDR', 'RIDAGEYR']].head(3))

print("\nBody Measures (IMC y cintura):")
print(bmx[['SEQN', 'BMXBMI', 'BMXWAIST']].head(3))
