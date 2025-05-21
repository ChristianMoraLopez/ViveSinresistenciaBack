# ? API de Predicción de Resistencia a la Insulina

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Backend desarrollado con Flask para predecir la resistencia a la insulina basado en parámetros clínicos y realizar análisis de sensibilidad.

## ? Características

- Predicción de resistencia a la insulina basada en modelo ML
- Cálculo automático de HOMA-IR y clasificación de riesgo
- Análisis de sensibilidad para variables críticas (glucosa, IMC, cintura)
- API RESTful con endpoints optimizados

## ? Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/prediccion-resistencia-insulina.git
   cd prediccion-resistencia-insulina
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   ```

3. Activar el entorno virtual:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ? Uso

1. Asegúrate de tener los modelos necesarios en la carpeta `models/`:
   - `modelo_resistencia.pkl` - Modelo entrenado
   - `scaler_resistencia.pkl` - Scaler para normalización

2. Ejecutar la aplicación:
   ```bash
   python app.py
   ```

3. La API estará disponible en: `http://localhost:5000`

## ? Endpoints

### Verificación de estado
```
GET /api/health
```
Respuesta: `{"status": "ok"}` si la API está funcionando correctamente.

### Predicción de resistencia a la insulina
```
POST /api/predict
```

**Parámetros (JSON)**:
```json
{
  "genero": 1,       // 1=masculino, 2=femenino
  "edad": 35.0,
  "glucosa": 100.0,
  "insulina": 15.0,
  "imc": 27.5,
  "cintura": 95.0
}
```

### Análisis de sensibilidad
```
POST /api/analisis-sensibilidad
```

Realiza un análisis de sensibilidad generando matrices de probabilidades variando glucosa, IMC y cintura. Acepta los mismos parámetros que el endpoint de predicción.

## ? Ejemplo de uso con curl

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"genero":1,"edad":35,"glucosa":100,"insulina":15,"imc":27.5,"cintura":95}'
```

## ? Notas técnicas

- El cálculo de HOMA-IR se realiza automáticamente en el endpoint `/api/predict`
- La clasificación de riesgo se determina según valores clínicos establecidos
- El análisis de sensibilidad devuelve matrices con probabilidades de riesgo para diferentes valores de:
  - Glucosa
  - IMC
  - Perímetro de cintura

## ? Licencia

[MIT](LICENSE)

---

Desarrollado con ?? por [Tu Nombre/Organización]