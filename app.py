from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address  # Asegúrate que esta importación esté presente
from dotenv import load_dotenv
import joblib
import numpy as np
import os
import sympy as sp
from sklearn.preprocessing import StandardScaler
from marshmallow import Schema, fields, ValidationError, validate
import json

# Cargar variables de entorno
load_dotenv()

# Inicializar la aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')  # Clave secreta desde .env
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Restringir orígenes en producción
Talisman(app, force_https=True, strict_transport_security=True, session_cookie_secure=True)

# --- MODIFICACIÓN AQUÍ ---
# Inicializar Flask-Limiter usando el patrón init_app
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)  # Asociar el limiter con la app


# --- FIN DE LA MODIFICACIÓN ---

# Esquema de validación para datos de entrada
class PredictionSchema(Schema):
    genero = fields.Int(required=True, validate=validate.OneOf([1, 2]))
    edad = fields.Float(required=True, validate=validate.Range(min=18, max=120))
    glucosa = fields.Float(required=True, validate=validate.Range(min=50, max=300))
    insulina = fields.Float(required=True, validate=validate.Range(min=2, max=100))
    imc = fields.Float(required=True, validate=validate.Range(min=10, max=50))
    cintura = fields.Float(required=True, validate=validate.Range(min=50, max=150))


# Cargar modelo y scaler
MODEL_PATH = os.path.join('models', 'modelo_resistencia.pkl')
SCALER_PATH = os.path.join('models', 'scaler_resistencia.pkl')

try:
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    app.logger.error(f"Error cargando modelo o scaler: {str(e)}")
    # Considera si quieres que la app falle al iniciar o manejar esto de otra forma
    # Por ahora, si el modelo no carga, la app no debería iniciar correctamente.
    raise


# Función para calcular las derivadas parciales del riesgo
def calcular_derivadas(genero, edad, glucosa, insulina, imc, cintura):
    # Asegúrate de que los símbolos x, y, z se correspondan con las variables que quieres derivar.
    # En tu implementación original, x, y, z se usan para glucosa, imc, cintura.
    x_sym, y_sym, z_sym = sp.symbols(
        'x_sym y_sym z_sym')  # Usar nombres diferentes para evitar confusión con variables locales
    alpha, beta, gamma = 0.05, 0.03, 0.02  # Ajustar según modelo entrenado
    delta, epsilon, zeta = 0.001, 0.0005, 0.0003
    # Aquí la función R debería usar las variables simbólicas
    R = alpha * x_sym + beta * y_sym + gamma * z_sym + delta * x_sym * y_sym + epsilon * x_sym * z_sym + zeta * y_sym * z_sym

    dR_dx = sp.diff(R, x_sym).subs({x_sym: glucosa, y_sym: imc, z_sym: cintura})
    dR_dy = sp.diff(R, y_sym).subs({x_sym: glucosa, y_sym: imc, z_sym: cintura})
    dR_dz = sp.diff(R, z_sym).subs({x_sym: glucosa, y_sym: imc, z_sym: cintura})
    dR_dxdy = sp.diff(sp.diff(R, x_sym), y_sym).subs({x_sym: glucosa, y_sym: imc, z_sym: cintura})
    dR_dxdz = sp.diff(sp.diff(R, x_sym), z_sym).subs({x_sym: glucosa, y_sym: imc, z_sym: cintura})
    dR_dydz = sp.diff(sp.diff(R, y_sym), z_sym).subs({x_sym: glucosa, y_sym: imc, z_sym: cintura})
    return {
        'glucosa': float(dR_dx),
        'imc': float(dR_dy),
        'cintura': float(dR_dz),
        'glucosa_imc': float(dR_dxdy),
        'glucosa_cintura': float(dR_dxdz),
        'imc_cintura': float(dR_dydz)
    }


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
@limiter.limit("100 per day")
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({'status': 'ok'}), 200


@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")  # Este decorador debe funcionar correctamente después del cambio
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    schema = PredictionSchema()
    try:
        validated_data = schema.load(data)
        genero = validated_data['genero']
        edad = validated_data['edad']
        glucosa = validated_data['glucosa']
        insulina = validated_data['insulina']
        imc = validated_data['imc']
        cintura = validated_data['cintura']
    except ValidationError as e:
        return jsonify({'error': e.messages}), 400  # e.messages es más útil que str(e.messages)

    try:
        # Calcular HOMA-IR
        homa_ir = (glucosa * insulina) / 405

        # Preparar datos para la predicción
        # Asegúrate que el orden de las características coincida con el entrenamiento del scaler y modelo
        X = np.array([[genero, edad, glucosa, insulina, imc, cintura]])
        X_scaled = scaler.transform(X)

        # Realizar predicción
        probabilidad = modelo.predict_proba(X_scaled)[0][1]
        prediccion = 1 if probabilidad >= 0.5 else 0

        # Calcular sensibilidad (derivadas parciales)
        sensibilidad = calcular_derivadas(genero, edad, glucosa, insulina, imc,
                                          cintura)  # Pasas las variables correctas

        # Categorizar el nivel de riesgo basado en HOMA-IR
        if homa_ir < 1.96:
            categoria = "Sin resistencia a la insulina"
            nivel_riesgo = "bajo"
        elif 1.96 <= homa_ir <= 3.0:  # Más explícito
            categoria = "Sospecha de resistencia a la insulina"
            nivel_riesgo = "medio"
        else:  # homa_ir > 3.0
            categoria = "Resistencia a la insulina"
            nivel_riesgo = "alto"

        # Preparar recomendaciones basadas en la sensibilidad
        recomendaciones = []
        # Ordenar por el valor absoluto de la sensibilidad podría ser más relevante si la dirección no importa
        # o si las derivadas pueden ser negativas y su magnitud es lo que cuenta.
        # Por ahora, se mantiene tu lógica original.
        sensibilidades_ordenadas = sorted(
            [("glucosa", sensibilidad["glucosa"]),
             ("imc", sensibilidad["imc"]),
             ("cintura", sensibilidad["cintura"])],
            key=lambda item: abs(item[1]),  # Considera usar abs() si la magnitud es lo importante
            reverse=True
        )

        # Umbrales para recomendaciones, podrías hacerlos configurables
        umbral_sensibilidad = 0.01  # Ejemplo, ajusta este valor

        for variable, valor_sensibilidad in sensibilidades_ordenadas:
            if abs(valor_sensibilidad) > umbral_sensibilidad:  # Chequear si la sensibilidad es significativa
                if variable == "glucosa":
                    recomendaciones.append("Priorizar el control de glucosa en sangre.")
                elif variable == "imc":
                    recomendaciones.append("Enfocarse en la reducción del IMC mediante dieta y ejercicio.")
                elif variable == "cintura":
                    recomendaciones.append("Trabajar en la reducción de la grasa abdominal.")

        if not recomendaciones:
            recomendaciones.append(
                "Mantener hábitos saludables. Todos los factores de sensibilidad están actualmente bajos.")

        return jsonify({
            'prediccion': bool(prediccion),
            'probabilidad': float(probabilidad),
            'homa_ir': float(homa_ir),
            'categoria': categoria,
            'nivel_riesgo': nivel_riesgo,
            'sensibilidad': sensibilidad,
            'recomendaciones': recomendaciones
        })

    except AttributeError as e:  # Específicamente para errores como predict_proba no existente
        app.logger.error(f"Error de atributo con el modelo o scaler: {str(e)}")
        return jsonify({'error': f'Error interno del servidor: Problema con el modelo cargado. {str(e)}'}), 500
    except ValueError as e:  # Específicamente para errores de dimensionamiento en transform o predict
        app.logger.error(f"Error de valor (posiblemente dimensiones de datos): {str(e)}")
        return jsonify(
            {'error': f'Error interno del servidor: Problema con los datos de entrada para el modelo. {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Error inesperado en predicción: {str(e)}")
        # Para el cliente, es mejor no exponer detalles internos del error.
        return jsonify({'error': 'Ocurrió un error procesando la solicitud.'}), 500


@app.route('/api/analisis-sensibilidad', methods=['POST'])
@limiter.limit("10 per minute")
def analisis_sensibilidad():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    schema = PredictionSchema()
    try:
        validated_data = schema.load(data)
        genero = validated_data['genero']
        edad = validated_data['edad']
        glucosa_base = validated_data['glucosa']
        insulina = validated_data[
            'insulina']  # Asegúrate que la insulina también se use si el modelo la necesita para predicciones
        imc_base = validated_data['imc']
        cintura_base = validated_data['cintura']
    except ValidationError as e:
        return jsonify({'error': e.messages}), 400

    try:
        # No se usa homa_ir_base en este endpoint, se puede quitar si no es necesario
        # homa_ir_base = (glucosa_base * insulina) / 405

        resultados = {
            'matriz_glucosa_imc': [],
            'matriz_glucosa_cintura': [],
            'matriz_imc_cintura': []
        }

        # Variaciones de glucosa, IMC y cintura (ej. +/- 20%)
        # Usar un rango más pequeño para 5 puntos: -20%, -10%, 0%, +10%, +20%
        variaciones_relativas = [-0.2, -0.1, 0.0, 0.1, 0.2]  # 5 puntos

        variacion_glucosa = [glucosa_base * (1 + i) for i in variaciones_relativas]
        variacion_imc = [imc_base * (1 + i) for i in variaciones_relativas]
        variacion_cintura = [cintura_base * (1 + i) for i in variaciones_relativas]

        # Matriz glucosa vs IMC
        for g_val in variacion_glucosa:
            fila = []
            for i_val in variacion_imc:
                X = np.array([[genero, edad, g_val, insulina, i_val, cintura_base]])
                X_scaled = scaler.transform(X)
                probabilidad = modelo.predict_proba(X_scaled)[0][1]
                fila.append(float(probabilidad))
            resultados['matriz_glucosa_imc'].append(fila)

        # Matriz glucosa vs cintura
        for g_val in variacion_glucosa:
            fila = []
            for c_val in variacion_cintura:
                X = np.array([[genero, edad, g_val, insulina, imc_base, c_val]])
                X_scaled = scaler.transform(X)
                probabilidad = modelo.predict_proba(X_scaled)[0][1]
                fila.append(float(probabilidad))
            resultados['matriz_glucosa_cintura'].append(fila)

        # Matriz IMC vs cintura
        for i_val in variacion_imc:
            fila = []
            for c_val in variacion_cintura:
                X = np.array([[genero, edad, glucosa_base, insulina, i_val, c_val]])
                X_scaled = scaler.transform(X)
                probabilidad = modelo.predict_proba(X_scaled)[0][1]
                fila.append(float(probabilidad))
            resultados['matriz_imc_cintura'].append(fila)

        return jsonify({
            'resultados': resultados,
            'ejes': {
                'glucosa': [float(g) for g in variacion_glucosa],
                'imc': [float(i) for i in variacion_imc],
                'cintura': [float(c) for c in variacion_cintura]
            }
        })
    except AttributeError as e:
        app.logger.error(f"Error de atributo con el modelo o scaler en análisis de sensibilidad: {str(e)}")
        return jsonify({'error': f'Error interno del servidor: Problema con el modelo cargado. {str(e)}'}), 500
    except ValueError as e:
        app.logger.error(f"Error de valor en análisis de sensibilidad: {str(e)}")
        return jsonify(
            {'error': f'Error interno del servidor: Problema con los datos de entrada para el modelo. {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Error inesperado en análisis de sensibilidad: {str(e)}")
        return jsonify({'error': 'Ocurrió un error procesando la solicitud de análisis de sensibilidad.'}), 500

if __name__ == '__main__':
    app.run(
        debug=(os.getenv('FLASK_ENV') == 'development'),
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )