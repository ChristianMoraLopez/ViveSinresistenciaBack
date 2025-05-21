import numpy as np
import sympy as sp


class ModelHelper:
    """
    Clase auxiliar para realizar cálculos matemáticos relacionados con el modelo
    de predicción de resistencia a la insulina.
    """

    def __init__(self, model, scaler):
        """
        Inicializa el helper con el modelo y el scaler

        Args:
            model: Modelo de machine learning entrenado
            scaler: Scaler para normalizar los datos
        """
        self.model = model
        self.scaler = scaler
        # Valores estimados para los coeficientes del modelo simplificado
        # Estos deberían ajustarse según los datos reales
        self.coefs = {
            'alpha': 0.05,  # Glucosa
            'beta': 0.03,  # IMC
            'gamma': 0.02,  # Cintura abdominal
            'delta': 0.001,  # Interacción glucosa-IMC
            'epsilon': 0.0005,  # Interacción glucosa-cintura
            'zeta': 0.0003  # Interacción IMC-cintura
        }

    def calcular_homa_ir(self, glucosa, insulina):
        """
        Calcula el índice HOMA-IR

        Args:
            glucosa: Nivel de glucosa en ayunas (mg/dL)
            insulina: Nivel de insulina en ayunas (μU/mL)

        Returns:
            float: Valor del índice HOMA-IR
        """
        return (glucosa * insulina) / 405

    def categorizar_riesgo(self, homa_ir):
        """
        Categoriza el nivel de riesgo basado en HOMA-IR

        Args:
            homa_ir: Valor del índice HOMA-IR

        Returns:
            dict: Categoría y nivel de riesgo
        """
        if homa_ir < 1.96:
            return {
                "categoria": "Sin resistencia a la insulina",
                "nivel_riesgo": "bajo"
            }
        elif homa_ir >= 1.96 and homa_ir <= 3.0:
            return {
                "categoria": "Sospecha de resistencia a la insulina",
                "nivel_riesgo": "medio"
            }
        else:
            return {
                "categoria": "Resistencia a la insulina",
                "nivel_riesgo": "alto"
            }

    def generar_recomendaciones(self, sensibilidad):
        """
        Genera recomendaciones basadas en el análisis de sensibilidad

        Args:
            sensibilidad: Diccionario con los valores de sensibilidad para cada variable

        Returns:
            list: Lista de recomendaciones priorizadas
        """
        recomendaciones = []
        sensibilidades_ordenadas = sorted(
            [("glucosa", sensibilidad["glucosa"]),
             ("imc", sensibilidad["imc"]),
             ("cintura", sensibilidad["cintura"])],
            key=lambda x: x[1],
            reverse=True
        )

        for variable, valor in sensibilidades_ordenadas:
            if variable == "glucosa" and valor > 0.1:
                recomendaciones.append({
                    "variable": "glucosa",
                    "mensaje": "Priorizar el control de glucosa en sangre",
                    "detalles": [
                        "Reducir el consumo de carbohidratos refinados",
                        "Aumentar el consumo de fibra",
                        "Realizar actividad física regular",
                        "Considerar consulta con endocrinólogo"
                    ]
                })
            elif variable == "imc" and valor > 0.1:
                recomendaciones.append({
                    "variable": "imc",
                    "mensaje": "Enfocarse en la reducción del IMC mediante dieta y ejercicio",
                    "detalles": [
                        "Establecer una meta de pérdida de peso gradual",
                        "Aumentar la actividad física diaria",
                        "Adoptar una dieta balanceada",
                        "Considerar apoyo profesional para plan personalizado"
                    ]
                })
            elif variable == "cintura" and valor > 0.1:
                recomendaciones.append({
                    "variable": "cintura",
                    "mensaje": "Trabajar en la reducción de la grasa abdominal",
                    "detalles": [
                        "Ejercicios específicos para la zona central",
                        "Reducir consumo de grasas saturadas y azúcares",
                        "Controlar niveles de estrés",
                        "Mejorar calidad del sueño"
                    ]
                })

        return recomendaciones

    def calcular_derivadas_parciales(self, glucosa, imc, cintura):
        """
        Calcula las derivadas parciales del modelo de riesgo

        Args:
            glucosa: Nivel de glucosa en ayunas
            imc: Índice de masa corporal
            cintura: Circunferencia abdominal

        Returns:
            dict: Derivadas parciales calculadas
        """
        # Definir variables simbólicas
        x, y, z = sp.symbols('x y z')

        # Extraer coeficientes
        alpha = self.coefs['alpha']
        beta = self.coefs['beta']
        gamma = self.coefs['gamma']
        delta = self.coefs['delta']
        epsilon = self.coefs['epsilon']
        zeta = self.coefs['zeta']

        # Función de riesgo (modelo simplificado)
        R = alpha * x + beta * y + gamma * z + delta * x * y + epsilon * x * z + zeta * y * z

        # Calcular derivadas parciales evaluadas en los valores actuales
        dR_dx = sp.diff(R, x).subs({x: glucosa, y: imc, z: cintura})
        dR_dy = sp.diff(R, y).subs({x: glucosa, y: imc, z: cintura})
        dR_dz = sp.diff(R, z).subs({x: glucosa, y: imc, z: cintura})

        # Calcular derivadas cruzadas
        dR_dxdy = sp.diff(sp.diff(R, x), y).subs({x: glucosa, y: imc, z: cintura})
        dR_dxdz = sp.diff(sp.diff(R, x), z).subs({x: glucosa, y: imc, z: cintura})
        dR_dydz = sp.diff(sp.diff(R, y), z).subs({x: glucosa, y: imc, z: cintura})

        return {
            'glucosa': float(dR_dx),
            'imc': float(dR_dy),
            'cintura': float(dR_dz),
            'glucosa_imc': float(dR_dxdy),
            'glucosa_cintura': float(dR_dxdz),
            'imc_cintura': float(dR_dydz)
        }

    def calcular_diferencial_total(self, glucosa, imc, cintura, dglucosa, dimc, dcintura):
        """
        Calcula el diferencial total del riesgo

        Args:
            glucosa: Nivel base de glucosa
            imc: Nivel base de IMC
            cintura: Nivel base de circunferencia abdominal
            dglucosa: Cambio en glucosa
            dimc: Cambio en IMC
            dcintura: Cambio en circunferencia abdominal

        Returns:
            float: Cambio estimado en el riesgo
        """
        # Obtener derivadas parciales
        derivadas = self.calcular_derivadas_parciales(glucosa, imc, cintura)

        # Calcular diferencial total
        dRiesgo = (derivadas['glucosa'] * dglucosa +
                   derivadas['imc'] * dimc +
                   derivadas['cintura'] * dcintura)

        return float(dRiesgo)

    def generar_matriz_sensibilidad(self, genero, edad, glucosa_base, insulina, imc_base,
                                    cintura_base, var1, var2, var_range=0.2, steps=5):
        """
        Genera una matriz de sensibilidad para dos variables

        Args:
            genero: Género del paciente (1=M, 2=F)
            edad: Edad del paciente
            glucosa_base: Nivel base de glucosa
            insulina: Nivel base de insulina
            imc_base: Nivel base de IMC
            cintura_base: Nivel base de circunferencia
            var1: Primera variable a variar ('glucosa', 'imc', 'cintura')
            var2: Segunda variable a variar ('glucosa', 'imc', 'cintura')
            var_range: Rango de variación (porcentaje)
            steps: Número de pasos en la variación

        Returns:
            dict: Matriz de sensibilidad y valores de los ejes
        """
        # Crear valores para cada variable
        valores_base = {
            'genero': genero,
            'edad': edad,
            'glucosa': glucosa_base,
            'insulina': insulina,
            'imc': imc_base,
            'cintura': cintura_base
        }

        # Crear rangos de variación
        factor_range = np.linspace(1 - var_range, 1 + var_range, steps)
        valores_var1 = [valores_base[var1] * factor for factor in factor_range]
        valores_var2 = [valores_base[var2] * factor for factor in factor_range]

        # Generar matriz
        matriz = []
        for v1 in valores_var1:
            fila = []
            for v2 in valores_var2:
                # Copiar valores base
                valores = valores_base.copy()
                # Actualizar con valores variados
                valores[var1] = v1
                valores[var2] = v2

                # Preparar datos para la predicción
                X = np.array([[
                    valores['genero'],
                    valores['edad'],
                    valores['glucosa'],
                    valores['insulina'],
                    valores['imc'],
                    valores['cintura']
                ]])

                # Escalar y predecir
                X_scaled = self.scaler.transform(X)
                probabilidad = self.model.predict_proba(X_scaled)[0][1]
                fila.append(float(probabilidad))
            matriz.append(fila)

        return {
            'matriz': matriz,
            'ejes': {
                var1: [float(v) for v in valores_var1],
                var2: [float(v) for v in valores_var2]
            }
        }