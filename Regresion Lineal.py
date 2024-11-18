# Tratamiento de datos
# ==============================================================================
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import statsmodels.api as sm

# Configuración matplotlib
# ==============================================================================
plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('once')

# Datos para el dataframe
data = {
    "Horas Estudiadas": [2, 3, 1, 4, 5, 6, 7, 8, 2.5, 3.5, 5.5, 7.5, 9],
    "Calificación Obtenida": [6.5, 7.0, 5.5, 7.5, 8.0, 8.5, 9.0, 9.2, 6.8, 7.3, 8.3, 9.1, 9.5]
}

# Crear el dataframe
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(6, 3.84))

df.plot(
    x='Horas Estudiadas',
    y='Calificación Obtenida',
    c='firebrick',
    kind="scatter",
    ax=ax
)
#ax.set_title('grafica estudios/horas invertidas');

corr_test = pearsonr(x = df['Horas Estudiadas'], y =  df['Calificación Obtenida'])
print(f"Coeficiente de correlación de Pearson: {corr_test[0]}")
print(f"P-value: {corr_test[1]}")

X = df[['Horas Estudiadas']]
y = df['Calificación Obtenida']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
modelo = LinearRegression()
modelo.fit(X = X_train, y = y_train)

print(f"Intercept: {modelo.intercept_}")
print(f"Coeficiente: {list(zip(modelo.feature_names_in_, modelo.coef_))}")
print(f"Coeficiente de determinación R^2:", modelo.score(X, y))

predicciones = modelo.predict(X=X_test)
rmse = root_mean_squared_error(y_true=y_test, y_pred=predicciones)
print(f"Primeras cinco predicciones: {predicciones[0:5]}")
print(f"El error (rmse) de test es: {rmse}")

plt.show()
#Stastmodels
#######################################################################################################################

X = df[['Horas Estudiadas']]
y = df['Calificación Obtenida']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size=0.8,
                                        random_state=1234,
                                        shuffle=True
                                    )

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train)
modelo = modelo.fit()
print(modelo.summary())

intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci.columns = ['2.5%', '97.5%']
intervalos_ci

predicciones = modelo.get_prediction(exog=X_train).summary_frame(alpha=0.05)
predicciones.head(4)

predicciones = modelo.get_prediction(exog=X_train).summary_frame(alpha=0.05)
predicciones['x'] = X_train.loc[:, 'Horas Estudiadas']
predicciones['y'] = y_train
predicciones = predicciones.sort_values('x')

fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(predicciones['x'], predicciones['y'], marker='o', color="gray")
ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.3)
ax.legend();


plt.show()
