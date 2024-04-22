import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import skew, kurtosis, norm
from scipy.linalg import sqrtm
import yfinance as yf
import cvxpy as cp
import riskfolio as rp
import scipy.stats as scs
from scipy.optimize import minimize
import plotly.graph_objects as go

# print(plt.style.available) #list of available styles
plt.style.use('ggplot')


FECHA_INICIO = '2014-02-27'
FECHA_FIN = '2024-02-27'
market_benchmark = '^RUT'
stocks_tickers = [
    'SMCI',  # SUPER MICRO COMPUTER, INC.
    'SSD',   # SIMPSON MANUFACTURING CO., INC.
    'CYTK',  # CYTOKINETICS, INCORPORATED
    'CAC',   # E.L.F. BEAUTY, INC.
    'MSTR',  # MICROSTRATEGY INCORPORATED
    'UFPI',  # UFP INDUSTRIES, INC.
    'LNW',   # LIGHT & WONDER, INC.
    'T',  # ONTO INNOVATION INC.
    'RMBS',  # RAMBUS INC.
    'NVDA',  # BELLRING BRANDS, INC.
    'QLYS',  # QUALYS, INC.
    'FIX',   # COMFORT SYSTEMS USA, INC.
    'JNJ',   # API GROUP CORPORATION
    'SPSC',  # SPS COMMERCE, INC.
    'FN',    # FABRINET
    'JNPR',  # CHORD ENERGY CORPORATION
    'NFLX',  # WEATHERFORD INTERNATIONAL PLC
    'CVX',  # INTRA-CELLULAR THERAPIES, INC.
    'AIT',   # APPLIED INDUSTRIAL TECHNOLOGIES, INC.
    'TSLA',  # DUOLINGO, INC.
    'MTDR',  # MATADOR RESOURCES COMPANY
    'RHP',   # RYMAN HOSPITALITY PROPERTIES, INC.
    'SSB',   # SOUTHSTATE CORPORATION
    'MTH',   # MERITAGE HOMES CORPORATION
    'MUR',   # MURPHY OIL CORPORATION
    'ENSG',  # THE ENSIGN GROUP, INC.
    'CSCO',  # ATKORE INC.
    'NOVT',  # NOVANTA INC.
    'SIGI',  # SELECTIVE INSURANCE GROUP, INC.
    'CVLG'   # VAXCYTE, INC.
]


prices_stocks = yf.download(stocks_tickers, start=FECHA_INICIO, end=FECHA_FIN)['Adj Close']

retornos_stocks = np.log(prices_stocks).diff().dropna()
prices_benchmark = yf.download(market_benchmark, start=FECHA_INICIO, end=FECHA_FIN)['Adj Close']
retornos_benchmark = np.log(prices_benchmark).diff().dropna()

plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.show()


#CAPM
FECHA_INICIO_CAPM = '2014-02-28'

factors = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', skiprows=3)
factors.head()

factors.columns = ['date', 'mkt-rf', 'smb', 'hml', 'rf']
factors.head()
factors['date'] = factors['date'].astype(str)
factors.head()
factors['date'] = pd.to_datetime(factors['date'],format='%Y%m%d', errors='coerce').dt.strftime("%Y-%m-%d")
factors.head()
factors = factors.set_index('date')
factors.head()
factors = factors.loc[FECHA_INICIO_CAPM:FECHA_FIN]
factors.head()
factors.index = pd.to_datetime(factors.index)
factors.head()
factors = factors.apply(pd.to_numeric, errors='coerce').div(100)
rf = factors['rf']
rf.head()
rf= rf.to_numpy().mean() 


def cartera_max_sharpe(retornos, rf):
    num_act = len(retornos_stocks.columns)
    retornos_esperados = retornos_stocks.mean().to_numpy() 
    matriz_cov = retornos_stocks.cov().to_numpy()

    x = cp.Variable(num_act)
    pesos = x / cp.sum(x)

    pi = retornos_esperados - rf

    riesgo = cp.quad_form(x, matriz_cov)

    restricciones = [pi @ x == 1,
                    x >= 0]

    objetivo = cp.Minimize(riesgo)

    problema = cp.Problem(objetivo, restricciones)

    resultado = problema.solve('ECOS')

    pesos_sharpe = pesos.value
    pesos_sharpe[pesos_sharpe <= 1e-4] = 0
    
    return pesos_sharpe

pesos_sharpe = cartera_max_sharpe(retornos_stocks, rf)

num_act = len(retornos_stocks)

pesos_ajustados = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos_sharpe])

activos_sharpe_filtrados = [asset for i, asset in enumerate(stocks_tickers) if pesos_ajustados[i] > 0]
pesos_sharpe_filtrados = [x for x in pesos_ajustados if x > 0]

# Crear el gráfico de donut
plt.figure(figsize=(8, 8))

plt.pie(pesos_sharpe_filtrados, labels=activos_sharpe_filtrados, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3, 'edgecolor': 'black'})

# Añadir título
plt.title('Composición de la Cartera')

# Mostrar el gráfico
plt.show()


df_sharpe = pd.DataFrame()
df_sharpe["Tickers"] = activos_sharpe_filtrados
df_sharpe["Pesos"] = pesos_sharpe_filtrados
df_sharpe

retornos_cartera_sharpe = (retornos_stocks @ pesos_sharpe)
rent_sharpe = retornos_cartera_sharpe.mean() 
vol_sharpe = retornos_cartera_sharpe.std() 
var_sharpe = retornos_cartera_sharpe.var() 



#dibujamos la frontera eficinte

# Calcular la cartera de mínimo riesgo.
num_activos = len(retornos_stocks.columns)
retornos_esperados = retornos_stocks.mean().to_numpy()
matriz_cov = retornos_stocks.cov().to_numpy()

pesos = cp.Variable(num_activos)

restricciones = [cp.sum(pesos) == 1,
                 pesos >= 0]

rent = retornos_esperados @ pesos
riesgo = cp.quad_form(pesos, matriz_cov)

objetivo = cp.Minimize(riesgo)

problema = cp.Problem(objetivo, restricciones)

resultado = problema.solve("ECOS")

pesos_min_riesgo = pesos.value
pesos_min_riesgo[pesos_min_riesgo <= 1e-4] = 0



retornos_cartera = retornos_stocks@ pesos_min_riesgo
rent_min_riesgo = retornos_cartera.mean()
risk_min_riesgo = retornos_cartera.std()

plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(risk_min_riesgo*np.sqrt(252), rent_min_riesgo*252, 'x', label="Min riesgo")
plt.legend()
plt.show()

# 3. Dibuja la frontera eficiente.
imax = np.argmax(retornos_stocks.mean())
max_riesgo = np.sqrt(matriz_cov[imax, imax])
riesgos = np.linspace(risk_min_riesgo, max_riesgo, 200)
rentabilidades = []
for risk in riesgos:
    pesos = cp.Variable(num_activos)

    rent = retornos_esperados @ pesos
    riesgo = cp.quad_form(pesos, matriz_cov)
    
    restricciones = [cp.sum(pesos) == 1,
                     pesos >= 0,
                     riesgo <= risk**2]

    objetivo = cp.Maximize(rent)

    problema = cp.Problem(objetivo, restricciones)

    resultado = problema.solve("ECOS")
    
    pesos = pesos.value
    retornos_cartera = retornos_stocks @ pesos
    rent = retornos_cartera.mean()
    rentabilidades.append(rent)
rentabilidades = np.array(rentabilidades)


plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(riesgos*np.sqrt(252), rentabilidades*252, '-')
plt.plot(risk_min_riesgo*np.sqrt(252), rent_min_riesgo*252, 'x', label="Min riesgo")
plt.plot(vol_sharpe * np.sqrt(252), rent_sharpe * 252, 'x', label="Cartera Sharpe")
plt.legend()
plt.show()





# vamos a poner que la volatilidad objetivo sea 0.35 anual

#calculamos la cartera lineal con el activo libre de riesgo
# para fijar el riesgo objetivo.

rent_objetivo = 0.35 / 252
w = (rent_objetivo - rent_sharpe) / (rf - rent_sharpe)
risk_objetivo = (1 - w) * vol_sharpe

plt.style.use('ggplot')  # Estilo de gráfico
plt.figure(figsize=(10, 8))  # Tamaño del gráfico
plt.plot(0, rf*252, 'x', label="Activo libre de riesgo", markersize=10, color='red')  # Marcador para el activo libre de riesgo
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, 'o', color='blue', alpha=0.7, label='Acciones')  # Puntos para las acciones
plt.plot(riesgos*np.sqrt(252), rentabilidades*252, '-', color='green', label='Frontera eficiente', linewidth=2)  # Línea para la frontera eficiente
plt.plot(risk_min_riesgo*np.sqrt(252), rent_min_riesgo*252, 'x', markersize=10, color='purple', label="Min riesgo")  # Marcador para el mínimo riesgo
plt.plot(vol_sharpe * np.sqrt(252), rent_sharpe * 252, 'x', markersize=10, color='orange', label="Cartera Sharpe")  # Marcador para la cartera de máxima relación Sharpe
plt.plot([0, risk_objetivo*np.sqrt(252)], [rf*252, rent_objetivo*252], '--', color='grey', alpha=0.5, linewidth=1.5, label='Línea de mercado de capitales')  # Línea del mercado de capitales
plt.title('Visualización de Cartera de Inversiones')  # Título del gráfico
plt.xlabel('Riesgo (Desviación estándar anualizada)')  # Etiqueta del eje X
plt.ylabel('Retorno Esperado Anualizado')  # Etiqueta del eje Y
plt.legend()  # Leyenda
plt.show()  # Mostrar el gráfico

# EL MISMO GRAFICO PERO INTERACTIVO

fig = go.Figure()

# Agregar el activo libre de riesgo
fig.add_trace(go.Scatter(x=[0], y=[rf*252], mode='markers', marker=dict(color='red', size=10), name='Activo libre de riesgo'))

# Agregar las acciones
fig.add_trace(go.Scatter(x=retornos_stocks.std()*np.sqrt(252), y=retornos_stocks.mean()*252, mode='markers', marker=dict(color='blue', opacity=0.7), name='Acciones'))

# Agregar la frontera eficiente
fig.add_trace(go.Scatter(x=riesgos*np.sqrt(252), y=rentabilidades*252, mode='lines', line=dict(color='green', width=2), name='Frontera eficiente'))

# Agregar el punto de mínimo riesgo
fig.add_trace(go.Scatter(x=[risk_min_riesgo*np.sqrt(252)], y=[rent_min_riesgo*252], mode='markers', marker=dict(color='purple', size=10), name='Min riesgo'))

# Agregar la cartera Sharpe
fig.add_trace(go.Scatter(x=[vol_sharpe*np.sqrt(252)], y=[rent_sharpe*252], mode='markers', marker=dict(color='orange', size=10), name='Cartera Sharpe'))

# Agregar la línea del mercado de capitales
fig.add_trace(go.Scatter(x=[0, risk_objetivo*np.sqrt(252)], y=[rf*252, rent_objetivo*252], mode='lines', line=dict(color='grey', width=2, dash='dash'), name='Línea de mercado de capitales'))

# Actualizar el diseño del gráfico
fig.update_layout(title='Visualización de Cartera de Inversiones',
                  xaxis_title='Riesgo (Desviación estándar anualizada)',
                  yaxis_title='Retorno Esperado Anualizado',
                  plot_bgcolor='white')

# Mostrar el gráfico
fig.show()


# CREAR CARTERA EQUIPONDERADA

pesos_equi = np.ones(num_activos) /num_activos

retornos_cartera_equi = retornos_stocks @ pesos_equi
rent_equi = retornos_cartera_equi.mean()
vol_equi = retornos_cartera_equi.std()
var_equi = retornos_cartera_equi.var()

# graficar la cartera equiponderada

plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_equi*np.sqrt(252), rent_equi*252, 'x', label="Equiponderada")
plt.legend()
plt.show()


# CARTERA DE MÁXIMO DRAWDOW

def calcular_drawdown(portfolio_returns):
    wealth_index = (1 + portfolio_returns).cumprod() #FALTA EL 1000 ANTES DE (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

# Simular 10000 combinaciones aleatorias de pesos de cartera
n_simulaciones = 10000
n_activos = retornos_stocks.shape[1]
resultados = []

for _ in range(n_simulaciones):
    weights = np.random.random(n_activos)
    weights /= np.sum(weights)  # Asegurar que la suma de los pesos sea 1
    portfolio_returns = retornos_stocks.dot(weights)
    drawdown = calcular_drawdown(portfolio_returns)
    resultados.append((drawdown, weights))


min_drawdown, optimal_weights = min(resultados, key=lambda x: x[0])

print("Mínimo Drawdown:", min_drawdown)
print("Pesos óptimos:", optimal_weights)    


# calcular la cartera de máximo drawdown
retornos_cartera_drawdown = retornos_stocks @ optimal_weights
rent_drawdown = retornos_cartera_drawdown.mean()
vol_drawdown = retornos_cartera_drawdown.std()
var_drawdown = retornos_cartera_drawdown.var()

# graficar la cartera de máximo drawdown
plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_drawdown*np.sqrt(252), rent_drawdown*252, 'x', label="Max Drawdown")
plt.legend()
plt.show()




















# Cartera que minimiza el CVAR al 95%

def calcular_cvar(portfolio_returns, alpha=0.05):
    return -portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, alpha)].mean()

# Simular 10000 combinaciones aleatorias de pesos de cartera
n_simulaciones = 10000
n_activos = retornos_stocks.shape[1]
resultados = []

for _ in range(n_simulaciones):
    weights = np.random.random(n_activos)
    weights /= np.sum(weights)  # Asegurar que la suma de los pesos sea 1
    portfolio_returns = retornos_stocks.dot(weights)
    cvar = calcular_cvar(portfolio_returns)
    resultados.append((cvar, weights))


min_cvar, optimal_weights = min(resultados, key=lambda x: x[0])

print("Mínimo CVAR:", min_cvar)
print("Pesos óptimos:", optimal_weights)

# graficar la cartera de mínimo CVAR
retornos_cartera_cvar = retornos_stocks @ optimal_weights
rent_cvar = retornos_cartera_cvar.mean()
vol_cvar = retornos_cartera_cvar.std()
var_cvar = retornos_cartera_cvar.var()

plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_cvar*np.sqrt(252), rent_cvar*252, 'x', label="Min CVAR")
plt.legend()
plt.show()


# VOY A TRATAR DE CALCULAR LA CARTERA DE MINIMO CVAR AL 95% COMO EN EL ARCHIBO DE: RIESGOS-SOLUCION

retornos_stocks_np = retornos_stocks.to_numpy()
retornos_esperados_stocks_np = retornos_stocks.mean().to_numpy()

# Calculamos los datos necesarios para la optimización

alpha = 0.05    # Nivel de confianza

num_data, num_act = retornos_stocks.shape # Número de datos y número de activos

# Variables de la optimización

pesos = cp.Variable(num_act)
t = cp.Variable()  #VaR
ui = cp.Variable(num_data) #Exceso de pérdida con respecto al VaR

riesgo = t + cp.sum(ui)/(alpha * num_data)  #CVaR
retornos = retornos_esperados.T @ pesos

restricciones = [
    -retornos_stocks_np @ pesos - t - ui <= 0,  # Las pérdidas son menores que VaR más el exceso de pérdida
    ui >= 0,  # Los excesos son positivos
    cp.sum(pesos) == 1,
    pesos >= 0,  # No se pueden tener posiciones cortas
]

objective = cp.Minimize(riesgo)

# Solve the problem
prob = cp.Problem(objective, restricciones)
cvar95_min_cvar = prob.solve(solver='ECOS')

#ponemos a cero los pesos menores a 10**-4

pesos_ajustados_cvar = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos.value])

activos_filtrados_cvar = [asset for i, asset in enumerate(stocks_tickers) if pesos_ajustados_cvar[i] > 0] # Activos con pesos mayores a 10**-4
pesos_filtrados_cvar = [x for x in pesos_ajustados_cvar if x > 0] # Pesos mayores a 10**-4

activos_filtrados_cvar = [asset for i, asset in enumerate(stocks_tickers) if pesos_ajustados_cvar[i] > 0] # Activos con pesos mayores a 10**-4
pesos_filtrados_cvar = [x for x in pesos_ajustados_cvar if x > 0] # Pesos mayores a 10**-4

#graficar el binomio rentabilidad- riesgo la cartera de mínimo CVAR 
plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_cvar*np.sqrt(252), rent_cvar*252, 'x', label="Min CVAR")
plt.legend()
plt.show()

# Crear el gráfico de donut
plt.figure(figsize=(8, 8))

plt.pie(pesos_filtrados_cvar, labels=activos_filtrados_cvar, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3, 'edgecolor': 'black'})

# Añadir título
plt.title('Composición de la Cartera')

# Mostrar el gráfico
plt.show()

# Calcular la cartera Risk Parity

def cartera_risk_parity (ret):
    
    ''' 
    Función que calcula la cartera de riesgo paridad para un DataFrame de rendimientos
    ret: DataFrame de rendimientos
    Retorna pesos_ajustados: Array con los pesos de la cartera de riesgo paridad'''
    
    if isinstance(ret, pd.DataFrame):
        
        num_act = ret.shape[1]
        matriz_cov = ret.cov().to_numpy()
        retornos_esperados = ret.mean().to_numpy()
        
        b = 1/num_act

        x = cp.Variable(num_act)
        gamma = cp.Variable(num_act, nonneg=True)
        psi = cp.Variable(nonneg=True)

        z = matriz_cov @ x

        obj = cp.pnorm(b**0.5 * psi - gamma, p=2)
        ret = retornos_esperados.T @ x

        constraints = [cp.sum(x) == 1,
                    x >= 0,
                    cp.SOC(psi, sqrtm(matriz_cov) @ x)]

        for i in range(num_act):
            constraints += [cp.SOC(x[i] + z[i],
                                cp.vstack([2*gamma[i], x[i] - z[i]]))
                            ]

        objective = cp.Minimize(obj * 1000)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='ECOS')

        pesos_ajustados = np.array([np.round(xi, 3) if xi > 10**-4 else 0 for xi in x.value])

        return pesos_ajustados 
    
    
    else:
        raise ValueError('La función cartera_risk_parity solo acepta un DataFrame como argumento')
    
pesos_risk_parity = cartera_risk_parity(retornos_stocks)


# plot risk parity

pesos_ajustados_risk_parity = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos_risk_parity])

activos_filtrados_risk_parity = [asset for i, asset in enumerate(stocks_tickers) if pesos_ajustados_risk_parity[i] > 0] # Activos con pesos mayores a 10**-4
pesos_filtrados_risk_parity = [x for x in pesos_ajustados_risk_parity if x > 0] # Pesos mayores a 10**-4

# Crear el gráfico de donut
plt.figure(figsize=(8, 8))

plt.pie(pesos_filtrados_risk_parity, labels=activos_filtrados_risk_parity, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3, 'edgecolor': 'black'})

# Añadir título

plt.title('Composición de la Cartera')

# Mostrar el gráfico

plt.show()


retornos_cartera_risk_parity = (retornos_stocks @ pesos_risk_parity)
rent_risk_parity = retornos_cartera_risk_parity.mean()
vol_risk_parity = retornos_cartera_risk_parity.std()
var_risk_parity = retornos_cartera_risk_parity.var()

# graficar la cartera de riesgo paridad
plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_risk_parity*np.sqrt(252), rent_risk_parity*252, 'x', label="Risk Parity")
plt.legend()
plt.show()


# voy a comprobar que de otra manera que esté bien hecha la cartera de sharpe

def cartera_max_sharpe(ret, ret_rf):
    
    ''' Función que calcula la cartera de máximo índice de Sharpe para un DataFrame de rendimientos
    ret: DataFrame de rendimientos
    ret_rf: Rendimiento del activo libre de riesgo
    Retorna pesos_ajustados: Array con los pesos de la cartera de máximo índice de Sharpe'''
    
    if isinstance(ret, pd.DataFrame):
    
        num_act = ret.shape[1]
        matriz_cov = ret.cov().to_numpy()
        retornos_esperados = ret.mean()

        # Variable de decisión (pesos del portafolio)
        x = cp.Variable(num_act)
        # Riesgo (desviación estándar) del portafolio
        riesgo = cp.quad_form(x, matriz_cov)

        #Cálculo de pi como retornos esperados menos la rantabilidad del activo libre de riesgo
        pi = np.array(retornos_esperados - ret_rf)

        #Restricciones
        constraints = [pi @ x ==1, # para que el numerador sea 1
                    x>=0]       # sin posiciones cortas

        objective = cp.Minimize(riesgo) # Minimizo el riesgo

        # Problema de optimización
        problema = cp.Problem(objective, constraints)        

        # Resolver el problema
        resultado  = problema.solve(solver=cp.ECOS)

        # Normalizo los pesos
        pesos = x.value
        pesos /= pesos.sum()

        pesos_ajustados = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos])

        return pesos_ajustados
    
    else:
        raise ValueError('La función cartera_max_sharpe solo acepta un DataFrame como argumento')
    
pesos_sharpe = cartera_max_sharpe(retornos_stocks, rf)

# plot sharpe

pesos_ajustados_sharpe = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos_sharpe])

activos_filtrados_sharpe = [asset for i, asset in enumerate(stocks_tickers) if pesos_ajustados_sharpe[i] > 0] # Activos con pesos mayores a 10**-4

pesos_filtrados_sharpe = [x for x in pesos_ajustados_sharpe if x > 0] # Pesos mayores a 10**-4

# Crear el gráfico de donut

plt.figure(figsize=(8, 8))

plt.pie(pesos_filtrados_sharpe, labels=activos_filtrados_sharpe, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3, 'edgecolor': 'black'})

# Añadir título

plt.title('Composición de la Cartera')

# Mostrar el gráfico

plt.show()

def calc_regresion(r_ind, r):
    res = []
    summary = []
    
    for activo in r.columns:
        X = r_ind
        y = r[activo]
        X_sm = sm.add_constant(X)
        
        modelo = sm.OLS(y, X_sm).fit()
        
        resultado = {
        'activo': activo,
        'alpha': modelo.params[0],
        'beta': modelo.params[1],
        'p_value_alpha': modelo.pvalues[0],
        'p_value_beta': modelo.pvalues[1],
        't_value_alpha': modelo.tvalues[0],
        't_value_beta': modelo.tvalues[1],
        'rsquared': modelo.rsquared,
        'fvalue': modelo.fvalue,
        'conf_int_alpha_low': modelo.conf_int()[0][0],
        'conf_int_alpha_high': modelo.conf_int()[0][1],
        'conf_int_beta_low': modelo.conf_int()[1][0],
        'conf_int_beta_high': modelo.conf_int()[1][1],
        'aic': modelo.aic,
        'bic': modelo.bic
        }
        
        res.append(resultado)
        summary.append(modelo.summary())
        
    df_resul = pd.DataFrame(res)
    df_resul = df_resul.set_index('activo')
    
    df_summ = pd.DataFrame(summary)
    df_summ.index = r.columns
    
    return df_resul, df_summ




# calcular la cartera que genera un mayor alpha

def cartera_max_alpha(ret, ret_rf):
        
        ''' Función que calcula la cartera de máximo alpha para un DataFrame de rendimientos
        ret: DataFrame de rendimientos
        ret_rf: Rendimiento del activo libre de riesgo
        Retorna pesos_ajustados: Array con los pesos de la cartera de máximo alpha'''
        
        if isinstance(ret, pd.DataFrame):
        
            num_act = ret.shape[1]
            matriz_cov = ret.cov().to_numpy()
            retornos_esperados = ret.mean()
    
            # Variable de decisión (pesos del portafolio)
            x = cp.Variable(num_act)
            # Riesgo (desviación estándar) del portafolio
            riesgo = cp.quad_form(x, matriz_cov)
    
            #Cálculo de pi como retornos esperados menos la rantabilidad del activo libre de riesgo
            pi = np.array(retornos_esperados - ret_rf)
    
            #Restricciones
            constraints = [pi @ x ==1, # para que el numerador sea 1
                        x>=0]       # sin posiciones cortas
    
            objective = cp.Maximize(pi @ x) # Maximo alpha
    
            # Problema de optimización
            problema = cp.Problem(objective, constraints)        
    
            # Resolver el problema
            resultado  = problema.solve(solver=cp.ECOS)
    
            # Normalizo los pesos
            pesos = x.value
            pesos /= pesos.sum()
    
            pesos_ajustados = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos])
    
            return pesos_ajustados
        
        else:
            raise ValueError('La función cartera_max_alpha solo acepta un DataFrame como argumento')
        

pesos_alpha = cartera_max_alpha(retornos_stocks, rf)

# plot alpha

pesos_ajustados_alpha = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in pesos_alpha])

activos_filtrados_alpha = [asset for i, asset in enumerate(stocks_tickers) if pesos_ajustados_alpha[i] > 0] # Activos con pesos mayores a 10**-4

pesos_filtrados_alpha = [x for x in pesos_ajustados_alpha if x > 0] # Pesos mayores a 10**-4

# Crear el gráfico de donut

plt.figure(figsize=(8, 8))

plt.pie(pesos_filtrados_alpha, labels=activos_filtrados_alpha, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3, 'edgecolor': 'black'})

# Añadir título

plt.title('Composición de la Cartera')

# Mostrar el gráfico

plt.show()

#graficar el binomio rentabilidad- riesgo la cartera de máximo alpha

retornos_cartera_alpha = (retornos_stocks @ pesos_alpha)
rent_alpha = retornos_cartera_alpha.mean()
vol_alpha = retornos_cartera_alpha.std()
var_alpha = retornos_cartera_alpha.var()

plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_alpha*np.sqrt(252), rent_alpha*252, 'x', label="Max Alpha")
plt.legend()
plt.show()


def calc_alpha_cartera(weights, r_ind, r):
    alphas = []
    for activo in r.columns:
        X = sm.add_constant(r_ind)  # Añadir constante
        y = r[activo]
        model = sm.OLS(y, X).fit()
        alphas.append(model.params[0])  # Solo el alfa
    
    # Calcula el alfa de la cartera como el promedio ponderado de alfas individuales
    cartera_alpha = np.dot(weights, alphas)
    return cartera_alpha

from scipy.optimize import minimize

def optimizar_cartera(r_ind, r):
    n = len(r.columns)  # Número de activos

    # Función objetivo que queremos minimizar (negativo de alfa para maximizar)
    def objetivo(weights):
        return -calc_alpha_cartera(weights, r_ind, r)

    # Restricciones y límites
    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Los pesos deben sumar 1
    bounds = [(0, 1) for _ in range(n)]  # Los pesos deben estar entre 0 y 1

    # Punto inicial (pesos iniciales igualmente distribuidos)
    init_guess = [1/n] * n

    # Optimización
    opt_results = minimize(objetivo, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    return opt_results.x  # Los pesos óptimos

# Luego puedes llamar a optimizar_cartera pasando r_ind y retornos_stocks
optimal_weights = optimizar_cartera(retornos_benchmark, retornos_stocks)
optimal_weights_aju = np.array([np.round(x, 3) if x > 10**-4 else 0  for x in optimal_weights])

activos_filtrados_optimal = [asset for i, asset in enumerate(stocks_tickers) if optimal_weights_aju[i] > 0] # Activos con pesos mayores a 10**-4
pesos_filtrados_optimal = [x for x in optimal_weights_aju if x > 0] # Pesos mayores a 10**-4

# Crear el gráfico de donut
plt.figure(figsize=(8, 8))

plt.pie(pesos_filtrados_optimal, labels=activos_filtrados_optimal, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.3, 'edgecolor': 'black'})

# Añadir título

plt.title('Composición de la Cartera')

# Mostrar el gráfico

plt.show()

retornos_cartera_optimal = (retornos_stocks @ optimal_weights_aju)
rent_optimal = retornos_cartera_optimal.mean()
vol_optimal = retornos_cartera_optimal.std()
var_optimal = retornos_cartera_optimal.var()

# graficar la cartera de máximo alpha
plt.figure(figsize=(8, 6))
plt.plot(retornos_stocks.std()*np.sqrt(252), retornos_stocks.mean()*252, '.')
plt.plot(vol_optimal*np.sqrt(252), rent_optimal*252, 'x', label="Max Alpha")
plt.legend()
plt.show()


datos_df = pd.concat([factors, retornos_cartera_sharpe], axis=1, join='inner')
datos_df.columns = list(factors.columns) + ['retornos_cartera_optimal']
datos_df.head(5)
datos_df.tail()

Y = datos_df.loc[:, 'retornos_cartera_optimal']-datos_df['rf']
X = datos_df[['mkt-rf', 'smb', 'hml']]

X = sm.add_constant(X)

modelo = sm.OLS(Y, X).fit()

print(modelo.summary())