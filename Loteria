import csv
import random
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Configuración de loterías
LOTERIAS = [
    {"nombre": "NY", "sorteos": ["Midday", "Evening"], "digitos": [3, 4]},
    {"nombre": "FL", "sorteos": ["Midday", "Evening"], "digitos": [3, 4]},
    {"nombre": "Leidsa", "sorteos": ["Dia", "Noche"], "digitos": [3, 4]},
    {"nombre": "LotoReal", "sorteos": ["Dia", "Noche"], "digitos": [3, 4]},
    {"nombre": "Powerball", "sorteos": ["Main"], "digitos": [6]},           # 5+1 (5 blancos, 1 rojo)
    {"nombre": "MegaMillions", "sorteos": ["Main"], "digitos": [6]}         # 5+1 (5 blancos, 1 dorado)
]

RANGOS = {
    "NY": 10,
    "FL": 10,
    "Leidsa": 10,
    "LotoReal": 10,
    "Powerball": [69, 26],      # 5 de 1-69 y 1 de 1-26
    "MegaMillions": [70, 25]    # 5 de 1-70 y 1 de 1-25
}

# Descarga de resultados históricos para Powerball/MegaMillions
def descargar_powerball():
    url = "https://www.powerball.com/api/v1/numbers/powerball/recent100"
    r = requests.get(url)
    data = r.json()
    resultados = []
    for draw in data:
        nums = draw["field_winning_numbers"].split()
        powerball = draw["field_powerball"]
        resultados.append(("Powerball", "Main") + tuple(map(int, nums + [powerball])))
    return resultados

def descargar_megamillions():
    url = "https://data.ny.gov/resource/5xaw-6ayf.json?$limit=100"
    r = requests.get(url)
    data = r.json()
    resultados = []
    for draw in data:
        nums = draw["winning_numbers"].split()
        mega = draw["mega_ball"]
        resultados.append(("MegaMillions", "Main") + tuple(map(int, nums + [mega])))
    return resultados

def actualizar_historico(archivo):
    powerball = descargar_powerball()
    megamillions = descargar_megamillions()
    with open(archivo, "a", newline="") as f:
        writer = csv.writer(f)
        for fila in powerball + megamillions:
            writer.writerow(fila)

def cargar_historicos(archivo_csv, estado, sorteo, digitos=3):
    resultados = []
    try:
        with open(archivo_csv, newline='') as csvfile:
            lector = csv.reader(csvfile)
            for fila in lector:
                if len(fila) >= (2 + digitos) and fila[0] == estado and fila[1] == sorteo:
                    resultados.append(tuple(map(int, fila[2:2+digitos])))
    except:
        pass
    return resultados

def frecuencia_combos(resultados, top=10):
    return Counter(resultados).most_common(top)

def frecuencia_numeros(resultados, digitos=3):
    todos = [n for combo in resultados for n in combo]
    return Counter(todos).most_common(digitos)

def sumas_frecuentes(resultados, top=10):
    sumas = [sum(combo) for combo in resultados]
    return Counter(sumas).most_common(top)

def parejas_frecuentes(resultados, top=10):
    parejas = []
    for combo in resultados:
        parejas += [tuple(sorted([combo[i], combo[j]])) for i in range(len(combo)) for j in range(i+1, len(combo))]
    return Counter(parejas).most_common(top)

def tripletas_frecuentes(resultados, top=10):
    tripletas = []
    for combo in resultados:
        if len(combo) >= 3:
            tripletas += [tuple(sorted([combo[i], combo[j], combo[k]])) for i in range(len(combo)) for j in range(i+1, len(combo)) for k in range(j+1, len(combo))]
    return Counter(tripletas).most_common(top)

def mandel_combos(digitos=3, rango=10, especial=None):
    if especial is None:
        return list(itertools.product(range(rango), repeat=digitos))
    else:
        blancos = list(itertools.combinations(range(1, especial[0]+1), digitos-1))
        bolas = range(1, especial[1]+1)
        resultado = []
        for combo in blancos:
            for bola in bolas:
                resultado.append(tuple(list(combo) + [bola]))
        return resultado

def haigh_combo(resultados, digitos=3, rango=10, especial=None):
    populares = [combo for combo, _ in Counter(resultados).most_common(30)]
    intentos = 0
    while True:
        if especial is None:
            comb = tuple(random.sample(range(rango), digitos))
        else:
            blancos = random.sample(range(1, especial[0]+1), digitos-1)
            bola = random.choice(range(1, especial[1]+1))
            comb = tuple(blancos + [bola])
        if comb not in populares:
            return comb
        intentos += 1
        if intentos > 1000:
            return comb

def preparar_X_y(resultados, digitos=3):
    X, y = [], []
    for i in range(len(resultados)-1):
        X.append(list(resultados[i]))
        y.append(resultados[i+1][0])
    return np.array(X), np.array(y)

def ia_prediccion_rf(resultados, digitos=3, rango=10, especial=None):
    X, y = preparar_X_y(resultados, digitos)
    if len(X) < 30:
        return None
    modelo = RandomForestClassifier()
    modelo.fit(X, y)
    ultimos = np.array([list(resultados[-1])])
    prediccion = modelo.predict(ultimos)
    if especial is None:
        otros = random.sample(range(rango), digitos-1)
        combo = [int(prediccion[0])] + otros
    else:
        blancos = random.sample(range(1, especial[0]+1), digitos-1)
        bola = random.choice(range(1, especial[1]+1))
        combo = blancos + [bola]
    return tuple(combo)

def ia_prediccion_nn(resultados, digitos=3, rango=10, especial=None):
    X, y = preparar_X_y(resultados, digitos)
    if len(X) < 100:
        return None
    y_cat = to_categorical(y, num_classes=rango)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(digitos,)),
        Dense(16, activation='relu'),
        Dense(rango, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=30, verbose=0)
    ultimos = np.array([list(resultados[-1])])
    probas = model.predict(ultimos)[0]
    primer = np.argmax(probas)
    if especial is None:
        otros = random.sample(range(rango), digitos-1)
        combo = [primer] + otros
    else:
        blancos = random.sample(range(1, especial[0]+1), digitos-1)
        bola = random.choice(range(1, especial[1]+1))
        combo = blancos + [bola]
    return tuple(combo)

def exportar_csv(combos, nombre_archivo):
    df = pd.DataFrame(combos, columns=[f'N{i+1}' for i in range(len(combos[0]))])
    df.to_csv(nombre_archivo, index=False)
    print(f"Exportado a {nombre_archivo}")

def sugerencias_loteria(archivo, loteria, sorteo, digitos):
    print(f"\n--- {loteria} {sorteo} Pick-{digitos} ---")
    resultados = cargar_historicos(archivo, loteria, sorteo, digitos)
    if not resultados:
        print("No hay datos históricos suficientes.")
        return
    rango = RANGOS[loteria] if loteria in RANGOS and isinstance(RANGOS[loteria], int) else 10
    especial = None
    if loteria in ["Powerball", "MegaMillions"]:
        especial = RANGOS[loteria]
    print("Frecuencia de combos:", frecuencia_combos(resultados))
    print("Frecuencia de números:", frecuencia_numeros(resultados, digitos))
    print("Sumas más frecuentes:", sumas_frecuentes(resultados))
    print("Parejas frecuentes:", parejas_frecuentes(resultados))
    print("Tripletas frecuentes:", tripletas_frecuentes(resultados))
    print("Haigh (no populares):", haigh_combo(resultados, digitos, rango, especial))
    print("Mandel (primeras 5):", mandel_combos(digitos, rango, especial)[:5])
    rf_sug = ia_prediccion_rf(resultados, digitos, rango, especial)
    print("IA Random Forest:", rf_sug if rf_sug else "No suficiente data")
    nn_sug = ia_prediccion_nn(resultados, digitos, rango, especial)
    print("IA Red Neuronal:", nn_sug if nn_sug else "No suficiente data")
    exportar_csv([haigh_combo(resultados, digitos, rango, especial) for _ in range(20)], f"{loteria}_{sorteo}_haigh.csv")
    exportar_csv(mandel_combos(digitos, rango, especial)[:20], f"{loteria}_{sorteo}_mandel.csv")

if __name__ == "__main__":
    archivo = "historicos_loteria.csv"
    actualizar_historico(archivo)
    for lot in LOTERIAS:
        for sorteo in lot["sorteos"]:
            for digitos in lot["digitos"]:
                sugerencias_loteria(archivo, lot["nombre"], sorteo, digitos)
