## **¿Qué hace este script?**

1. **Descarga y actualiza resultados históricos**
   - Obtiene automáticamente los últimos resultados de Powerball y Mega Millions desde fuentes oficiales en internet.
   - Puedes agregar históricos manuales para NY, FL, Leidsa y Loto Real en el archivo `historicos_loteria.csv`.

2. **Procesa y analiza los datos**
   - Lee el archivo CSV y organiza los resultados por lotería, sorteo y cantidad de dígitos.
   - Calcula estadísticas como:
     - Combinaciones más frecuentes
     - Frecuencia de cada número
     - Sumas más comunes
     - Parejas y tripletas más comunes

3. **Genera combinaciones con algoritmos clásicos**
   - **Mandel:** Genera todas las combinaciones posibles para cada juego.
   - **Haigh:** Sugiere combinaciones menos populares (evita las más jugadas).

4. **Predice con inteligencia artificial**
   - **Random Forest:** Analiza patrones en los datos históricos y sugiere posibles combinaciones futuras.
   - **Red Neuronal (Keras/TensorFlow):** Detecta patrones más complejos para predecir el número más probable y genera una combinación.

5. **Exporta combinaciones sugeridas**
   - Guarda en archivos CSV las combinaciones sugeridas por Mandel y Haigh para cada sorteo y lotería.

6. **Muestra resultados en consola**
   - Imprime estadísticas, sugerencias y resultados de IA para cada lotería y sorteo.

---

## **¿Cómo usarlo en Termux (o cualquier sistema con Python)?**

1. **Instala dependencias:**
   ```bash
   pkg install python
   pip install numpy pandas scikit-learn tensorflow requests
   ```

2. **Guarda el script como `super_loterias.py`.**

3. **Crea el archivo de históricos `historicos_loteria.csv`** (puedes dejarlo vacío para solo Powerball/Mega Millions, o agregar tus históricos de NY, FL, Leidsa, Loto Real).

4. **Ejecuta el script:**
   ```bash
   python super_loterias.py
   ```

5. **Obtendrás:**
   - Estadísticas y sugerencias en consola.
   - Archivos CSV con combinaciones para cada lotería/sorteo.

---

## **¿Qué más puedo hacer con este script?**

- Expandirlo para más loterías y más tipos de análisis.
- Integrar visualización en gráficos (matplotlib/seaborn).
- Exportar resultados a Excel.
- Enviar sugerencias por Telegram/WhatsApp.
- Descargar históricos de otras loterías si existe fuente web.
        
