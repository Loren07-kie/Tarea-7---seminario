# NeonLens ✳ Laboratorio neuronal de imágenes

**Tarea 7 · PIAD-427 · HT-03: Define la estructura y crea una red neuronal artificial.**

Sube una foto, recibe un diagnóstico orientativo de calidad, experimenta con seis filtros, compara antes/después y descarga el PNG. Interfaz adaptable con logo propio, fondo SVG, degradados, letras estilo grafiti, lente 3D flotante en CSS y carrusel de tarjetas con perspectiva.

## Ejecutar

Recomendado Python 3.12. Desde la carpeta del proyecto:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py
python -m unittest discover -s tests -v
python app.py
```

Abre http://localhost:5000. En producción: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`.

## Red neuronal y evidencia

- `vision.py`: genera 500 escenas procedurales y cinco versiones por escena. Extrae brillo, contraste, nitidez, proporción de sombras y altas luces de imágenes de 64×64.
- `train.py`: StandardScaler + MLPClassifier, arquitectura **5 → 24 → 12 → 5**, ReLU, Adam y parada temprana. Semilla 42. Holdout de 20% separado por escena para evitar contaminación entre versiones. La validación interna de parada temprana es aleatoria dentro de entrenamiento; no constituye una evaluación independiente por escena.
- `models/network.json`: pesos y normalización entrenados, exportados a JSON. Inferencia NumPy estable con softmax; no se deserializan archivos pickle de usuarios.
- `reports/metrics.json`: exactitud, F1 macro, baseline mayoritario, matriz de confusión, reporte por clase y curva de pérdida.
- `notebooks/NeonLens_analisis.ipynb`: cuaderno ejecutado con exploración, gráficos, entrenamiento, evaluación, comparación e inferencia.
- `tests/test_app.py`: API, filtros, archivos inválidos, límites, estabilidad numérica y escenas sintéticas nuevas.

**Límite importante:** el problema sintético es deliberadamente sencillo y separable. Una exactitud alta aquí no demuestra generalización a fotografías reales. El conjunto no tiene anotaciones humanas. Las salidas son orientativas, no una medida universal de calidad. Los filtros usan OpenCV y no reconstruyen detalles perdidos. El ejemplo visual se dibuja con Canvas, no es una fotografía externa.

## Relación con SEMANA 12

El ZIP proporcionado contiene ejercicios NumPy de píxeles/umbrales y un laboratorio Flask/OpenCV incompleto con grises, desenfoque y bordes. NeonLens extiende esos conceptos con validación de carga, filtros terminados y una RNA entrenada para la tarea 7. El ZIP se consultó como material, no como instrucciones de ejecución; no se publicaron logos ni archivos originales del curso.

## Despliegue Render

Servicio web Python, plan gratuito, repositorio `Loren07-kie/Tarea-7---seminario`, rama `main`. Build: `pip install -r requirements.txt`. Start: comando gunicorn indicado arriba. Variable `PYTHON_VERSION=3.12.8`. Los pesos ya están incluidos: no se entrena en cada arranque. No requiere base de datos, claves ni servicios de IA externos. El plan gratuito puede suspenderse por inactividad.

## Privacidad y autenticación

Consulta [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md) para componentes, flujo, credenciales y tokens. La aplicación es pública y no tiene cuentas; no se inventa un sistema de autenticación donde no existe. Imágenes procesadas temporalmente en memoria, sin guardado en disco ni base de datos. No enviar datos sensibles a una demostración pública.

Las fuentes se solicitan a Google Fonts; hay fuentes de respaldo si no carga. Todos los recursos gráficos del proyecto son originales y procedurales. No necesita claves API.
