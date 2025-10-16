# Reconocimiento de Acorde de Guitarra

Este proyecto utiliza una **red neuronal** entrenada con **TensorFlow/Keras** para **reconocer acordes de guitarra** a partir de **audio grabado en tiempo real** a través del micrófono del navegador. La interfaz web está construida con **Flask**.

## Características

- Reconocimiento de acordes en tiempo real desde el navegador.
- Interfaz web simple e intuitiva.
- Arquitectura CNN para clasificación de espectrogramas de audio.
- Preprocesamiento de audio consistente entre entrenamiento y predicción.

## Requisitos

- Python 3.8 o superior
- TensorFlow
- Librosa
- Flask
- SoundFile
- PyAudio (opcional, para entrenamiento local si se usa `src/predict.py` directamente)
- Navegador web moderno (Chrome, Firefox, Edge)

## Instalación

1. **Clona este repositorio:**

   ```bash
   git clone https://github.com/mattiviera/reconocimiento_acordes.git
   cd reconocimiento_acordes
   ```

2. **Crea un entorno virtual (recomendado):**

   ```bash
   python -m venv venv
   ```

3. **Activa el entorno virtual:**

   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Entrenamiento (opcional si ya tienes un modelo):**

   Si necesitas entrenar el modelo con tus propios datos:

   ```bash
   python main.py
   # Selecciona la opción 1: Entrenar modelo
   ```

2. **Ejecutar la aplicación web:**

   ```bash
   python app.py
   ```

3. **Acceder a la interfaz:**

   Abre tu navegador y ve a `http://127.0.0.1:5000/`.

4. **Utilizar la aplicación:**

   - Haz clic en el botón de micrófono para **iniciar la grabación**.
   - Toca un acorde en tu guitarra.
   - Haz clic nuevamente en el botón para **detener la grabación**.
   - La aplicación mostrará el **acorde reconocido** y la **confianza** de la predicción.

## Notas

- Asegúrate de otorgar permiso al navegador para acceder al micrófono.
- El modelo actual reconoce los acordes: Am, Bb, C, Dm, Em, F, G (puede variar según tu entrenamiento).
- La precisión depende de la calidad y variedad de los datos de entrenamiento.
- El archivo `predict.py` se utiliza internamente por `app.py` para realizar la predicción.
