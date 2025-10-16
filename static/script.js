// static/script.js

// --- Nueva función para convertir Float32Array de audio en un Blob WAV ---
// Basada en la lógica de RecorderJS (https://github.com/mattdiamond/Recorderjs)
function encodeWAV(samples, sampleRate) {
  // Calcula el tamaño del buffer necesario
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  // Función auxiliar para escribir cadenas en el buffer
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  // Escribe el encabezado WAV (RIFF)
  writeString(0, "RIFF"); // ChunkID
  // ChunkSize: tamaño total del archivo - 8 bytes
  view.setUint32(4, 36 + samples.length * 2, true); // (36 es el tamaño del encabezado fmt + data, + tamaño de los datos)
  writeString(8, "WAVE"); // Format
  writeString(12, "fmt "); // Subchunk1ID
  view.setUint32(16, 16, true); // Subchunk1Size (tamaño del bloque fmt, siempre 16 para PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 es PCM)
  view.setUint16(22, 1, true); // NumChannels (1 es mono)
  view.setUint32(24, sampleRate, true); // SampleRate
  // ByteRate: SampleRate * NumChannels * BitsPerSample/8
  view.setUint32(28, (sampleRate * 1 * 16) / 8, true);
  // BlockAlign: NumChannels * BitsPerSample/8
  view.setUint16(32, (1 * 16) / 8, true);
  view.setUint16(34, 16, true); // BitsPerSample
  writeString(36, "data"); // Subchunk2ID
  // Subchunk2Size: NumSamples * NumChannels * BitsPerSample/8
  view.setUint32(40, (samples.length * 1 * 16) / 8, true);

  // Escribe los datos PCM (muestras)
  let offset = 44; // Comienza después del encabezado
  for (let i = 0; i < samples.length; i++, offset += 2) {
    // Asegura que el valor esté entre -1 y 1
    const s = Math.max(-1, Math.min(1, samples[i]));
    // Convierte a entero de 16 bits y escríbelo en el buffer
    // s < 0 ? s * 0x8000 : s * 0x7FFF convierte el rango [-1, 1] a [-32768, 32767]
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  // Devuelve un Blob con tipo 'audio/wav'
  return new Blob([view], { type: "audio/wav" });
}

// --- Variables ---
let isRecording = false;
let mediaRecorder;
let audioChunks = []; // No se usa directamente ahora, pero MediaRecorder se usa para controlar inicio/detención
let audioContext; // Para manejar el audio en bruto
let audioProcessor; // Nodo para procesar el audio en bruto
let allAudioData = []; // Almacenamos los datos de audio aquí

const recordButton = document.getElementById("recordButton");
const predictionResult = document.getElementById("predictionResult");
const status = document.getElementById("status");

// --- Función para iniciar la grabación ---
async function startRecording() {
  try {
    status.textContent = "Solicitando acceso al micrófono...";
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Creamos el AudioContext
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);

    // Creamos un ScriptProcessorNode para capturar el audio en bruto
    // El buffer size (4096) puede ajustarse si hay problemas de latencia
    audioProcessor = audioContext.createScriptProcessor(4096, 1, 1); // Buffer size, input, output channels (mono)

    // Reiniciar el array de datos de audio
    allAudioData = [];

    audioProcessor.onaudioprocess = (e) => {
      if (isRecording) {
        // Obtiene los datos del canal de entrada (mono)
        const inputData = e.inputBuffer.getChannelData(0);
        // Guarda una copia de los datos en el array
        // Usamos slice() para copiar los valores, no la referencia
        allAudioData.push(new Float32Array(inputData));
      }
    };

    // Conectamos el flujo de audio: source -> processor -> destination (para evitar errores)
    source.connect(audioProcessor);
    audioProcessor.connect(audioContext.destination);

    status.textContent = "Grabando...";

    // Usamos MediaRecorder para controlar el inicio/detención de la grabación
    // No usamos sus 'dataavailable' para el audio en bruto, sino para el control de flujo
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      // Aunque no usemos este evento para el audio en bruto, lo dejamos por si acaso
      // En esta versión, no concatenamos estos chunks, sino los datos del processor
      // audioChunks.push(event.data); // Comentado
    };

    mediaRecorder.onstop = async () => {
      status.textContent = "Procesando audio...";

      // Detener el procesamiento de audio
      if (audioProcessor) {
        audioProcessor.disconnect();
      }
      // Detener todos los tracks del stream
      stream.getTracks().forEach((track) => track.stop());

      // Concatenar todos los datos de audio capturados en un solo Float32Array
      const finalAudioArray = new Float32Array(
        allAudioData.reduce((acc, curr) => acc + curr.length, 0)
      );
      let offset = 0;
      for (const chunk of allAudioData) {
        finalAudioArray.set(chunk, offset);
        offset += chunk.length;
      }

      // Convertir el array de audio en bruto a un Blob WAV usando nuestra función
      const audioBlob = encodeWAV(finalAudioArray, audioContext.sampleRate);

      // Enviar el Blob WAV al servidor Flask
      await sendAudioToServer(audioBlob);
    };

    // Iniciar la grabación con MediaRecorder
    mediaRecorder.start();
    isRecording = true;
    recordButton.textContent = "Detener Grabación";
  } catch (err) {
    console.error("Error al acceder al micrófono: ", err);
    status.textContent = "Error al acceder al micrófono. Revisa los permisos.";
  }
}

// --- Función para detener la grabación ---
function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop(); // Esto activa el evento 'onstop'
    isRecording = false;
    recordButton.textContent = "Iniciar Grabación";
    status.textContent = "Subiendo audio..."; // Mensaje temporal mientras se procesa
  }
}

// --- Función para enviar el audio al servidor Flask ---
async function sendAudioToServer(audioBlob) {
  const formData = new FormData();
  // Adjuntamos el Blob WAV al formData con el nombre 'audio'
  // Este nombre debe coincidir con el usado en app.py: request.files['audio']
  formData.append("audio", audioBlob, "grabacion.wav");

  try {
    const response = await fetch("/predict", {
      method: "POST", // Método POST para enviar el archivo
      body: formData, // El cuerpo contiene el archivo
    });

    // Verificamos si la respuesta del servidor es exitosa (código 2xx)
    if (!response.ok) {
      // Si no es exitosa, lanzamos un error para que lo maneje el catch
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Si la respuesta es exitosa, intentamos leerla como JSON
    const data = await response.json();

    // Verificamos si el servidor envió un mensaje de error en el JSON
    if (data.error) {
      // Si hay un error, lo mostramos en el div de resultados con clase 'error'
      predictionResult.innerHTML = `<p class="error">Error: ${data.error}</p>`;
    } else {
      // Si no hay error, mostramos el acorde y la confianza con clase 'success'
      // La confianza se multiplica por 100 y se redondea a 2 decimales
      predictionResult.innerHTML = `<p class="success">Acorde: ${
        data.chord
      } (Confianza: ${(data.confidence * 100).toFixed(2)}%)</p>`;
    }
  } catch (error) {
    // Si ocurrió un error en la solicitud fetch o al procesar la respuesta
    console.error("Error al enviar el audio o recibir la predicción:", error);
    // Mostramos el error en el div de resultados con clase 'error'
    predictionResult.innerHTML = `<p class="error">Error: ${error.message}</p>`;
  } finally {
    // Finalmente, limpiamos el div de estado
    status.textContent = "";
  }
}

// --- Event listener para el botón de grabación ---
recordButton.addEventListener("click", () => {
  // Si está grabando, detenemos; si no, iniciamos
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});
