import os
import sys
import tempfile
import traceback
import torch
import time
import threading
from tkinter import Tk, Toplevel, Label, Text, Button, filedialog, messagebox, StringVar, OptionMenu, Frame, Listbox, BooleanVar, Checkbutton
from tkinter import ttk
from pydub import AudioSegment

# -------------------------
# CONFIGURACIÓN DE VARIABLES DE ENTORNO Y DIRECTORIOS
# -------------------------
# Forzar un HOME escribible (esto debe ejecutarse antes de cualquier otro import que lo use)
user_home = os.path.expanduser('~')
if user_home == '/root':
    user_home = '/Users/franmartinez'  # Ajusta con tu ruta real
os.environ["HOME"] = user_home

# Directorio para almacenar modelos TTS
CUSTOM_TTS_PATH = os.path.join(user_home, 'tts_models')
os.makedirs(CUSTOM_TTS_PATH, exist_ok=True)
os.environ["COQUI_TTS_HOME"] = CUSTOM_TTS_PATH

# Directorio para caché
cache_dir = os.path.join(user_home, ".cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = cache_dir

# Directorios para Hugging Face (si aplica)
hf_home = os.path.join(user_home, ".cache", "huggingface")
os.makedirs(hf_home, exist_ok=True)
os.environ["HF_HOME"] = hf_home

transformers_cache = os.path.join(user_home, ".cache", "transformers")
os.makedirs(transformers_cache, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = transformers_cache

# Directorio para guardar audios de referencia
EXAMPLES_DIR = os.path.join(user_home, "examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

# -------------------------
# IMPORTAR DEPENDENCIAS
# -------------------------
try:
    from TTS.api import TTS
except ImportError as e:
    messagebox.showerror("Error", f"Required libraries not found: {e}")
    sys.exit(1)

try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
except ImportError:
    messagebox.showerror("Error", "Instala sounddevice y scipy (pip install sounddevice scipy) para grabar con el micrófono.")
    sys.exit(1)

# -------------------------
# CONFIGURACIÓN DE OPCIONES DE IDIOMA Y VOZ
# -------------------------
language_options = ["Español", "Inglés", "Francés", "Multilingüe"]
voice_options = {
    "Español": ["tts_models/es/css10/vits", "tts_models/es/mai/tacotron2-DDC"],
    "Inglés": ["tts_models/en/ljspeech/tacotron2-DDC", "tts_models/fr/bits"],
    "Francés": ["tts_models/fr/mai_tacotron", "tts_models/fr/bits"],
    # En "Multilingüe" incluimos dos opciones: una para Bark y otra para XTTS_v2 (que necesita referencia)
    "Multilingüe": ["tts_models/multilingual/multi-dataset/bark", "tts_models/multilingual/multi-dataset/xtts_v2"]
}

tts = None
reference_file = None

# -------------------------
# CREACIÓN DE LA VENTANA RAÍZ Y VARIABLES DE INTERFAZ
# -------------------------
app = Tk()
app.title("Generador de Voces")
app.geometry("600x600")

language_var = StringVar(app, value=language_options[0])
voice_var = StringVar(app, value=voice_options[language_options[0]][0])

# Variables específicas para Bark
bark_lang_options = ["en", "es", "fr", "de", "it", "ja", "ko"]
bark_lang_var = StringVar(app, value="es")
bark_prompt_options = ["Unconditional", "Announcer"] + [f"Speaker {i}" for i in range(10)]
bark_prompt_var = StringVar(app, value="Speaker 0")
is_song_var = BooleanVar(app, value=False)

# -------------------------
# DEFINICIÓN DE FUNCIONES
# -------------------------
def get_device():
    """Detecta el mejor dispositivo disponible (MPS para Mac, CUDA o CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def ensure_tts_model(model_name):
    """Carga el modelo TTS seleccionado usando el dispositivo adecuado."""
    try:
        device = get_device()
        print(f"Cargando modelo: {model_name} en dispositivo: {device}")
        custom_cache = os.path.join(user_home, "tts_cache")
        os.makedirs(custom_cache, exist_ok=True)
        tts_instance = TTS(
            model_name=model_name,
            gpu=(device.type == "cuda"),
            progress_bar=True,
        )
        tts_instance.model_name = model_name
        return tts_instance
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el modelo TTS: {e}")
        return None

def record_reference():
    """Graba un audio de referencia de 5 segundos y permite renombrarlo."""
    global reference_file
    duration = 5  # segundos
    fs = 44100    # frecuencia de muestreo
    messagebox.showinfo("Grabación", "La grabación iniciará ahora durante 5 segundos. Habla en el micrófono.")
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    for i in range(duration):
        time.sleep(1)
        print(f"Tiempo restante: {duration - i} segundos...")
    sd.wait()
    
    filename = filedialog.asksaveasfilename(
        initialdir=EXAMPLES_DIR,
        title="Guardar referencia como...",
        defaultextension=".wav",
        filetypes=[("Archivo WAV", "*.wav")]
    )
    if filename:
        wavfile.write(filename, fs, recording)
        messagebox.showinfo("Grabación", f"Referencia guardada en {filename}")
        reference_file = filename
        update_reference_list()

def update_reference_list():
    """Actualiza la lista de archivos de referencia disponibles."""
    reference_listbox.delete(0, "end")
    files = sorted([f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".wav")])
    for file in files:
        reference_listbox.insert("end", file)

def select_reference():
    """Permite seleccionar un archivo de referencia guardado previamente."""
    global reference_file
    selected = reference_listbox.curselection()
    if selected:
        reference_file = os.path.join(EXAMPLES_DIR, reference_listbox.get(selected[0]))
        messagebox.showinfo("Referencia", f"Archivo seleccionado: {reference_file}")

def insert_tag(tag):
    """Inserta un tag en el widget de texto en la posición actual."""
    text_widget = text_input
    text_widget.insert("insert", tag + " ")

def open_bark_tags():
    """Abre una ventana con botones para insertar tags predefinidos para Bark."""
    tags_window = Toplevel(app)
    tags_window.title("Insertar Tags para Bark")
    tags_window.geometry("400x200")
    
    Label(tags_window, text="Inserta etiquetas:", font=("Arial", 12, "bold")).pack(pady=5)
    
    # Lista de tags predefinidos
    tags = [
        "[laughter]",
        "[laughs]",
        "[sighs]",
        "[music]",
        "[gasps]",
        "[clears throat]",
        "—",
        "♪",
        "[MAN]",
        "[WOMAN]"
    ]
    
    # Crear un frame para los botones
    btn_frame = Frame(tags_window)
    btn_frame.pack(pady=5, padx=5)
    
    # Crear un botón para cada tag
    for tag in tags:
        btn = Button(btn_frame, text=tag, command=lambda t=tag: insert_tag(t), font=("Arial", 10))
        btn.pack(side="left", padx=2, pady=2)
    
    Button(tags_window, text="Cerrar", command=tags_window.destroy, font=("Arial", 12)).pack(pady=10)

def generate_audio():
    """Genera audio según el modelo seleccionado:
    
    - Si el modelo contiene 'bark': usa la API de Bark (flujo clásico, enviando el texto tal cual, lo que permite incluir etiquetas).
    - Si contiene 'xtts_v2': usa TTS con audio de referencia y parámetro 'language'.
    - En el resto, usa TTS de forma convencional.
    """
    global tts, reference_file
    selected_model = voice_var.get()
    text = text_input.get("1.0", "end").strip()
    if not text:
        messagebox.showwarning("Advertencia", "Por favor, introduce un texto.")
        return

    output_file = filedialog.asksaveasfilename(
        defaultextension=".mp3",
        filetypes=[("MP3 Audio", "*.mp3")]
    )
    if not output_file:
        return

    # Rama para modelos Bark
    if "bark" in selected_model.lower():
        try:
            from bark import preload_models, generate_audio as bark_generate_audio, SAMPLE_RATE
            preload_models()
            device = get_device()
            try:
                from bark.generation import semantic_model, coarse_model, fine_model
                semantic_model.to(device)
                coarse_model.to(device)
                fine_model.to(device)
                print(f"Modelos Bark movidos a: {device}")
            except Exception as e:
                print("Warning: No se pudieron mover algunos módulos de Bark, usando dispositivo por defecto.", e)
            
            # Enviar el texto tal cual (las etiquetas se incluirán según lo que el usuario inserte)
            audio_array = bark_generate_audio(text, history_prompt="v2/es_speaker_2")
            import scipy.io.wavfile as wavfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            wavfile.write(temp_wav_path, SAMPLE_RATE, audio_array)
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(output_file, format="mp3")
            os.remove(temp_wav_path)
            messagebox.showinfo("Éxito", f"Audio generado: {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generando audio con Bark: {e}")
    # Rama para modelos XTTS_v2 que requieren referencia
    elif "xtts_v2" in selected_model.lower():
        if tts is None or tts.model_name != selected_model:
            tts = ensure_tts_model(selected_model)
            if not tts:
                return
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            lang_map = {"Español": "es", "Inglés": "en", "Francés": "fr"}
            lang_code = lang_map.get(language_var.get(), "es")
            if not reference_file:
                messagebox.showwarning("Advertencia", "Selecciona o graba un audio de referencia.")
                return
            tts.tts_to_file(text=text, speaker_wav=reference_file, language=lang_code, file_path=temp_wav_path)
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(output_file, format="mp3")
            os.remove(temp_wav_path)
            messagebox.showinfo("Éxito", f"Audio generado: {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el audio: {e}")
    # Rama convencional de TTS
    else:
        if tts is None or tts.model_name != selected_model:
            tts = ensure_tts_model(selected_model)
            if not tts:
                return
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            tts.tts_to_file(text=text, file_path=temp_wav_path)
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(output_file, format="mp3")
            os.remove(temp_wav_path)
            messagebox.showinfo("Éxito", f"Audio generado: {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el audio: {e}")
    
def update_voice_options(*args):
    """Actualiza las opciones de voz según el idioma seleccionado."""
    selected_language = language_var.get()
    voices = voice_options.get(selected_language, [])
    menu = voice_option_menu["menu"]
    menu.delete(0, "end")
    for voice in voices:
        menu.add_command(label=voice, command=lambda v=voice: voice_var.set(v))
    if voices:
        voice_var.set(voices[0])

def update_bark_options_visibility():
    """Si el modelo seleccionado contiene 'bark', muestra el botón para abrir opciones de tags."""
    selected_model = voice_var.get()
    if "bark" in selected_model.lower():
        bark_tags_button.pack(pady=5)
    else:
        bark_tags_button.forget()

# -------------------------
# CREACIÓN DE LA INTERFAZ GRÁFICA
# -------------------------
Label(app, text="Introduce tu texto:", font=("Arial", 14)).pack(pady=5)
text_input = Text(app, wrap="word", height=5)
text_input.pack(padx=10, pady=5, fill="both", expand=True)

Label(app, text="Idioma:", font=("Arial", 12)).pack(pady=5)
OptionMenu(app, language_var, *language_options).pack(pady=5)

Label(app, text="Voz:", font=("Arial", 12)).pack(pady=5)
voice_option_menu = OptionMenu(app, voice_var, *voice_options[language_options[0]])
voice_option_menu.pack(pady=5)

# Botón para abrir opciones de tags para Bark
bark_tags_button = Button(app, text="Insertar Tags Bark", font=("Arial", 12), command=open_bark_tags)
bark_tags_button.forget()

Button(app, text="Grabar Referencia", font=("Arial", 12), command=record_reference).pack(pady=5)
Label(app, text="Seleccionar referencia guardada:", font=("Arial", 12)).pack()
reference_listbox = Listbox(app, height=5)
reference_listbox.pack(padx=10, pady=5, fill="both")
Button(app, text="Seleccionar", command=select_reference).pack(pady=5)

Button(app, text="Generar Audio", font=("Arial", 14), command=generate_audio).pack(pady=10)

language_var.trace("w", update_voice_options)
voice_var.trace("w", lambda *args: update_bark_options_visibility())

update_voice_options()
update_reference_list()



app.mainloop()

