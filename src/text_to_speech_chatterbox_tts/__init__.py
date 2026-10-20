import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print ("Cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)
texto = "Hola , bienvenidos a Inteligencia artificial generativa, vamos a crear audios , imagenes, texto, y si el pc aguanta videos :)"
audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.8,# expresion o emocion
    cfg_weight=0.4, # nos ajusta el  ritmo, si la voz va rapida
    temperature=0.8, # control la vairabilidad, menos predecible
    repetition_penalty=1.2,#penaliza repeticiones extrañas
    min_p=0.05, #filtra opciones improbables durante la generacion
    top_p=1.0 # controla el muestreo probabilistico

    
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)
print("audio guardado.")
