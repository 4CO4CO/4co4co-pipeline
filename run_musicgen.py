import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MusicGen.get_pretrained('facebook/musicgen-small')  # <-- 최신 방식 권장
model.set_generation_params(duration=10)  # 8초짜리 음악 생성

descriptions = ["Cinderella is dancing her way to the Grand ballroom."]

wav = model.generate(descriptions, progress=True)

audio_write("sample1", wav[0].cpu(), model.sample_rate, strategy="loudness")
