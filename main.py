import os
import re
import librosa
import numpy as np

file_path = 'musicas'

folder = os.listdir(file_path)

cromas = []
rotulos = []

padrao = r"\d+([A-G](?:#|b)?m?)\.mp3"

def obter_tonalidade(nome_arquivo):
    match = re.search(padrao, nome_arquivo)
    if match:
        return match.group(1)
    else:
        return None

def obter_cromas(files):
    for file in files:
        caminho_arquivo = os.path.join('musicas',file)
        y, sr = librosa.load(caminho_arquivo)
        croma = librosa.feature.chroma_cqt(y=y, sr=sr)
        cromas.append(croma)
        rotulos.append(obter_tonalidade(file))

obter_cromas(folder)

i = 1

for croma in cromas:
    np.save(f'dataset/{i}', croma)
    i += 1
    