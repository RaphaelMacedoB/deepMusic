import os
import re
import librosa
import numpy as np

file_path = 'musicas'
output_path = 'dataset'

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
        caminho_arquivo = os.path.join(file_path, file)
        y, sr = librosa.load(caminho_arquivo)
        croma = librosa.feature.chroma_cqt(y=y, sr=sr)
        croma_normalizado = librosa.util.normalize(croma)  # Normalização dos cromas
        cromas.append(croma_normalizado)
        rotulos.append(obter_tonalidade(file))
        print(caminho_arquivo,obter_tonalidade(file))

# Extrair os cromas e rótulos
obter_cromas(folder)

# Salvar os dados em arquivos .npy com os respectivos rótulos
for i, croma in enumerate(cromas):
    np.save(os.path.join(output_path, f'{i+1}.npy'), croma)
    np.save(os.path.join(output_path, f'{i+1}_label.npy'), rotulos[i])

print('Dados salvos em', output_path)
