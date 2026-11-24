# Cancelamento-de-ruido-em-sinais-de-violino-com-filtros-classicos
Código utilizado no Estudo de Caso de Análise de Sistemas Lineares - Engenharia de Computação UTFPR campus Apucarana

## Como rodar
1. Instale o Python 3 e as bibliotecas: pip install numpy scipy matplotlib pandas
2. Deixe os arquivos re4.wav, la4.wav e mi5.wav na mesma pasta do script.
3. Execute: python experimento_violino.py
4. Os gráficos e o resultados.csv são salvos na pasta "resultados".

## Dados
Gravações de um violino (notas Ré4, Lá4 e Mi5) feitas com um celular em um quarto, convertidas para WAV (mono, 48 kHz).

## Parâmetros
SNR de entrada: 10 dB. Notch em 60, 120 e 180 Hz (Q = 30). Passa-altas Butterworth de 4ª ordem em 200 Hz. Passa-baixas Butterworth de 4ª ordem em 5 kHz. Semente do ruído: 42.

Autores: [Luiza Lavagnoli Barchi, Lucas Kimura de Oliveira Mendes, Maria Clara da Silva, Mayane Rye Aomoto Ioaoka, Vinicius de Freitas] 
