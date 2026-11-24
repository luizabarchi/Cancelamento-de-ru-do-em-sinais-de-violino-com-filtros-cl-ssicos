#Estudo de caso - Analise de Sistemas Lineares (UTFPR Apucarana)
#Cancelamento de ruido em sinais de violino com filtros classicos.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

#CONFIGURACAO
NOTAS = {# nome : (arquivo, f0 nominal em Hz)
    "Grave (Re4)": ("re4.wav", 293.7),
    "Media (La4)": ("la4.wav", 440.0),
    "Aguda (Mi5)": ("mi5.wav", 659.3),
}
SNR_ENTRADA_DB = 10.0          # nivel do ruido adicionado
FC_PASSA_ALTAS = 200.0         # Hz (abaixo da nota mais grave, 293,7 Hz)
FC_PASSA_BAIXAS = 5000.0       # Hz
ORDEM = 4                      # ordem dos filtros Butterworth
Q_NOTCH = 30.0                 # fator de qualidade do notch
FREQ_REDE = [60.0, 120.0, 180.0]
JANELA_MS, PASSO_MS = 50, 25   # janelas deslizantes (media e variancia)
N_HARMONICAS = 10
TESTE_CRUZADO = True           # True: cada ruido passa pelos 3 filtros (27 casos)
                               # False: so o filtro "esperado" (9 casos)

SEMENTE = 42                   # garante que o ruido seja sempre o mesmo
PASTA = "resultados"


rng = np.random.default_rng(SEMENTE)
os.makedirs(PASTA, exist_ok=True)
RUIDOS = ["Zumbido", "Baixa frequencia", "Ruido branco"]
FILTROS = ["Notch", "Passa-altas", "Passa-baixas"]
ESPERADO = dict(zip(RUIDOS, FILTROS))


#ENTRADA
def carregar(caminho):
    #Le o wav, converte para mono, normaliza e remove o silencio das pontas.
    fs, x = wavfile.read(caminho)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= np.max(np.abs(x))
    env = np.abs(signal.hilbert(x))
    env = np.convolve(env, np.ones(int(0.02 * fs)) / int(0.02 * fs), "same")
    ativo = np.where(env > 0.05 * env.max())[0]
    return fs, x[ativo[0]:ativo[-1]]


#RUIDOS
def gerar_ruido(tipo, n, fs):
    #gerando ruído para somar ao áudio original
    t = np.arange(n) / fs
    if tipo == "Zumbido":
        r = sum(a * np.sin(2 * np.pi * f * t)
                for f, a in zip(FREQ_REDE, [1.0, 0.5, 0.25]))
    elif tipo == "Baixa frequencia":
        sos = signal.butter(4, 100, "low", fs=fs, output="sos")
        r = signal.sosfilt(sos, rng.standard_normal(n))
    else:
        r = rng.standard_normal(n)
    return r


def somar_ruido(x, r, snr_db):
    #Escala r para que 10*log10(Px/Pr) = snr_db e soma a x.
    r = r * np.sqrt(np.mean(x**2) / (np.mean(r**2) * 10**(snr_db / 10)))
    return x + r, r

#FILTROS
def aplicar_filtro(nome, y, fs):
    #Todos os filtros sao aplicados com fase zero (sosfiltfilt)
    if nome == "Notch":
        for f in FREQ_REDE:
            b, a = signal.iirnotch(f, Q_NOTCH, fs=fs)
            y = signal.filtfilt(b, a, y)
        return y
    if nome == "Passa-altas":
        sos = signal.butter(ORDEM, FC_PASSA_ALTAS, "high", fs=fs, output="sos")
    else:
        sos = signal.butter(ORDEM, FC_PASSA_BAIXAS, "low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, y)

#METRICAS
def snr_db(ref, y):
    return 10 * np.log10(np.sum(ref**2) / np.sum((y - ref)**2))

def amplitudes_harmonicas(x, fs, f0, n_harm):
    """Amplitude (FFT com janela Hann) do pico proximo de cada k*f0."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    # refina f0 pelo pico mais proximo do valor nominal
    banda = (f > 0.9 * f0) & (f < 1.1 * f0)
    f0r = f[banda][np.argmax(X[banda])]
    amps = []
    for k in range(1, n_harm + 1):
        if k * f0r > fs / 2 * 0.95:
            break
        b = (f > k * f0r * 0.98) & (f < k * f0r * 1.02)
        amps.append(X[b].max())
    return f0r, np.array(amps)

def preservacao(x, y, fs, f0):
    f0r, ax = amplitudes_harmonicas(x, fs, f0, N_HARMONICAS)
    _, ay = amplitudes_harmonicas(y, fs, f0r, N_HARMONICAS)
    return 20 * np.log10(ay[:len(ax)] / ax)      # dB por harmonica


def janelas(x, fs):
    n, p = int(JANELA_MS * fs / 1000), int(PASSO_MS * fs / 1000)
    ini = np.arange(0, len(x) - n, p)
    m = np.array([x[i:i + n].mean() for i in ini])
    v = np.array([x[i:i + n].var() for i in ini])
    return (ini + n / 2) / fs, m, v


#GRAFICOS
def espectro_db(x, fs):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)
    return np.fft.rfftfreq(len(x), 1 / fs), 20 * np.log10(X + 1e-12)

def fig_limpo(dados):
    fig, ax = plt.subplots(len(dados), 2, figsize=(11, 3 * len(dados)))
    for i, (nome, d) in enumerate(dados.items()):
        fs, x, f0 = d["fs"], d["x"], d["f0"]
        t = np.arange(len(x)) / fs
        ax[i, 0].plot(t, x, lw=.5)
        ax[i, 0].set_title(f"{nome} - forma de onda")
        ax[i, 0].set_xlabel("Tempo (s)"); ax[i, 0].set_ylabel("Amplitude")
        f, S = espectro_db(x, fs)
        ax[i, 1].plot(f, S, lw=.6)
        for k in range(1, N_HARMONICAS + 1):
            ax[i, 1].axvline(k * f0, color="r", ls=":", lw=.6)
        ax[i, 1].set_xlim(0, 5000)
        ax[i, 1].set_title(f"{nome} - espectro (linhas vermelhas: k*f0)")
        ax[i, 1].set_xlabel("Frequencia (Hz)"); ax[i, 1].set_ylabel("dB")
    fig.tight_layout(); fig.savefig(f"{PASTA}/fig1_sinais_limpos.png", dpi=200)
    plt.close(fig)


def fig_espectros(nota, d, casos):
    fig, ax = plt.subplots(len(RUIDOS), 1, figsize=(10, 3 * len(RUIDOS)))
    for a, rd in zip(ax, RUIDOS):
        c = casos[(rd, ESPERADO[rd])]
        for sinal, rot, cor in [(d["x"], "Limpo", "k"),
                                (c["y_ruidoso"], "Ruidoso", "C3"),
                                (c["y_filtrado"], "Filtrado", "C2")]:
            f, S = espectro_db(sinal, d["fs"])
            a.plot(f, S, lw=.6, color=cor, label=rot, alpha=.8)
        a.set_xlim(0, 5000)
        a.set_title(f"{nota} - {rd} ({ESPERADO[rd]})")
        a.set_xlabel("Frequencia (Hz)"); a.set_ylabel("dB"); a.legend()
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig2_espectros_{nota[:5].strip()}.png", dpi=200)
    plt.close(fig)


def fig_janelas(nota, d, casos):
    fig, ax = plt.subplots(len(RUIDOS), 2, figsize=(11, 3 * len(RUIDOS)))
    for i, rd in enumerate(RUIDOS):
        c = casos[(rd, ESPERADO[rd])]
        for sinal, rot, cor in [(d["x"], "Limpo", "k"),
                                (c["y_ruidoso"], "Ruidoso", "C3"),
                                (c["y_filtrado"], "Filtrado", "C2")]:
            t, m, v = janelas(sinal, d["fs"])
            ax[i, 0].plot(t, m, color=cor, label=rot, lw=.8)
            ax[i, 1].plot(t, v, color=cor, label=rot, lw=.8)
        ax[i, 0].set_title(f"{rd} - media (janela {JANELA_MS} ms)")
        ax[i, 1].set_title(f"{rd} - variancia")
        for j in (0, 1):
            ax[i, j].set_xlabel("Tempo (s)"); ax[i, j].legend()
    fig.suptitle(nota); fig.tight_layout()
    fig.savefig(f"{PASTA}/fig3_janelas_{nota[:5].strip()}.png", dpi=200)
    plt.close(fig)


def fig_espectrogramas(nota, d, casos):
    fig, ax = plt.subplots(len(RUIDOS), 3, figsize=(14, 3 * len(RUIDOS)))
    for i, rd in enumerate(RUIDOS):
        c = casos[(rd, ESPERADO[rd])]
        for j, (sinal, rot) in enumerate([(d["x"], "Limpo"),
                                          (c["y_ruidoso"], "Ruidoso"),
                                          (c["y_filtrado"], "Filtrado")]):
            f, t, S = signal.spectrogram(sinal, d["fs"], nperseg=2048,
                                         noverlap=1536, window="hann")
            ax[i, j].pcolormesh(t, f, 10 * np.log10(S + 1e-12),
                                shading="auto", vmin=-110, vmax=-20)
            ax[i, j].set_ylim(0, 4000)
            ax[i, j].set_title(f"{rd} - {rot}")
            ax[i, j].set_xlabel("Tempo (s)"); ax[i, j].set_ylabel("Hz")
    fig.suptitle(nota); fig.tight_layout()
    fig.savefig(f"{PASTA}/fig4_espectrogramas_{nota[:5].strip()}.png", dpi=200)
    plt.close(fig)


def fig_matriz(df):
    nomes = list(NOTAS)
    fig, ax = plt.subplots(1, len(nomes), figsize=(5 * len(nomes), 4))
    for a, nota in zip(np.atleast_1d(ax), nomes):
        M = df[df.Nota == nota].pivot(index="Ruido", columns="Filtro",
                                      values="Ganho_SNR_dB").loc[RUIDOS, FILTROS]
        im = a.imshow(M.values, cmap="viridis")
        a.set_xticks(range(3)); a.set_xticklabels(FILTROS)
        a.set_yticks(range(3)); a.set_yticklabels(RUIDOS)
        for (r, c), v in np.ndenumerate(M.values):
            a.text(c, r, f"{v:.1f}", ha="center", va="center", color="w")
        a.set_title(f"{nota} - ganho de SNR (dB)")
        plt.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(f"{PASTA}/fig5_matriz_ganho_snr.png", dpi=200)
    plt.close(fig)


def fig_harmonicas(registro):
    fig, ax = plt.subplots(1, len(NOTAS), figsize=(5 * len(NOTAS), 4),
                           sharey=True)
    for a, nota in zip(np.atleast_1d(ax), NOTAS):
        for (n, rd, fl), h in registro.items():
            if n == nota and fl == ESPERADO[rd]:
                a.plot(range(1, len(h) + 1), h, "o-", label=f"{rd} / {fl}")
        a.axhline(0, color="k", lw=.5)
        a.set_title(nota); a.set_xlabel("Harmonica k")
        a.set_ylabel("Variacao de amplitude (dB)"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{PASTA}/fig6_preservacao_harmonicas.png", dpi=200)
    plt.close(fig)


#PRINCIPAL
def main():
    dados, linhas, registro = {}, [], {}
    for nota, (arq, f0) in NOTAS.items():
        fs, x = carregar(arq)
        dados[nota] = dict(fs=fs, x=x, f0=f0)
        casos = {}
        for rd in RUIDOS:
            y_r, r = somar_ruido(x, gerar_ruido(rd, len(x), fs), SNR_ENTRADA_DB)
            for fl in FILTROS:
                if not TESTE_CRUZADO and fl != ESPERADO[rd]:
                    continue
                y_f = aplicar_filtro(fl, y_r, fs)
                h = preservacao(x, y_f, fs, f0)
                registro[(nota, rd, fl)] = h
                casos[(rd, fl)] = dict(y_ruidoso=y_r, y_filtrado=y_f)
                linhas.append(dict(
                    Nota=nota, Ruido=rd, Filtro=fl,
                    SNR_entrada_dB=snr_db(x, y_r), SNR_saida_dB=snr_db(x, y_f),
                    Ganho_SNR_dB=snr_db(x, y_f) - snr_db(x, y_r),
                    MSE=np.mean((x - y_f)**2),
                    Correlacao=np.corrcoef(x, y_f)[0, 1],
                    Harm_media_abs_dB=np.mean(np.abs(h)),
                    Harm_pior_dB=h[np.argmax(np.abs(h))]))
        dados[nota]["casos"] = casos
        fig_espectros(nota, dados[nota], casos)
        fig_janelas(nota, dados[nota], casos)
        fig_espectrogramas(nota, dados[nota], casos)
    df = pd.DataFrame(linhas)
    df.to_csv(f"{PASTA}/resultados.csv", index=False, float_format="%.4f")
    fig_limpo(dados)
    fig_matriz(df)
    fig_harmonicas(registro)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(f"\nFiguras e CSV salvos em: {os.path.abspath(PASTA)}")


if __name__ == "__main__":
    main()
