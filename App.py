import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -- Tema escuro global
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
})

# -- Carrega o CSV do mesmo diretório do script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, 'vmCloud_filtrado.csv')

print(f"  › Carregando: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Valida colunas mínimas necessárias
COLUNAS_NECESSARIAS = ['cpu_usage', 'network_traffic', 'execution_time']
for col in COLUNAS_NECESSARIAS:
    if col not in df.columns:
        raise ValueError(f"Coluna ausente no CSV: '{col}'")

print(f"  › {len(df)} registros carregados")
print(f"  › Colunas: {list(df.columns)}\n")

# Limpa NaN
df = df.dropna(subset=COLUNAS_NECESSARIAS).reset_index(drop=True)

# Limita a 2000 pontos para boa performance nos scatter 3D
MAX_PONTOS = 2000
df_plot = df.sample(min(MAX_PONTOS, len(df)), random_state=42).copy()

X = df_plot['cpu_usage'].values
Y = df_plot['network_traffic'].values
Z = df_plot['execution_time'].values


# -- Estiliza eixos 3D
def estilizar_ax3d(ax, titulo, xlabel, ylabel, zlabel):
    ax.set_title(titulo, color='#e6edf3', fontsize=14, pad=18, fontweight='bold')
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=10, labelpad=10)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=10, labelpad=10)
    ax.set_zlabel(zlabel, color='#8b949e', fontsize=10, labelpad=10)
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#21262d')
    ax.yaxis.pane.set_edgecolor('#21262d')
    ax.zaxis.pane.set_edgecolor('#21262d')
    ax.xaxis._axinfo['grid']['color'] = '#2f363f'
    ax.yaxis._axinfo['grid']['color'] = '#2f363f'
    ax.zaxis._axinfo['grid']['color'] = '#2f363f'
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)


# -- Estiliza eixos 2D (mantido para uso futuro)
def estilizar_ax2d(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, color='#e6edf3', fontsize=10, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.grid(True, color='#2f363f', linestyle='--', linewidth=0.4, alpha=0.6)


# ---------------------------------------------------------------
# Gráfico 1 — Scatter 3D: CPU × Tráfego × Tempo de Execução
#   Cor do ponto proporcional ao tempo de execução (escala viridis)
# ---------------------------------------------------------------
def abrir_grafico_1(event=None):
    fig = plt.figure(figsize=(11, 8), num='vmCloud 3D — CPU × Rede × Tempo')
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(Z.min(), Z.max())
    sc = ax.scatter(
        X, Y, Z,
        c=Z, cmap='viridis', norm=norm,
        s=48, alpha=0.9, edgecolors='none', depthshade=True
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    cbar.outline.set_edgecolor('#30363d')
    cbar.set_label('Tempo de Execução (s)', color='#8b949e', fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=8)

    ax.view_init(elev=28, azim=-60)

    estilizar_ax3d(
        ax,
        'Tempo de Execução em função de CPU e Tráfego de Rede',
        'Uso da CPU (%)', 'Tráfego de Rede (MB)', 'Tempo de Execução (s)'
    )

    fig.suptitle('vmCloud — Visualização 3D dos Dados', color='#58a6ff', fontsize=18, fontweight='bold', y=0.95)
    fig.text(
        0.5, 0.02,
        'Pontos reais do dataset vmCloud — sem interpolação',
        ha='center', fontsize=9, color='#8b949e'
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


# -- Resumo estatístico no terminal
print("=" * 55)
print("  vmCloud Analytics — Estatísticas Rápidas")
print("=" * 55)
for col, nome in [
    ('cpu_usage',       'Uso da CPU (%)        '),
    ('network_traffic', 'Tráfego de Rede (MB) '),
    ('execution_time',  'Tempo de Execução (s)   '),
]:
    s = df[col]
    print(f"  {nome}  média={s.mean():.2f}  std={s.std():.2f}  "
          f"min={s.min():.2f}  max={s.max():.2f}")
print("=" * 55)
print(f"  Registros totais : {len(df):,}")
print(f"  Amostra no plot  : {len(df_plot):,}")
print("=" * 55)
print("\n  › Exibindo apenas o scatter 3D dos dados.\n")

abrir_grafico_1()