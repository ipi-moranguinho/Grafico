import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
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
    ax.set_title(titulo, color='#e6edf3', fontsize=10, pad=10, fontweight='bold')
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=8, labelpad=6)
    ax.set_zlabel(zlabel, color='#8b949e', fontsize=8, labelpad=6)
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#21262d')
    ax.yaxis.pane.set_edgecolor('#21262d')
    ax.zaxis.pane.set_edgecolor('#21262d')
    ax.grid(True, alpha=0.3)


# -- Estiliza eixos 2D
def estilizar_ax2d(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, color='#e6edf3', fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=8)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------
# Gráfico 1 — Scatter 3D: CPU × Tráfego × Tempo de Execução
#   Cor do ponto proporcional ao tempo de execução (escala viridis)
# ---------------------------------------------------------------
def abrir_grafico_1(event=None):
    fig = plt.figure(figsize=(11, 8), num='[1] CPU × Tráfego × Tempo de Execução (3D)')
    fig.patch.set_facecolor('#0d1117')
    ax  = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(Z.min(), Z.max())
    sc   = ax.scatter(X, Y, Z, c=Z, cmap='viridis', norm=norm,
                      s=20, alpha=0.75, edgecolors='none', depthshade=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Execution Time (s)', color='#8b949e', fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)

    estilizar_ax3d(ax,
                   'CPU × Tráfego de Rede × Tempo de Execução\n(cor = tempo de execução)',
                   'CPU Usage (%)', 'Network Traffic (MB)', 'Execution Time (s)')

    fig.text(0.5, 0.02, 'Arraste para girar  •  Scroll para zoom',
             ha='center', fontsize=8, color='#484f58')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# Gráfico 2 — Scatter 3D: CPU × Tráfego × Tempo
#   Cor do ponto proporcional ao CPU usage (escala plasma)
# ---------------------------------------------------------------
def abrir_grafico_2(event=None):
    fig = plt.figure(figsize=(11, 8), num='[2] Intensidade de CPU no espaço 3D')
    fig.patch.set_facecolor('#0d1117')
    ax  = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(X.min(), X.max())
    sc   = ax.scatter(X, Y, Z, c=X, cmap='plasma', norm=norm,
                      s=20, alpha=0.75, edgecolors='none', depthshade=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('CPU Usage (%)', color='#8b949e', fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)

    estilizar_ax3d(ax,
                   'Intensidade de CPU\nno espaço 3D (cor = CPU %)',
                   'CPU Usage (%)', 'Network Traffic (MB)', 'Execution Time (s)')

    fig.text(0.5, 0.02, 'Arraste para girar  •  Scroll para zoom',
             ha='center', fontsize=8, color='#484f58')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# Gráfico 3 — Superfície interpolada: CPU × Tráfego → Tempo
# ---------------------------------------------------------------
def abrir_grafico_3(event=None):
    from scipy.interpolate import griddata

    fig = plt.figure(figsize=(11, 8), num='[3] Superfície: CPU × Tráfego → Tempo')
    fig.patch.set_facecolor('#0d1117')
    ax  = fig.add_subplot(111, projection='3d')

    xi = np.linspace(X.min(), X.max(), 60)
    yi = np.linspace(Y.min(), Y.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')

    surf = ax.plot_surface(Xi, Yi, Zi, cmap='inferno', alpha=0.85,
                           linewidth=0, antialiased=True)

    # Pontos reais sobrepostos
    ax.scatter(X, Y, Z, c='#58a6ff', s=4, alpha=0.3, depthshade=False)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1)
    cbar.set_label('Execution Time (s)', color='#8b949e', fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)

    estilizar_ax3d(ax,
                   'Superfície Interpolada: Tempo de Execução\nem função de CPU e Tráfego de Rede',
                   'CPU Usage (%)', 'Network Traffic (MB)', 'Execution Time (s)')

    fig.text(0.5, 0.02, 'Arraste para girar  •  Superfície via scipy griddata  •  pontos azuis = dados reais',
             ha='center', fontsize=8, color='#484f58')
    plt.tight_layout()
    plt.show()


# -- Janela principal com menu de navegação
fig_menu = plt.figure(figsize=(10, 6), num='vmCloud — Visualizador Interativo')
fig_menu.patch.set_facecolor('#0d1117')

ax_titulo = fig_menu.add_axes([0, 0.6, 1, 0.4])
ax_titulo.set_axis_off()
ax_titulo.text(0.5, 0.65, '  vmCloud Analytics', ha='center', va='center',
               fontsize=26, fontweight='bold', color='#58a6ff',
               fontfamily='monospace', transform=ax_titulo.transAxes)
ax_titulo.text(0.5, 0.25,
               f'Dataset filtrado: {len(df):,} registros  •  3 variáveis: CPU, Rede, Tempo',
               ha='center', va='center', fontsize=11, color='#8b949e',
               transform=ax_titulo.transAxes)

botoes_info = [
    ('1', 'Scatter 3D  (cor = Tempo)',     'grafico_1'),
    ('2', 'Scatter 3D  (cor = CPU)',        'grafico_2'),
    ('3', 'Superfície: CPU × Rede → Tempo', 'grafico_3'),
]

BOTOES_AX  = {}
BOTOES_OBJ = {}

for i, (num, label, key) in enumerate(botoes_info):
    col = i % 3
    row = i // 3
    ax_b = fig_menu.add_axes([0.06 + col * 0.31, 0.32 - row * 0.22, 0.28, 0.14])
    btn  = Button(ax_b, f'  [{num}]  {label}', color='#161b22', hovercolor='#1f6feb')
    btn.label.set_fontsize(9)
    btn.label.set_color('#e6edf3')
    btn.label.set_fontfamily('monospace')
    BOTOES_AX[key]  = ax_b
    BOTOES_OBJ[key] = btn

ax_info = fig_menu.add_axes([0, 0, 1, 0.08])
ax_info.set_axis_off()
ax_info.text(0.5, 0.5,
             'Clique em um gráfico para abrir  •  Gráficos 3D são giráveis com o mouse',
             ha='center', va='center', fontsize=8, color='#484f58',
             transform=ax_info.transAxes)

BOTOES_OBJ['grafico_1'].on_clicked(abrir_grafico_1)
BOTOES_OBJ['grafico_2'].on_clicked(abrir_grafico_2)
BOTOES_OBJ['grafico_3'].on_clicked(abrir_grafico_3)

# -- Resumo estatístico no terminal
print("=" * 55)
print("  vmCloud Analytics — Estatísticas Rápidas")
print("=" * 55)
for col, nome in [
    ('cpu_usage',       'CPU Usage (%)        '),
    ('network_traffic', 'Network Traffic (MB) '),
    ('execution_time',  'Execution Time (s)   '),
]:
    s = df[col]
    print(f"  {nome}  média={s.mean():.2f}  std={s.std():.2f}  "
          f"min={s.min():.2f}  max={s.max():.2f}")
print("=" * 55)
print(f"  Registros totais : {len(df):,}")
print(f"  Amostra no plot  : {len(df_plot):,}")
print("=" * 55)
print("\n  › Menu aberto. Clique em um botão para abrir o gráfico.\n")

plt.show()