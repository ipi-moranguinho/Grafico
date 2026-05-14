import pandas as pd
import os

# -- Caminhos de entrada e saída (mesmo diretório do script)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_CSV = os.path.join(SCRIPT_DIR, 'vmCloud_data.csv')
SAIDA_CSV   = os.path.join(SCRIPT_DIR, 'vmCloud_filtrado.csv')

# -- Colunas que representam X, Y e Z na regressão
COL_X = 'cpu_usage'        # X — uso de servidores
COL_Y = 'network_traffic'  # Y — tráfego do sistema
COL_Z = 'execution_time'   # Z — tempo de resposta

# -- Parâmetros do binning (Filtro 5)
N_FAIXAS         = 30   # número de intervalos ao longo do eixo X
AMOSTRAS_POR_BIN = 10   # linhas coletadas de cada faixa

# -- Parâmetro do Filtro 6
JANELA_HORAS = 1        # janela mínima entre medições da mesma VM (em horas)

# ─────────────────────────────────────────────────────────────
print("  › Carregando CSV...")
df = pd.read_csv(ARQUIVO_CSV)
print(f"     {len(df):>10,} linhas no total")

# ─────────────────────────────────────────────────────────────
# Filtro 1 — Remove linhas com nulo em qualquer uma das 3 colunas
# Sem isso a regressão recebe entradas inválidas
df = df.dropna(subset=[COL_X, COL_Y, COL_Z])
print(f"\n  [F1] Após remover nulos em X, Y e Z")
print(f"     {len(df):>10,} linhas restantes")

# ─────────────────────────────────────────────────────────────
# Filtro 2 — Mantém somente tarefas concluídas
# Tarefas 'waiting' e 'running' têm execution_time incompleto
df['task_status'] = df['task_status'].str.lower().str.strip()
df = df[df['task_status'] == 'completed']
print(f"\n  [F2] Após manter task_status = 'completed'")
print(f"     {len(df):>10,} linhas restantes")

# ─────────────────────────────────────────────────────────────
# Filtro 4 — Mantém somente o tipo de tarefa 'compute'
# Isola uma fonte de variação para deixar a regressão mais limpa
df['task_type'] = df['task_type'].str.lower().str.strip()
df = df[df['task_type'] == 'compute']
print(f"\n  [F4] Após manter task_type = 'compute'")
print(f"     {len(df):>10,} linhas restantes")

# ─────────────────────────────────────────────────────────────
# Filtro 6 — Remove medições da mesma VM feitas dentro da janela mínima
# Converte timestamp, ordena por VM e tempo, e descarta linhas onde
# a diferença para a medição anterior da mesma VM é menor que JANELA_HORAS
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df.sort_values(['vm_id', 'timestamp'])

# Calcula a diferença de tempo entre medições consecutivas da mesma VM
df['_delta'] = df.groupby('vm_id')['timestamp'].diff()

# Mantém a primeira medição de cada VM (delta NaT) e as que estão fora da janela
janela = pd.Timedelta(hours=JANELA_HORAS)
df = df[df['_delta'].isna() | (df['_delta'] >= janela)]
df = df.drop(columns='_delta')

print(f"\n  [F6] Após remover duplicatas de vm_id dentro de {JANELA_HORAS}h")
print(f"     {len(df):>10,} linhas restantes")

# ─────────────────────────────────────────────────────────────
# Filtro 5 — Amostragem por faixa (binning no eixo X)
# Divide cpu_usage em N_FAIXAS intervalos iguais e coleta
# AMOSTRAS_POR_BIN linhas de cada faixa — garante cobertura
# uniforme de todo o espectro sem amontoar pontos no centro
df['_bin'] = pd.cut(df[COL_X], bins=N_FAIXAS, labels=False)

df_final = (
    df.groupby('_bin', group_keys=False)
      .apply(lambda g: g.sample(min(AMOSTRAS_POR_BIN, len(g)), random_state=42))
      .reset_index(drop=True)
)

# Remove a coluna auxiliar caso ainda exista (comportamento varia por versão do pandas)
if '_bin' in df_final.columns:
    df_final = df_final.drop(columns='_bin')

print(f"\n  [F5] Após binning ({N_FAIXAS} faixas × {AMOSTRAS_POR_BIN} amostras)")
print(f"     {len(df_final):>10,} linhas restantes")

# ─────────────────────────────────────────────────────────────
# Salva apenas as colunas necessárias para o GeoGebra
df_final[[COL_X, COL_Y, COL_Z]].to_csv(SAIDA_CSV, index=False)

print(f"\n  › Arquivo salvo em: {SAIDA_CSV}")
print(f"  › Colunas exportadas: {COL_X}, {COL_Y}, {COL_Z}")
print(f"  › Pronto para importar no GeoGebra.\n")