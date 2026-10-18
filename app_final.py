"""
TrainHer — App Streamlit Único (Consolidado)
==================================================
Junta app.py + app_ui.py em uma única versão, resolvendo as
divergências entre eles para ficar 100% compatível com os
artefatos gerados por pipeline_final.py:

  - Carregamento do modelo com tratamento de erro (de app_ui.py)
  - Nome de feature "Em média qual a duração do seu treino? (minutos)"
    (o que pipeline_final.py realmente usa e salva em feature_cols.csv)
  - Escalonamento via scaler.feature_names_in_ (compatível com o
    StandardScaler treinado em cima de um DataFrame)
  - Artefatos lidos da pasta artefatos/ (mesma pasta de saída do pipeline)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import datetime
from pathlib import Path

# ── Configuração da Página ──
import base64
from PIL import Image

APP_NAME = "TrainHer"
_BASE = Path(__file__).parent
LOGO_PATH = _BASE / "logo_trainher_header.jpeg"
ICON_PATH = _BASE / "trainher_icon.png"


def logo_b64() -> str:
    """Logo em base64 para embutir em HTML (vazio se o arquivo não existir)."""
    try:
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except Exception:
        return ""


try:
    _page_icon = Image.open(ICON_PATH)
except Exception:
    _page_icon = "🧘‍♀️"

st.set_page_config(page_title=APP_NAME, page_icon=_page_icon, layout="centered")

# ══════════════════════════════════════════════
# DESIGN — paleta por fase do ciclo + estilo global
# ══════════════════════════════════════════════
# Cada fase tem sua própria cor: não é decoração, é o estado hormonal
# real sendo comunicado visualmente de forma consistente no app inteiro.
PHASE_STYLE = {
    "Menstrual":  {"cor": "#C1435A", "clara": "#F8E6EA", "emoji": "🩸"},
    "Folicular":  {"cor": "#4C9A6B", "clara": "#E5F1EA", "emoji": "🌱"},
    "Ovulatória": {"cor": "#C98A1F", "clara": "#FAF0DC", "emoji": "☀️"},
    "Lútea":      {"cor": "#5B5EA6", "clara": "#E9E9F5", "emoji": "🌙"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #111111;
    --ink-soft: #5B5B5B;
    --bg: #F4F5F8;
    --card-bg: #FFFFFF;
    --card-border: #EBEBEF;
    --card-shadow: 0 2px 18px rgba(17,17,17,0.06);
    --plum: #6B3F63;
    --plum-dark: #45283F;
    --sidebar-bg: #17121E;
    --sidebar-bg-2: #241B2F;
}

.stApp { background: var(--bg); color-scheme: light; }

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }

/* Força cor do texto em todo o app — corrige o tema escuro automático do
   navegador deixando texto branco invisível sobre o fundo claro. Rótulos e
   textos em preto (#111111), como pedido. */
.stApp p, .stApp span, .stApp label, .stApp div, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    background-image: none !important;
}

h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink); letter-spacing: -0.01em; }

/* ── Sidebar premium (fundo escuro, texto claro) ── */
.stApp section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--sidebar-bg-2) 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stApp section[data-testid="stSidebar"] * {
    color: #F3EDF2 !important;
    -webkit-text-fill-color: #F3EDF2 !important;
}
.stApp section[data-testid="stSidebar"] input {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}
.stApp section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    width: 100%;
}
.stApp section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.16); }

/* ── Botões ── */
.stApp .stButton > button {
    background: var(--plum);
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stApp .stButton > button p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:hover {
    background: var(--plum-dark);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(107,63,99,0.28);
}

/* ── Abas ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--card-border); }
.stTabs [data-baseweb="tab"] {
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    color: #8A8A8A;
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 1rem;
}
.stTabs [aria-selected="true"] { color: var(--plum) !important; background: #F3EDF2; }

/* ── Métricas ── */
div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: var(--card-shadow);
}
.stApp div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--plum) !important;
    -webkit-text-fill-color: var(--plum) !important;
}
.stApp div[data-testid="stMetricLabel"] {
    font-family: 'Manrope', sans-serif;
    color: var(--ink-soft) !important;
    -webkit-text-fill-color: var(--ink-soft) !important;
}

/* ── Cards (containers com borda) — brancos, sombra suave ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border-color: var(--card-border) !important;
    background: var(--card-bg) !important;
    box-shadow: var(--card-shadow);
}

/* ── Inputs ── */
.stApp input, .stApp textarea, .stApp select {
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

/* Campos numéricos — fundo branco e texto preto. */
.stApp [data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

/* ── Classe utilitária de card branco (usada nos painéis custom em HTML) ── */
.ctai-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    box-shadow: var(--card-shadow);
    padding: 1.3rem 1.5rem;
}

/* ── Caixas/controles: fundo branco ── */
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="textarea"] > div,
.stApp [data-testid="stTextInput"] > div,
.stApp [data-testid="stNumberInput"] > div {
    background-color: #FFFFFF !important;
}

.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

.stApp [data-baseweb="select"] svg {
    fill: #111111 !important;
}


/* ── Sliders personalizados ──
   Frequência = roxo | Duração = rosa/vermelho
   O restante do layout continua claro e com caixas brancas.
*/
.stApp [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background: #E1E3E8 !important;
    height: 5px !important;
    border-radius: 999px !important;
}

.stApp [data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background: #6B3F63 !important;
    height: 5px !important;
    border-radius: 999px !important;
}

/* Bolinha/indicador do slider */
.stApp [data-testid="stSlider"] [role="slider"] {
    background: #6B3F63 !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Duração do treino: rosa/vermelho */
.stApp [data-testid="stSlider"]:has([aria-label="Duração do treino (minutos)"])
    [data-baseweb="slider"] > div > div:nth-child(2) {
    background: #FF4F5E !important;
}

.stApp [data-testid="stSlider"]:has([aria-label="Duração do treino (minutos)"])
    [role="slider"] {
    background: #FF4F5E !important;
}

/* Frequência semanal: roxo */
.stApp [data-testid="stSlider"]:has([aria-label="Frequência semanal de treinos"])
    [data-baseweb="slider"] > div > div:nth-child(2) {
    background: #8E4BB8 !important;
}

.stApp [data-testid="stSlider"]:has([aria-label="Frequência semanal de treinos"])
    [role="slider"] {
    background: #8E4BB8 !important;
}

/* Valor mostrado acima do indicador */
.stApp [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artefatos"
DB_PATH = BASE_DIR / "cycletrain.db"


# ══════════════════════════════════════════════
# 1. BANCO DE DADOS (SQLite)
# ══════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS registros
                 (username TEXT, data TEXT, fase TEXT, score REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS checkins_diarios
                 (username TEXT, data TEXT, dor INTEGER, humor TEXT,
                  energia INTEGER, sono TEXT, fluxo TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS pss10_respostas
                 (username TEXT, data TEXT, score_total INTEGER, respostas TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS mqs_respostas
                 (username TEXT, data TEXT, score_total INTEGER, respostas TEXT)""")
    conn.commit()
    conn.close()


def add_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user


def salvar_registro(username, fase, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO registros (username, data, fase, score) VALUES (?, ?, ?, ?)",
        (username, data_atual, fase, score),
    )
    conn.commit()
    conn.close()


def salvar_checkin(username, dor, humor, energia, sono, fluxo):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO checkins_diarios (username, data, dor, humor, energia, sono, fluxo) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, data_atual, dor, humor, energia, sono, fluxo),
    )
    conn.commit()
    conn.close()


def salvar_pss10(username, score_total, respostas):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO pss10_respostas (username, data, score_total, respostas) VALUES (?, ?, ?, ?)",
        (username, data_atual, score_total, str(respostas)),
    )
    conn.commit()
    conn.close()


def salvar_mqs(username, score_total, respostas):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO mqs_respostas (username, data, score_total, respostas) VALUES (?, ?, ?, ?)",
        (username, data_atual, score_total, str(respostas)),
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
# PSS-10 — Escala de Estresse Percebido
# ══════════════════════════════════════════════
# Perguntas parafraseadas (mesmo constructo do instrumento original de
# Cohen, Kamarck & Mermelstein, texto próprio para não reproduzir o
# questionário original protegido por direitos autorais).
PSS10_PERGUNTAS = [
    "No último mês, com que frequência você se sentiu incomodada(o) por algo que aconteceu sem você esperar?",
    "No último mês, com que frequência você sentiu que não conseguia controlar as coisas importantes da sua vida?",
    "No último mês, com que frequência você se sentiu nervosa(o) ou estressada(o)?",
    "No último mês, com que frequência você se sentiu confiante na sua capacidade de resolver problemas pessoais?",
    "No último mês, com que frequência você sentiu que as coisas estavam acontecendo do jeito que você queria?",
    "No último mês, com que frequência você sentiu que não conseguiria dar conta de tudo que precisava fazer?",
    "No último mês, com que frequência você conseguiu controlar as irritações da sua vida?",
    "No último mês, com que frequência você sentiu que estava por cima da situação?",
    "No último mês, com que frequência você ficou irritada(o) por causa de coisas fora do seu controle?",
    "No último mês, com que frequência você sentiu que as dificuldades estavam se acumulando a ponto de você não conseguir superá-las?",
]

PSS10_OPCOES = {
    0: "Nunca",
    1: "Quase nunca",
    2: "Às vezes",
    3: "Frequentemente",
    4: "Muito frequentemente",
}

# Itens com pontuação invertida (1-indexado, conforme especificação: 4, 5, 7, 8)
PSS10_ITENS_INVERTIDOS = {4, 5, 7, 8}


def calcular_pss10(respostas: list) -> int:
    """respostas: lista de 10 valores de 0 a 4, na ordem das perguntas (item 1 a 10)."""
    total = 0
    for i, valor in enumerate(respostas, start=1):
        if i in PSS10_ITENS_INVERTIDOS:
            total += 4 - valor
        else:
            total += valor
    return total


# ══════════════════════════════════════════════
# MQS — Escala de Percepção Subjetiva
# ══════════════════════════════════════════════
# Cada item: (categoria, texto da pergunta, é_item_positivo)
# Itens positivos (Disposição/Energia, Qualidade do Sono, Foco e Motivação)
# já vêm "alto = bom"; os demais são invertidos no cálculo do score para
# que, no resultado final, um valor alto sempre represente algo favorável.
MQS_PERGUNTAS = [
    ("MINI QUESTIONÁRIO DO SONO.",
     "Você tem dificuldade em adormecer à noite?", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     "Você acorda de madrugada e não consegue adormecer de novo?", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     "Você toma remédios para dormir ou tranquilizantes?   ", True),
    ("MINI QUESTIONÁRIO DO SONO.",
     "Você dorme durante o dia? (sem contar cochilos ou sonecas programados)", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     "Ao acordar de manhã, você ainda se sente cansado(a)?", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     "Você ronca à noite (que você saiba)?", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     " Você acorda durante a noite?", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     " Você sente cansaço sem ter nenhum motivo aparente?", False),
    ("MINI QUESTIONÁRIO DO SONO.",
     "Você tem sono agitado? (mudanças constantes de posição ou movimentos de pernas/braços)", False),

]

MQS_OPCOES = {
    1: "Nunca",
    2: "Muito rara",
    3: "Rara ",
    4: "Às vezes",
    5: "Frequente",
    6: "Muito frequente",
    7: "Sempre",
}


def calcular_mqs(respostas: list) -> int:
    """respostas: lista de valores 1-7 na mesma ordem de MQS_PERGUNTAS.
    Inverte os itens negativos para que, no total, valor alto = resultado favorável."""
    total = 0
    for (categoria, texto, positivo), valor in zip(MQS_PERGUNTAS, respostas):
        total += valor if positivo else (8 - valor)  # inversão: 1↔7, 2↔6, 3↔5, 4↔4
    return total


def buscar_checkin_hoje(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hoje = datetime.date.today().strftime("%Y-%m-%d")
    c.execute(
        "SELECT dor, humor, energia, sono, fluxo FROM checkins_diarios "
        "WHERE username=? AND data LIKE ? ORDER BY data DESC LIMIT 1",
        (username, f"{hoje}%"),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"dor": row[0], "humor": row[1], "energia": row[2], "sono": row[3], "fluxo": row[4]}
    return None


def buscar_pss10_recente(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT score_total, data FROM pss10_respostas WHERE username=? ORDER BY data DESC LIMIT 1",
        (username,),
    )
    row = c.fetchone()
    conn.close()
    return {"score_total": row[0], "data": row[1]} if row else None


def buscar_mqs_recente(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT score_total, data FROM mqs_respostas WHERE username=? ORDER BY data DESC LIMIT 1",
        (username,),
    )
    row = c.fetchone()
    conn.close()
    return {"score_total": row[0], "data": row[1]} if row else None


def buscar_ultimo_registro(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT fase, score, data FROM registros WHERE username=? ORDER BY data DESC LIMIT 1",
        (username,),
    )
    row = c.fetchone()
    conn.close()
    return {"fase": row[0], "score": row[1], "data": row[2]} if row else None


def buscar_historico_registros(username):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT data, fase, score FROM registros WHERE username=? ORDER BY data ASC",
        conn, params=(username,),
    )
    conn.close()
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])
    return df


def buscar_historico_checkins(username):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT data, dor, energia, humor, sono, fluxo FROM checkins_diarios "
        "WHERE username=? ORDER BY data ASC",
        conn, params=(username,),
    )
    conn.close()
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])
    return df


def buscar_historico_pss10(username):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT data, score_total FROM pss10_respostas WHERE username=? ORDER BY data ASC",
        conn, params=(username,),
    )
    conn.close()
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])
    return df


def buscar_historico_mqs(username):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT data, score_total FROM mqs_respostas WHERE username=? ORDER BY data ASC",
        conn, params=(username,),
    )
    conn.close()
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])
    return df


def render_gauge(score, cor, label="Score de Prontidão", tamanho=150):
    """Gauge circular (donut) em CSS puro — painel visual do resultado da IA."""
    pct = max(0.0, min(100.0, (score / 10.0) * 100))
    interno = int(tamanho * 0.72)
    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:0.5rem;">
      <div style="width:{tamanho}px; height:{tamanho}px; border-radius:50%;
                  background: conic-gradient({cor} {pct}%, #EDEDF2 {pct}% 100%);
                  display:flex; align-items:center; justify-content:center;
                  box-shadow: var(--card-shadow);">
        <div style="width:{interno}px; height:{interno}px; border-radius:50%; background:#FFFFFF;
                    display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <span style="font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600; color:{cor} !important;">
            {score:.1f}
          </span>
          <span style="font-size:0.65rem; color:#5B5B5B !important;">/ 10.0</span>
        </div>
      </div>
      <span style="font-family:'Manrope',sans-serif; font-weight:600; font-size:0.8rem; color:#111111 !important; text-align:center;">
        {label}
      </span>
    </div>
    """


def render_timeline_ciclo(dia_ciclo, dur_ciclo, dur_mens):
    """Barra horizontal mostrando as 4 fases do ciclo, com marcador no dia atual."""
    dur_ciclo = max(dur_ciclo, 1)
    dia_norm = dia_ciclo % dur_ciclo if dur_ciclo > 0 else dia_ciclo
    ov_start = max(dur_ciclo - 16, dur_mens + 3)
    ov_end = min(dur_ciclo - 11, ov_start + 4)

    segmentos = [
        (0, dur_mens, PHASE_STYLE["Menstrual"]["cor"], "Menstrual"),
        (dur_mens, ov_start, PHASE_STYLE["Folicular"]["cor"], "Folicular"),
        (ov_start, ov_end, PHASE_STYLE["Ovulatória"]["cor"], "Ovulatória"),
        (ov_end, dur_ciclo, PHASE_STYLE["Lútea"]["cor"], "Lútea"),
    ]
    barras = "".join(
        f'<div title="{nome}" style="flex:{max(fim - ini, 0.3)}; background:{cor}; height:100%;"></div>'
        for ini, fim, cor, nome in segmentos
    )
    marcador_pct = max(0.0, min(100.0, (dia_norm / dur_ciclo) * 100))

    return f"""
    <div class="ctai-card" style="padding:1.1rem 1.4rem;">
      <div style="font-family:'Manrope',sans-serif; font-weight:700; font-size:0.75rem;
                  text-transform:uppercase; letter-spacing:0.06em; color:#5B5B5B !important; margin-bottom:0.7rem;">
        🗓️ Timeline do Ciclo — dia {dia_norm} de {dur_ciclo}
      </div>
      <div style="position:relative; padding-top:12px;">
        <div style="position:absolute; top:0; left:{marcador_pct}%; transform:translateX(-50%);
                    width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent;
                    border-top:9px solid #111111;"></div>
        <div style="display:flex; height:16px; border-radius:8px; overflow:hidden;">{barras}</div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:0.5rem;
                  font-family:'Manrope',sans-serif; font-size:0.68rem; color:#5B5B5B !important;">
        <span>🩸 Menstrual</span><span>🌱 Folicular</span><span>☀️ Ovulatória</span><span>🌙 Lútea</span>
      </div>
    </div>
    """


# Inicializa o banco ao carregar o módulo
init_db()

# ══════════════════════════════════════════════
# 2. GERENCIAMENTO DE SESSÃO (Login/Cadastro)
# ══════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ── Sidebar premium: marca + estado da sessão ──
_logo = logo_b64()
if _logo:
    st.sidebar.markdown(f"""
<div style="background:#FAFAFA; border-radius:14px; padding:0.6rem; margin:0.2rem 0 1rem 0; text-align:center;">
  <img src="data:image/jpeg;base64,{_logo}" alt="TrainHer" style="width:100%; max-width:210px; height:auto;"/>
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
<div style="padding:0.4rem 0 1rem 0;">
  <div style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.16em; font-size:0.65rem;
              color:#B9A9C4 !important; text-transform:uppercase;">TrainHer</div>
  <div style="font-family:'Fraunces',serif; font-size:1.3rem; color:#F3EDF2 !important; margin-top:0.15rem;">
    🟣 Painel
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.logged_in:
    ultimo = buscar_ultimo_registro(st.session_state.username)
    st.sidebar.markdown(f"""
    <div style="background:rgba(255,255,255,0.06); border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.8rem;">
      <div style="font-size:0.72rem; color:#B9A9C4 !important; text-transform:uppercase; letter-spacing:0.05em;">Logada como</div>
      <div style="font-family:'Manrope',sans-serif; font-weight:700; font-size:1rem; color:#F3EDF2 !important;">{st.session_state.username}</div>
    </div>
    """, unsafe_allow_html=True)
    if ultimo:
        st.sidebar.markdown(f"""
        <div style="background:rgba(255,255,255,0.06); border-radius:12px; padding:0.9rem 1rem; margin-bottom:1rem;">
          <div style="font-size:0.72rem; color:#B9A9C4 !important; text-transform:uppercase; letter-spacing:0.05em;">Última prontidão</div>
          <div style="font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1.4rem; color:#F3EDF2 !important;">{ultimo['score']:.1f} / 10.0</div>
          <div style="font-size:0.78rem; color:#B9A9C4 !important;">{ultimo['fase']}</div>
        </div>
        """, unsafe_allow_html=True)
    if st.sidebar.button("🚪 Sair"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
else:
    st.sidebar.caption("Faça login no card central para acessar seu painel.")


# ══════════════════════════════════════════════
# 3. CARREGANDO OS ARTEFATOS DA IA
# ══════════════════════════════════════════════
@st.cache_resource
def carregar_modelo():
    try:
        modelo = joblib.load(ARTIFACTS_DIR / "modelo_campeao.joblib")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        feature_cols = pd.read_csv(ARTIFACTS_DIR / "feature_cols.csv")["0"].tolist()
        return modelo, scaler, feature_cols
    except Exception as e:
        st.error(
            f"Erro ao carregar modelo: {e}. "
            f"Rode o pipeline_final.py primeiro para gerar os artefatos em '{ARTIFACTS_DIR}/'."
        )
        return None, None, []


modelo, scaler, feature_cols = carregar_modelo()


# ══════════════════════════════════════════════
# 4a. LOGIN — card centralizado (quando não logada)
# ══════════════════════════════════════════════
if not st.session_state.logged_in:
    if _logo:
        st.markdown(f"""
        <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
          <img src="data:image/jpeg;base64,{_logo}" alt="TrainHer"
               style="width:100%; max-width:340px; height:auto; mix-blend-mode:multiply;"/>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0 1.5rem 0;">
          <div style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.14em; font-size:0.72rem;
                      color:#5B5B5B !important; text-transform:uppercase;">Treino sincronizado ao ciclo</div>
          <h1 style="margin:0.25rem 0 0 0; font-size:2.2rem; color:#111111 !important;">🟣 TrainHer</h1>
        </div>
        """, unsafe_allow_html=True)

    col_esq, col_meio, col_dir = st.columns([1, 1.3, 1])
    with col_meio:
        with st.container(border=True):
            aba_login, aba_criar = st.tabs(["🔑 Entrar", "✨ Criar Conta"])

            with aba_login:
                user_input = st.text_input("Usuário", key="login_user")
                pass_input = st.text_input("Senha", type="password", key="login_pass")
                if st.button("Entrar", use_container_width=True, type="primary"):
                    if login_user(user_input, pass_input):
                        st.session_state.logged_in = True
                        st.session_state.username = user_input
                        st.rerun()
                    else:
                        st.error("Usuário ou senha incorretos.")

            with aba_criar:
                novo_user = st.text_input("Escolha um usuário", key="novo_user")
                nova_senha = st.text_input("Escolha uma senha", type="password", key="nova_senha")
                if st.button("Criar Conta", use_container_width=True):
                    if add_user(novo_user, nova_senha):
                        st.success("Conta criada com sucesso! Vá para a aba Entrar.")
                    else:
                        st.error("Nome de usuário já existe.")
    st.stop()


# ══════════════════════════════════════════════
# 4. INTERFACE PRINCIPAL (só mostra se logado)
# ══════════════════════════════════════════════

if _logo:
    st.markdown(f"""
<div style="display:flex; align-items:center; gap:1.2rem; padding: 0.3rem 0 1.2rem 0;">
  <img src="data:image/jpeg;base64,{_logo}" alt="TrainHer"
       style="width:170px; height:auto; flex-shrink:0; mix-blend-mode:multiply;"/>
  <p style="color:#6B5E69 !important; font-size:1.02rem; margin:0;">
    Sua inteligência artificial para treinos adaptados ao ciclo menstrual.
  </p>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style="padding: 0.3rem 0 1.2rem 0;">
  <div style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.14em; font-size:0.72rem;
              color:#8A7E88 !important; text-transform:uppercase;">Treino sincronizado ao ciclo</div>
  <h1 style="margin:0.25rem 0 0 0; font-size:2.4rem; color:#111111 !important;">🟣 TrainHer</h1>
  <p style="color:#6B5E69 !important; font-size:1.02rem; margin-top:0.3rem;">
    Sua inteligência artificial para treinos adaptados ao ciclo menstrual.
  </p>
</div>
""", unsafe_allow_html=True)

tab_dashboard, tab_prontidao, tab_checkin, tab_pss10, tab_mqs, tab_historico = st.tabs(
    ["🏠 Dashboard", "🔮 Calcular Prontidão", "📝 Check-in Diário", "🧠 PSS-10 (Estresse)", "📊 MQS", "📈 Histórico"]
)

with tab_dashboard:
    ultimo = buscar_ultimo_registro(st.session_state.username)
    checkin_hoje_dash = buscar_checkin_hoje(st.session_state.username)

    if ultimo:
        estilo_dash = PHASE_STYLE.get(ultimo["fase"], PHASE_STYLE["Folicular"])
        col_gauge, col_info = st.columns([1, 1.6])
        with col_gauge:
            with st.container(border=True):
                st.markdown(
                    render_gauge(ultimo["score"], estilo_dash["cor"], "Última Prontidão Calculada"),
                    unsafe_allow_html=True,
                )
        with col_info:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-family:'Manrope',sans-serif; font-weight:700; color:{estilo_dash['cor']} !important;
                            text-transform:uppercase; font-size:0.72rem; letter-spacing:0.08em;">
                    {estilo_dash['emoji']} Fase na última medição
                </div>
                <div style="font-family:'Fraunces',serif; font-size:1.6rem; color:#111111 !important; margin-top:0.1rem;">
                    {ultimo['fase']}
                </div>
                <div style="font-size:0.85rem; color:#5B5B5B !important; margin-top:0.4rem;">
                    Registrado em {ultimo['data']}
                </div>
                """, unsafe_allow_html=True)
                if checkin_hoje_dash:
                    st.success("✅ Você já fez o check-in de hoje.")
                else:
                    st.warning("⏳ Você ainda não fez o check-in de hoje.")
    else:
        with st.container(border=True):
            st.markdown("### 👋 Bem-vinda ao TrainHer!")
            st.write(
                "Você ainda não tem nenhum cálculo de prontidão registrado. "
                "Vá até a aba **🔮 Calcular Prontidão** para começar."
            )

    st.write("")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    hist_reg = buscar_historico_registros(st.session_state.username)
    hist_check = buscar_historico_checkins(st.session_state.username)
    pss10_dash = buscar_pss10_recente(st.session_state.username)
    col_kpi1.metric("Cálculos de prontidão", len(hist_reg))
    col_kpi2.metric("Check-ins registrados", len(hist_check))
    col_kpi3.metric("PSS-10 mais recente", f"{pss10_dash['score_total']}/40" if pss10_dash else "—")

with tab_checkin:
    with st.container(border=True):
        st.markdown("### 📝 Check-in Diário de Sintomas")
        st.caption("Registre como você está se sentindo hoje. Isso alimenta o histórico usado para refinar suas previsões.")

        dor_checkin = st.slider("Nível de dor hoje (0-10)", 0, 10, 0, key="checkin_dor")
        humor_checkin = st.selectbox(
            "Como está seu humor hoje?", ["Ótimo", "Bom", "Neutro", "Baixo", "Muito baixo"], key="checkin_humor"
        )
        energia_checkin = st.slider("Nível de energia hoje (0-10)", 0, 10, 5, key="checkin_energia")
        sono_checkin = st.selectbox(
            "Como foi a qualidade do seu sono?", ["Excelente", "Bom", "Adequado", "Ruim"], key="checkin_sono"
        )
        fluxo_checkin = st.selectbox(
            "Como está seu fluxo menstrual hoje (se aplicável)?",
            ["Não estou menstruada", "Leve", "Moderado", "Intenso"],
            key="checkin_fluxo",
        )

        if st.button("💾 Salvar Check-in de Hoje", use_container_width=True):
            salvar_checkin(
                st.session_state.username, dor_checkin, humor_checkin,
                energia_checkin, sono_checkin, fluxo_checkin,
            )
            st.success("Check-in salvo! Continue registrando diariamente para previsões mais precisas.")

with tab_pss10:
    with st.container(border=True):
        st.markdown("### 🧠 PSS-10 — Escala de Estresse Percebido")
        st.caption("Pense no último mês e responda com que frequência cada situação ocorreu.")

        respostas_pss10 = []
        for i, pergunta in enumerate(PSS10_PERGUNTAS, start=1):
            resposta = st.select_slider(
                f"{i}. {pergunta}",
                options=list(PSS10_OPCOES.keys()),
                format_func=lambda x: PSS10_OPCOES[x],
                key=f"pss10_{i}",
            )
            respostas_pss10.append(resposta)

        if st.button("📤 Calcular Score do PSS-10", use_container_width=True):
            score_pss10 = calcular_pss10(respostas_pss10)
            salvar_pss10(st.session_state.username, score_pss10, respostas_pss10)
            st.metric("Score de Estresse Percebido (PSS-10)", f"{score_pss10} / 40")
            if score_pss10 <= 13:
                st.success("Nível de estresse percebido baixo.")
            elif score_pss10 <= 26:
                st.warning("Nível de estresse percebido moderado.")
            else:
                st.error("Nível de estresse percebido alto.")

with tab_mqs:
    with st.container(border=True):
        st.markdown("### 📊 MQS — MINI QUESTIONÁRIO DO SONO")
        st.caption(
            "Questionário completo para calibrar a IA. "
            "REFERENTE A NOITE ANTERIOR"
        )

        respostas_mqs = []
        categoria_atual = None
        for i, (categoria, pergunta, positivo) in enumerate(MQS_PERGUNTAS, start=1):
            if categoria != categoria_atual:
                st.markdown(f"#### {categoria}")
                categoria_atual = categoria
            rotulo = pergunta + (" *(item positivo)*" if positivo else "")
            resposta = st.select_slider(
                rotulo,
                options=list(MQS_OPCOES.keys()),
                format_func=lambda x: MQS_OPCOES[x],
                key=f"mqs_{i}",
            )
            respostas_mqs.append(resposta)

        if st.button("📤 Calcular Score do MQS", use_container_width=True):
            score_mqs = calcular_mqs(respostas_mqs)
            salvar_mqs(st.session_state.username, score_mqs, respostas_mqs)
            score_min, score_max = len(MQS_PERGUNTAS), 7 * len(MQS_PERGUNTAS)
            st.metric("Score MQS (já com itens invertidos — alto = favorável)", f"{score_mqs} / {score_max}")

            faixa = score_max - score_min
            if score_mqs >= score_min + faixa * 0.66:
                st.success("Percepção geral de bem-estar/recuperação boa.")
            elif score_mqs >= score_min + faixa * 0.33:
                st.warning("Percepção geral de bem-estar/recuperação moderada.")
            else:
                st.error("Percepção geral de bem-estar/recuperação baixa — considere priorizar recuperação.")

with tab_prontidao:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 👩 Perfil")
            idade = st.number_input("Idade", min_value=11, max_value=80, value=25)
            estatura = st.number_input("Estatura (m)", min_value=1.40, max_value=2.20, value=1.65, step=0.01)
            massa = st.number_input("Massa Corporal (kg)", min_value=40.0, max_value=150.0, value=65.0, step=0.1)
            imc = massa / (estatura ** 2)
            st.info(f"IMC Calculado: {imc:.1f}")
            lesao = st.slider("Escala de dor (0-10)", 0, 10, 0)

    with col2:
        with st.container(border=True):
            st.markdown("### 🩸 Ciclo Menstrual")
            dia_ciclo = st.number_input("Em qual dia do ciclo você está hoje?", min_value=0, max_value=40, value=14)
            duracao_media = st.number_input("Duração média do seu ciclo (dias)", min_value=20, max_value=40, value=28)
            duracao_mens = st.number_input("Quantos dias dura sua menstruação?", min_value=1, max_value=10, value=5)

            regular = st.selectbox("Seu ciclo é regular?", ["Regular", "Irregular"])
            contraceptivo = st.selectbox("Usa contraceptivo hormonal?", ["Não", "Sim"])

    with st.container(border=True):
        st.markdown("### 🏋️‍♀️ Exercício")
        frequencia = st.slider("Frequência semanal de treinos", 1, 7, 3)
        duracao_treino = st.slider("Duração do treino (minutos)", 0, 240, 60, step=5)

    st.write("")

    # ── Lógica de Predição ──
    if st.button("🔮 Calcular Prontidão para o Treino", use_container_width=True, type="primary"):

        if modelo is None:
            st.warning("O modelo não foi carregado corretamente.")
            st.stop()

        def calcular_fase(dia, dur_ciclo, dur_mens):
            dia_norm = dia % dur_ciclo if dur_ciclo > 0 else dia
            ov_start = max(dur_ciclo - 16, dur_mens + 3)
            ov_end = min(dur_ciclo - 11, ov_start + 4)
            if dia_norm <= dur_mens:
                return 0  # Menstrual
            elif dia_norm < ov_start:
                return 1  # Folicular
            elif dia_norm <= ov_end:
                return 2  # Ovulatória
            else:
                return 3  # Lútea

        fase_atual = calcular_fase(dia_ciclo, duracao_media, duracao_mens)
        nomes_fases = {0: "Menstrual", 1: "Folicular", 2: "Ovulatória", 3: "Lútea"}

        dados_input = {
            "IDADE_CRONOLOGICA": idade,
            "IMC": imc,
            "ESTATURA_M": estatura,
            "MASSA_CORPORAL_KG": massa,
            "dia_ciclo": dia_ciclo,
            "duracao_ciclo_media": duracao_media,
            "QUANTOS_DIAS_DUROU_SUA_ULTIMA_MENSTRUACAO": duracao_mens,
            "SEU_CICLO_E_REGULAR_OU_IRREGULAR_enc": 1 if regular == "Regular" else 0,
            "USA_ALGUM_METODO_CONTRACEPTIVO_HORMONAL_enc": 1 if contraceptivo == "Sim" else 0,
            "POSSUI_ALGUMA_LESAO_OU_RESTRICAO_MUSCULOESQUELETICO_enc": 1 if lesao >= 5 else 0,
            "QUAL_FREQUENCIA_QUE_VOCE_PRATICA_ESSE_EXERCICIO": frequencia,
            # o pipeline foi treinado com a duração em MINUTOS
            "EM_MEDIA_QUAL_A_DURACAO_DO_SEU_TREINO_EM_MINUTOS": duracao_treino * 60,
            "fase_ciclo": fase_atual,
            "COMO_E_SEU_FLUXO_MENSTRUAL_enc": 1,
        }

        df_input = pd.DataFrame([dados_input])

        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0.0
        df_input = df_input[feature_cols]

        # Escalona usando os nomes que o scaler aprendeu no fit (DataFrame),
        # com fallback posicional caso o scaler tenha sido treinado sem nomes.
        if hasattr(scaler, "feature_names_in_"):
            df_input_scaled = df_input.copy()
            cols_scaler = list(scaler.feature_names_in_)
            df_input_scaled[cols_scaler] = scaler.transform(df_input[cols_scaler])
            X_scaled = df_input_scaled.values
        else:
            X_scaled = scaler.transform(df_input.values)

        score_bruto = modelo.predict(X_scaled)[0]
        score_modelo = max(0.0, min(10.0, score_bruto))

        # ── Ajuste por regras usando check-in/PSS-10 de hoje (quando existirem) ──
        # O modelo de ML foi treinado só com perfil + fase do ciclo (não há dados
        # históricos reais de PSS-10/MQS no dataset). Por isso o ajuste aqui é por
        # regra, não aprendido — até acumularmos check-ins reais suficientes para
        # incluir isso no retreinamento do pipeline_final.py.
        ajuste_total = 0.0
        detalhes_ajuste = []

        checkin_hoje = buscar_checkin_hoje(st.session_state.username)
        if checkin_hoje:
            if checkin_hoje["dor"] >= 5:
                ajuste_total -= 1.0
                detalhes_ajuste.append(f"Dor relatada hoje ({checkin_hoje['dor']}/10): −1.0")
            if checkin_hoje["energia"] <= 3:
                ajuste_total -= 1.0
                detalhes_ajuste.append(f"Energia baixa hoje ({checkin_hoje['energia']}/10): −1.0")
            elif checkin_hoje["energia"] >= 8:
                ajuste_total += 0.5
                detalhes_ajuste.append(f"Energia alta hoje ({checkin_hoje['energia']}/10): +0.5")
            if checkin_hoje["humor"] in ("Baixo", "Muito baixo"):
                ajuste_total -= 0.5
                detalhes_ajuste.append(f"Humor relatado hoje ({checkin_hoje['humor']}): −0.5")

        pss10_recente = buscar_pss10_recente(st.session_state.username)
        if pss10_recente:
            if pss10_recente["score_total"] >= 27:
                ajuste_total -= 1.5
                detalhes_ajuste.append(f"PSS-10 indica estresse alto ({pss10_recente['score_total']}/40): −1.5")
            elif pss10_recente["score_total"] <= 13:
                ajuste_total += 0.5
                detalhes_ajuste.append(f"PSS-10 indica estresse baixo ({pss10_recente['score_total']}/40): +0.5")

        mqs_recente = buscar_mqs_recente(st.session_state.username)
        if mqs_recente:
            mqs_min, mqs_max = len(MQS_PERGUNTAS), 7 * len(MQS_PERGUNTAS)
            faixa_mqs = mqs_max - mqs_min
            if mqs_recente["score_total"] <= mqs_min + faixa_mqs * 0.33:
                ajuste_total -= 1.0
                detalhes_ajuste.append(f"MQS indica bem-estar baixo ({mqs_recente['score_total']}/{mqs_max}): −1.0")
            elif mqs_recente["score_total"] >= mqs_min + faixa_mqs * 0.66:
                ajuste_total += 0.5
                detalhes_ajuste.append(f"MQS indica bem-estar bom ({mqs_recente['score_total']}/{mqs_max}): +0.5")

        score_previsto = max(0.0, min(10.0, score_modelo + ajuste_total))

        salvar_registro(st.session_state.username, nomes_fases[fase_atual], score_previsto)

        # ── Painel visual do resultado da IA (elemento de assinatura) ──
        fase_nome = nomes_fases[fase_atual]
        estilo = PHASE_STYLE[fase_nome]

        col_gauge, col_fase = st.columns([1, 1.4])
        with col_gauge:
            with st.container(border=True):
                st.markdown(
                    render_gauge(score_previsto, estilo["cor"], "Prontidão de Hoje"),
                    unsafe_allow_html=True,
                )
        with col_fase:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-family:'Manrope',sans-serif; font-weight:700; color:{estilo['cor']} !important;
                            text-transform:uppercase; font-size:0.72rem; letter-spacing:0.08em;">
                    {estilo['emoji']} Fase Detectada
                </div>
                <div style="font-family:'Fraunces',serif; font-size:1.8rem; color:#111111 !important; margin-top:0.1rem;">
                    {fase_nome}
                </div>
                <div style="font-size:0.85rem; color:#5B5B5B !important; margin-top:0.5rem;">
                    Score base do modelo: {score_modelo:.1f} / 10.0
                    {f" · Ajuste do dia: {ajuste_total:+.1f}" if detalhes_ajuste else ""}
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        st.markdown(
            render_timeline_ciclo(dia_ciclo, duracao_media, duracao_mens),
            unsafe_allow_html=True,
        )
        st.write("")

        if detalhes_ajuste:
            col_a, col_b = st.columns(2)
            col_a.metric(label="Score Base (modelo)", value=f"{score_modelo:.1f} / 10.0")
            col_b.metric(
                label="Score Ajustado (com check-in/PSS-10 de hoje)",
                value=f"{score_previsto:.1f} / 10.0",
                delta=f"{ajuste_total:+.1f}",
            )
            with st.expander("Ver detalhes do ajuste"):
                for linha in detalhes_ajuste:
                    st.markdown(f"- {linha}")
        else:
            st.caption(
                "💡 Faça o check-in diário e/ou o PSS-10 para um score ajustado ao "
                "seu estado de hoje, não só ao perfil e fase do ciclo."
            )

        st.markdown("---")
        st.markdown("### 📋 Prescrição do Treino Baseada na Fisiologia")

        if fase_atual == 0:  # Menstrual
            if score_previsto < 5:
                st.error(
                    "🩸 **Recomendação:** Seu corpo está demandando energia para a descamação "
                    "uterina. Foco total em recuperação ativa, caminhada leve ou yoga. "
                    "Evite sobrecarga articular."
                )
            else:
                st.warning(
                    "🩸 **Recomendação:** Você está se sentindo disposta, mas mantenha o treino "
                    "de leve a moderado. Foque em volume baixo e movimentos que não causem "
                    "desconforto abdominal."
                )

        elif fase_atual == 1:  # Folicular
            if score_previsto < 5:
                st.warning(
                    "📈 **Recomendação:** O estrogênio está subindo, o que favorece ganho de "
                    "força, mas seu nível de fadiga está alto hoje. Faça um treino moderado, "
                    "focando na execução correta."
                )
            else:
                st.info(
                    "🔥 **Recomendação:** Janela de Ouro! Estrogênio em alta e ótima prontidão. "
                    "Excelente fase para hipertrofia, treinos de força intensos (HIIT) e buscar "
                    "progressão de carga."
                )

        elif fase_atual == 2:  # Ovulatória
            if score_previsto < 5:
                st.warning(
                    "🥚 **Recomendação:** Fase de pico hormonal, porém com score de prontidão "
                    "baixo. Treine com moderação. Cuidado redobrado com alongamentos excessivos "
                    "(maior frouxidão ligamentar)."
                )
            else:
                st.info(
                    "⚡ **Recomendação:** Força no pico! Ótimo para bater recordes pessoais (PRs). "
                    "**Atenção clínica:** O pico de estrogênio/relaxina pode aumentar a "
                    "instabilidade articular. Mantenha controle motor estrito!"
                )

        elif fase_atual == 3:  # Lútea
            if score_previsto < 5:
                st.error(
                    "🌙 **Recomendação:** Fase de alta progesterona — ela aumenta a temperatura "
                    "corporal, reduz a eficiência cardiovascular e deixa seu corpo mais propenso "
                    "à fadiga. Priorize recuperação, treinos leves e hidratação adequada. Evite "
                    "cargas altas e exercícios extenuantes."
                )
            else:
                st.warning(
                    "⚖️ **Recomendação:** Treino de manutenção. Mesmo com boa disposição, evite "
                    "treinos exaustivos ao calor. Prefira cardio em *steady-state* (ritmo "
                    "constante) e exercícios com cargas moderadas."
                )

with tab_historico:
    st.markdown("### 📈 Histórico")
    st.caption("Evolução dos seus registros ao longo do tempo.")

    hist_reg = buscar_historico_registros(st.session_state.username)
    hist_check = buscar_historico_checkins(st.session_state.username)
    hist_pss10 = buscar_historico_pss10(st.session_state.username)
    hist_mqs = buscar_historico_mqs(st.session_state.username)

    with st.container(border=True):
        st.markdown("#### 🔮 Score de Prontidão ao longo do tempo")
        if not hist_reg.empty:
            st.line_chart(hist_reg.set_index("data")["score"], color="#6B3F63")
            st.dataframe(hist_reg.sort_values("data", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("Ainda não há cálculos de prontidão registrados.")

    st.write("")
    with st.container(border=True):
        st.markdown("#### 📝 Check-in Diário — Energia e Dor")
        if not hist_check.empty:
            st.line_chart(hist_check.set_index("data")[["energia", "dor"]])
            st.dataframe(hist_check.sort_values("data", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("Ainda não há check-ins diários registrados.")

    st.write("")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        with st.container(border=True):
            st.markdown("#### 🧠 PSS-10 ao longo do tempo")
            if not hist_pss10.empty:
                st.bar_chart(hist_pss10.set_index("data")["score_total"], color="#C1435A")
            else:
                st.caption("Ainda não há respostas de PSS-10.")

    with col_h2:
        with st.container(border=True):
            st.markdown("#### 📊 MQS ao longo do tempo")
            if not hist_mqs.empty:
                st.bar_chart(hist_mqs.set_index("data")["score_total"], color="#4C9A6B")
            else:
                st.caption("Ainda não há respostas de MQS.")

