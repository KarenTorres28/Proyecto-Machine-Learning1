"""
Aplicación Web Interactiva en Streamlit para el Despliegue de F-KNN
Imputación de Series de Tiempo Meteorológicas — Aranzazu, Caldas
Tema visual: cielo, estrellas, nubes, sol y plantas. Gráficas Plotly con paleta Viridis.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False

# Importante: Importar la clase personalizada antes de cargar el .joblib
from src.fknn import FocalizedKNNImputer
from src.preprocessing import preparar_dataframe_entrada, limpiar_noches

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y RUTAS
# ==========================================
st.set_page_config(
    page_title="Imputador Meteorológico F-KNN",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# Paleta Viridis
VIRIDIS_DARK_PURPLE = "#440154"
VIRIDIS_BLUE = "#3b528b"
VIRIDIS_TEAL = "#21918c"
VIRIDIS_GREEN = "#5ec962"
VIRIDIS_YELLOW = "#fde725"

# Un icono por sensor (solo decorativo)
ICONOS = {
    "Indoor Temperature(°C)": "🏠",
    "Indoor Humidity(%)": "🏡",
    "Outdoor Temperature(°C)": "🌡️",
    "Outdoor Humidity(%)": "💧",
    "Absolute Pressure(hpa)": "🧭",
    "Wind Speed(km/h)": "🍃",
    "Gust(km/h)": "🌬️",
    "DewPoint(°C)": "🌫️",
    "WindChill(°C)": "❄️",
    "Hour Rainfall(mm)": "🌧️",
    "Light(lux)": "☀️",
}

# ==========================================
# ESTILOS (CSS)
# ==========================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Poppins:wght@600;700;800&display=swap');

/* ---------- Fondo: cielo que va de noche a amanecer ---------- */
.stApp{
    background: linear-gradient(180deg,#0b1026 0%,#141c47 30%,#27407a 62%,#2f6f8f 85%,#21918c 100%);
    background-attachment: fixed;
}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}

/* ---------- Estrellas (dos capas con titileo) ---------- */
.stApp::before, .stApp::after{
    content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
    -webkit-mask-image:linear-gradient(#000 25%,transparent 80%);
    mask-image:linear-gradient(#000 25%,transparent 80%);
}
.stApp::before{
    background-image:
        radial-gradient(1.5px 1.5px at 40px 60px,#fff,transparent),
        radial-gradient(1px 1px at 130px 120px,#fde725,transparent),
        radial-gradient(2px 2px at 210px 40px,#fff,transparent),
        radial-gradient(1px 1px at 90px 200px,#fff,transparent),
        radial-gradient(1.5px 1.5px at 260px 170px,#cfe3ff,transparent),
        radial-gradient(1px 1px at 180px 260px,#fff,transparent);
    background-size:300px 300px;
    animation:titilar 5s ease-in-out infinite alternate;
}
.stApp::after{
    background-image:
        radial-gradient(1px 1px at 70px 90px,#fff,transparent),
        radial-gradient(2px 2px at 300px 150px,#fde725,transparent),
        radial-gradient(1.5px 1.5px at 150px 330px,#fff,transparent),
        radial-gradient(1px 1px at 380px 40px,#cfe3ff,transparent),
        radial-gradient(1px 1px at 240px 230px,#fff,transparent);
    background-size:430px 430px;
    animation:titilar 7s ease-in-out -2s infinite alternate;
}
@keyframes titilar{from{opacity:.25}to{opacity:1}}

/* ---------- Nubes que se desplazan ---------- */
.nube{
    position:fixed;width:200px;height:56px;border-radius:100px;
    background:rgba(255,255,255,.10);filter:blur(1.5px);
    pointer-events:none;z-index:1;scale:var(--s,1);
    animation:derivar linear infinite;
}
.nube::before,.nube::after{content:"";position:absolute;background:inherit;border-radius:50%;}
.nube::before{width:90px;height:90px;top:-44px;left:28px;}
.nube::after{width:120px;height:120px;top:-62px;right:26px;}
@keyframes derivar{from{transform:translateX(-380px)}to{transform:translateX(112vw)}}

/* ---------- Contenido principal ---------- */
.block-container{position:relative;z-index:2;max-width:1200px;padding-top:2rem;padding-bottom:2rem;}
.stMarkdown, .stMarkdown p, label, .stSelectbox, .stButton button, .stDownloadButton button{
    font-family:'Nunito',sans-serif;
}

/* ---------- Hero ---------- */
.hero{
    display:flex;align-items:center;gap:2.2rem;flex-wrap:wrap;
    padding:2rem 2.4rem;margin-bottom:1.8rem;border-radius:26px;
    background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(255,255,255,.04));
    border:1px solid rgba(255,255,255,.18);
    box-shadow:0 18px 50px rgba(0,0,0,.35);
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
}
.hero-sol{position:relative;width:120px;height:120px;flex:0 0 120px;margin:20px;}
.rayos{
    position:absolute;inset:-34px;border-radius:50%;
    background:repeating-conic-gradient(from 0deg,rgba(253,231,37,.65) 0deg 5deg,transparent 5deg 22deg);
    -webkit-mask-image:radial-gradient(circle,#000 38%,transparent 70%);
    mask-image:radial-gradient(circle,#000 38%,transparent 70%);
    animation:girar 50s linear infinite;
}
.sol{
    position:absolute;inset:0;border-radius:50%;
    background:radial-gradient(circle at 35% 35%,#fffbc2,#fde725 55%,#f5a300);
    box-shadow:0 0 40px 10px rgba(253,231,37,.45),0 0 120px 40px rgba(253,231,37,.18);
    animation:pulso 5s ease-in-out infinite;
}
@keyframes girar{to{transform:rotate(360deg)}}
@keyframes pulso{50%{transform:scale(1.06)}}
.hero-texto{flex:1;min-width:280px;}
.insignia{
    display:inline-block;padding:.3rem .9rem;border-radius:999px;font-size:.82rem;font-weight:700;
    background:rgba(253,231,37,.15);color:#fde725;border:1px solid rgba(253,231,37,.4);
}
.hero h1{
    font-family:'Poppins',sans-serif;font-weight:800;font-size:2.3rem;line-height:1.15;
    margin:.7rem 0 .5rem 0;padding:0;
    background:linear-gradient(90deg,#ffffff,#fde725 60%,#5ec962);
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}
.hero p{color:#dbe7ff;font-size:1.02rem;margin:0 0 .9rem 0;}
.chips span{
    display:inline-block;margin:0 .45rem .45rem 0;padding:.28rem .8rem;border-radius:999px;
    font-size:.82rem;font-weight:700;color:#eaf2ff;
    background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
}

/* ---------- Encabezados de sección ---------- */
.sec{display:flex;align-items:center;gap:.9rem;margin:2.2rem 0 1rem 0;}
.sec-ico{
    width:52px;height:52px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;
    border-radius:16px;background:linear-gradient(135deg,rgba(253,231,37,.25),rgba(94,201,98,.2));
    border:1px solid rgba(255,255,255,.25);
}
.sec-t{font-family:'Poppins',sans-serif;font-weight:700;font-size:1.35rem;color:#fff;}
.sec-s{font-size:.9rem;color:#b9c9ee;}

/* ---------- Tarjetas KPI ---------- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;margin-bottom:1.2rem;}
.kpi{
    padding:1.1rem 1.2rem;border-radius:20px;text-align:center;
    background:linear-gradient(160deg,rgba(255,255,255,.14),rgba(255,255,255,.05));
    border:1px solid rgba(255,255,255,.2);box-shadow:0 10px 30px rgba(0,0,0,.25);
    transition:transform .25s ease, box-shadow .25s ease;
}
.kpi:hover{transform:translateY(-4px);box-shadow:0 16px 36px rgba(0,0,0,.35);}
.kpi-i{font-size:1.7rem;}
.kpi-v{font-family:'Poppins',sans-serif;font-weight:700;font-size:1.7rem;color:#fde725;margin:.2rem 0;}
.kpi-v.sm{font-size:1.05rem;line-height:2.1rem;}
.kpi-l{font-size:.82rem;font-weight:700;color:#b9c9ee;text-transform:uppercase;letter-spacing:.06em;}

/* ---------- Pastilla de datos activos ---------- */
.pastilla{
    display:inline-block;padding:.55rem 1.1rem;border-radius:14px;margin:.2rem 0 .6rem 0;
    background:rgba(94,201,98,.15);border:1px solid rgba(94,201,98,.45);color:#d9ffd9;font-weight:700;
}

/* ---------- Widgets de Streamlit ---------- */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"]{
    background:rgba(255,255,255,.07);border:2px dashed rgba(253,231,37,.55);border-radius:20px;
}
[data-testid="stAlert"]{
    border-radius:16px;backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.15);
}
[data-testid="stExpander"]{
    background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);border-radius:16px;
}
[data-testid="stPlotlyChart"]{
    background:linear-gradient(160deg,rgba(255,255,255,.10),rgba(255,255,255,.03));
    border:1px solid rgba(255,255,255,.18);border-radius:22px;padding:.6rem;
    box-shadow:0 12px 34px rgba(0,0,0,.3);
}
.stButton button, .stDownloadButton button{
    border-radius:14px;font-weight:800;padding:.6rem 1rem;
    background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.3);
    transition:transform .2s ease, box-shadow .2s ease;
}
.stButton button:hover, .stDownloadButton button:hover{
    transform:translateY(-2px);border-color:#fde725;color:#fde725;
}
button[kind="primary"], [data-testid="stBaseButton-primary"], .stDownloadButton button{
    background:linear-gradient(90deg,#fde725,#5ec962);color:#0b1026 !important;border:none;
    box-shadow:0 8px 22px rgba(253,231,37,.3);
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, .stDownloadButton button:hover{
    color:#0b1026 !important;box-shadow:0 12px 28px rgba(253,231,37,.5);
}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1026 0%,#16204a 55%,#134e4a 100%);
    border-right:1px solid rgba(255,255,255,.1);
}
.marca{display:flex;align-items:center;gap:.8rem;margin-bottom:1rem;}
.luna{
    width:48px;height:48px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;
    border-radius:50%;background:rgba(255,255,255,.1);box-shadow:0 0 22px rgba(207,227,255,.45);
}
.marca-t{font-family:'Poppins',sans-serif;font-weight:700;font-size:1.15rem;color:#fff;}
.marca-s{font-size:.82rem;color:#b9c9ee;}
.lado{
    padding:.85rem 1rem;margin-bottom:.8rem;border-radius:16px;font-size:.88rem;color:#e3ecff;
    background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
}
.lado b{color:#fde725;}
.lado small{display:block;margin-top:.5rem;color:#9fb3df;}
.lado code{color:#5ec962;background:rgba(0,0,0,.25);padding:.1rem .4rem;border-radius:6px;}
.mini{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-bottom:.8rem;}
.mini div{
    padding:.7rem;border-radius:14px;text-align:center;font-size:.8rem;color:#b9c9ee;
    background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
}
.mini strong{display:block;color:#fde725;font-size:1rem;margin-top:.25rem;}
[data-testid="stSidebar"] [data-testid="stImage"] img{
    background:#fff;border-radius:12px;padding:8px;
}

/* ---------- Plantas al pie ---------- */
.paisaje{width:100%;height:120px;display:block;margin-top:3rem;}
.sway{transform-box:fill-box;transform-origin:50% 100%;animation:mecer 4s ease-in-out infinite alternate;}
@keyframes mecer{from{transform:rotate(-3.5deg)}to{transform:rotate(3.5deg)}}
.pie{text-align:center;color:#d7f5e6;font-size:.85rem;padding:.6rem 0 .2rem 0;font-weight:600;}
"""


def limpiar_html(texto):
    """Quita sangrías y líneas vacías para que Markdown no lo interprete como bloque de código."""
    return "\n".join(l.strip() for l in texto.splitlines() if l.strip())


def inyectar_estilos():
    st.markdown(f"<style>{limpiar_html(CSS)}</style>", unsafe_allow_html=True)


def nubes_de_fondo():
    """Tres nubes semitransparentes que cruzan la pantalla a distintas alturas y velocidades."""
    nubes = [
        ("top:12%;--s:1.2;animation-duration:70s;animation-delay:-10s;"),
        ("top:30%;--s:0.8;animation-duration:95s;animation-delay:-55s;"),
        ("top:52%;--s:1.5;animation-duration:120s;animation-delay:-30s;"),
    ]
    html = "".join(f'<div class="nube" style="{estilo}"></div>' for estilo in nubes)
    st.markdown(html, unsafe_allow_html=True)


def paisaje_plantas():
    """SVG con colinas y plantas que se mecen suavemente."""
    plantas = []
    for k, x in enumerate(range(30, 1200, 70)):
        h = 38 + (k * 17) % 34
        y0, yc, ym = 120, 120 - h, 120 - h * 0.55
        retraso = -(k % 5) * 0.8
        plantas.append(
            f'<g class="sway" style="animation-delay:{retraso:.1f}s">'
            f'<path d="M{x} {y0} Q{x + 7} {y0 - h / 2:.0f} {x} {yc:.0f}" stroke="#3ea66b" '
            f'stroke-width="3" fill="none" stroke-linecap="round"/>'
            f'<ellipse cx="{x - 10}" cy="{ym:.0f}" rx="10" ry="4.5" fill="#5ec962" '
            f'transform="rotate(-35 {x - 10} {ym:.0f})"/>'
            f'<ellipse cx="{x + 10}" cy="{ym - 8:.0f}" rx="10" ry="4.5" fill="#5ec962" '
            f'transform="rotate(35 {x + 10} {ym - 8:.0f})"/>'
            f'<ellipse cx="{x}" cy="{yc - 6:.0f}" rx="4.5" ry="10" fill="#8fe388"/>'
            f'</g>'
        )
    svg = (
        '<svg class="paisaje" viewBox="0 0 1200 120" preserveAspectRatio="xMidYMax slice" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path d="M0 120 L0 80 Q150 50 320 76 T640 70 T960 64 T1200 74 L1200 120Z" fill="#12474a"/>'
        + "".join(plantas) +
        '<path d="M0 120 L0 106 Q200 96 420 108 T820 104 T1200 108 L1200 120Z" fill="#0c3335"/>'
        '</svg>'
    )
    st.markdown(svg, unsafe_allow_html=True)


def seccion(icono, titulo, subtitulo=""):
    st.markdown(
        f'<div class="sec"><div class="sec-ico">{icono}</div><div>'
        f'<div class="sec-t">{titulo}</div><div class="sec-s">{subtitulo}</div></div></div>',
        unsafe_allow_html=True
    )


def tarjetas_kpi(items):
    """items: lista de (icono, etiqueta, valor)."""
    cuerpo = "".join(
        f'<div class="kpi"><div class="kpi-i">{ico}</div>'
        f'<div class="kpi-v{" sm" if len(str(val)) > 14 else ""}">{val}</div>'
        f'<div class="kpi-l">{lab}</div></div>'
        for ico, lab, val in items
    )
    st.markdown(f'<div class="kpis">{cuerpo}</div>', unsafe_allow_html=True)


def estilo_plotly(fig, altura):
    """Aplica el tema nocturno transparente a cualquier figura de Plotly."""
    fig.update_layout(
        template="plotly_dark",
        height=altura,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito, sans-serif", color="#eaf2ff"),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


inyectar_estilos()
nubes_de_fondo()

# ==========================================
# GESTIÓN DE ESTADO (Session State)
# ==========================================
if "df_crudo" not in st.session_state:
    st.session_state["df_crudo"] = None
if "origen_datos" not in st.session_state:
    st.session_state["origen_datos"] = None
if "df_imputado" not in st.session_state:
    st.session_state["df_imputado"] = None


@st.cache_resource
def cargar_artefactos():
    """Carga el pipeline entrenado y sus metadatos desde disco."""
    pipeline_path = MODELS_DIR / "fknn_pipeline.joblib"
    meta_path = MODELS_DIR / "metadata_despliegue.json"

    if not pipeline_path.exists() or not meta_path.exists():
        st.error("No se encontraron los artefactos del modelo en 'models/'. Ejecuta primero 'python export_model.py'.")
        st.stop()

    pipeline = joblib.load(pipeline_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata


pipeline, metadata = cargar_artefactos()
variables_modelo = metadata["variables"]

# ==========================================
# BARRA LATERAL (Sidebar)
# ==========================================
with st.sidebar:
    st.markdown(
        '<div class="marca"><div class="luna">🌙</div><div>'
        '<div class="marca-t">Panel de Control F-KNN</div>'
        '<div class="marca-s">Estación Meteorológica Aranzazu (Caldas)</div></div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="lado"><b>👥 Autores</b><br>Gabriel Castro Benítez<br>Karen Daniela Torres Henao'
        '<small>Machine Learning 1 · Pregrado en Ciencia de Datos · Universidad Externado de Colombia</small></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="lado"><b>⚙️ Configuración del modelo</b><br>'
        f'<small>Algoritmo</small><code>{metadata["modelo"]}</code>'
        f'<small>Hiperparámetros</small><code>{metadata["configuracion"]}</code>'
        f'<small>Frecuencia esperada</small><code>{metadata["frecuencia"]}</code></div>',
        unsafe_allow_html=True
    )

    m_dis = metadata["metricas_prueba_2019"]["disjoint"]
    m_ov = metadata["metricas_prueba_2019"]["overlap"]
    st.markdown('<div class="lado"><b>📊 Métricas en prueba 2019</b></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mini">'
        f'<div>Disjoint<strong>R² {m_dis["R2_medio"]}</strong>MSE {m_dis["MSE_medio"]}</div>'
        f'<div>Overlap<strong>R² {m_ov["R2_medio"]}</strong>MSE {m_ov["MSE_medio"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="lado"><b>💡 Atajos de datos</b></div>', unsafe_allow_html=True)
    if st.button("📂 Cargar datos de ejemplo (Semana 2019)", use_container_width=True):
        ejemplo_path = DATA_DIR / "ejemplo_para_probar.csv"
        if ejemplo_path.exists():
            st.session_state["df_crudo"] = pd.read_csv(ejemplo_path, index_col=0, parse_dates=True)
            st.session_state["origen_datos"] = "Dataset de ejemplo (Semana 2019 con 20 % de faltantes)"
            st.session_state["df_imputado"] = None
            st.rerun()
        else:
            st.error("No se encontró el archivo 'data/ejemplo_para_probar.csv'.")

    if st.session_state["df_crudo"] is not None:
        if st.button("🔄 Limpiar / Cargar otro archivo", use_container_width=True):
            st.session_state["df_crudo"] = None
            st.session_state["origen_datos"] = None
            st.session_state["df_imputado"] = None
            st.rerun()

    # Declaración sobre el uso de Inteligencia Artificial
    with st.expander("🤖 Uso declarado de IA"):
        st.markdown("""
Durante el diseño, estructuración e implementación de este módulo de despliegue, **utilizamos** la siguiente herramienta de apoyo tecnológico bajo nuestra supervisión directa:

* **Antigravity (Google DeepMind, 2026):** Empleado como asistente de ingeniería de software para la modularización del estimador `FocalizedKNNImputer`, la estructuración del framework de despliegue en Streamlit, el control de persistencia de estado (`session_state`) y la construcción de las visualizaciones interactivas en Plotly con paleta Viridis.

La validación experimental, la verificación del código, la interpretación analítica de los resultados y la autoría íntegra del contenido técnico recaen exclusivamente sobre nosotros como autores del proyecto.
        """)

    st.markdown("---")

    # Logos institucionales
    logo_ext = ASSETS_DIR / "logo_externado.png"
    logo_cd = ASSETS_DIR / "logo_ciencia_de_datos.png"

    if logo_ext.exists() and logo_cd.exists():
        col_l1, col_l2 = st.columns([1.3, 1])
        with col_l1:
            st.image(str(logo_ext), use_container_width=True)
        with col_l2:
            st.image(str(logo_cd), use_container_width=True)
    elif logo_ext.exists():
        st.image(str(logo_ext), use_container_width=True)
    elif logo_cd.exists():
        st.image(str(logo_cd), width=160)

# ==========================================
# CUERPO PRINCIPAL: HERO
# ==========================================
st.markdown(limpiar_html(f"""
<div class="hero">
<div class="hero-sol"><div class="rayos"></div><div class="sol"></div></div>
<div class="hero-texto">
<span class="insignia">☁️ Estación meteorológica · Aranzazu, Caldas</span>
<h1>Sistema de Imputación de Sensores Meteorológicos</h1>
<p>Reconstruimos los valores que dejaron de registrar los sensores usando el modelo ganador
<b>F-KNN (Focalized K-Nearest Neighbors)</b>, que combina variables correlacionadas
(correlación de Spearman) con rezagos temporales de la propia serie.</p>
<div class="chips">
<span>⭐ F-KNN</span><span>🌿 Correlación de Spearman</span>
<span>🕐 Rezagos temporales</span><span>🌧️ {metadata['n_variables']} variables</span>
</div>
</div>
</div>
"""), unsafe_allow_html=True)

if not PLOTLY_INSTALLED:
    st.warning("⚠️ La librería `plotly` no está instalada. Para visualizar las gráficas interactivas, ejecuta en tu terminal: `pip install plotly`.")

# ==========================================
# SECCIÓN 1: CARGA DE ARCHIVO
# ==========================================
seccion("☁️", "1 · Cargar archivo de datos", "Sube el CSV con tus series de tiempo o usa el ejemplo del panel lateral")

archivo_subido = st.file_uploader(
    "Sube un archivo CSV con las series de tiempo (frecuencia recomendada de 30 minutos):",
    type=["csv"]
)

if archivo_subido is not None:
    try:
        nuevo_df = pd.read_csv(archivo_subido)
        if st.session_state["origen_datos"] != archivo_subido.name:
            st.session_state["df_crudo"] = nuevo_df
            st.session_state["origen_datos"] = archivo_subido.name
            st.session_state["df_imputado"] = None
            st.rerun()
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")

# ==========================================
# SECCIÓN 2: DIAGNÓSTICO
# ==========================================
df_crudo = st.session_state["df_crudo"]

if df_crudo is not None:
    st.markdown(
        f'<div class="pastilla">📁 Datos activos: {st.session_state["origen_datos"]}</div>',
        unsafe_allow_html=True
    )

    try:
        df_preparado = preparar_dataframe_entrada(df_crudo, variables_modelo)
    except Exception as err:
        st.error(f"Error de validación en los datos: {err}")
        st.stop()

    n_filas = len(df_preparado)
    total_celdas = df_preparado.size
    total_faltantes = int(df_preparado.isna().sum().sum())
    pct_faltantes_global = (total_faltantes / total_celdas) * 100

    seccion("🌧️", "2 · Diagnóstico de valores faltantes", "Qué tan completos llegaron los datos de cada sensor")

    rango_txt = (
        f"{df_preparado.index.min().strftime('%d/%m/%Y')} → "
        f"{df_preparado.index.max().strftime('%d/%m/%Y')}"
    )
    tarjetas_kpi([
        ("🌤️", "Total observaciones", f"{n_filas:,}"),
        ("📅", "Rango temporal", rango_txt),
        ("🕳️", "Celdas faltantes", f"{total_faltantes:,}"),
        ("📉", "% faltantes global", f"{pct_faltantes_global:.2f} %"),
    ])

    faltantes_col = df_preparado.isna().sum()
    pct_col = (faltantes_col / n_filas) * 100
    df_diag = pd.DataFrame({
        "Variable": variables_modelo,
        "Faltantes": [int(faltantes_col[v]) for v in variables_modelo],
        "% Faltante": [float(pct_col[v].round(2)) for v in variables_modelo],
        "Mín Histórico 2018": [metadata["rangos_historicos_2018"][v]["min"] for v in variables_modelo],
        "Máx Histórico 2018": [metadata["rangos_historicos_2018"][v]["max"] for v in variables_modelo]
    })

    if PLOTLY_INSTALLED and total_faltantes > 0:
        fig_diag = px.bar(
            df_diag.sort_values("% Faltante", ascending=True),
            x="% Faltante",
            y="Variable",
            orientation="h",
            color="% Faltante",
            color_continuous_scale="Viridis",
            title="Porcentaje de valores faltantes por sensor meteorológico",
            labels={"% Faltante": "% de faltantes", "Variable": "Sensor"},
            text="% Faltante"
        )
        fig_diag.update_traces(
            texttemplate='%{text:.1f}%', textposition='outside',
            marker_line_width=0, cliponaxis=False
        )
        estilo_plotly(fig_diag, 420)
        fig_diag.update_layout(coloraxis_showscale=False)
        fig_diag.update_xaxes(range=[0, max(df_diag["% Faltante"].max() * 1.25, 1)])
        st.plotly_chart(fig_diag, use_container_width=True)

    with st.expander("🔍 Ver tabla detallada de diagnóstico por sensor"):
        st.dataframe(df_diag.set_index("Variable").T, use_container_width=True)

    # ==========================================
    # SECCIÓN 3: RECONSTRUCCIÓN CON F-KNN
    # ==========================================
    seccion("⭐", "3 · Reconstrucción con F-KNN", "Rellena los huecos con el pipeline entrenado con los datos de 2018")

    if total_faltantes == 0:
        st.success("🎉 El dataset no presenta valores faltantes. No se requiere imputación.")
        st.session_state["df_imputado"] = df_preparado.copy()
    else:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            ejecutar = st.button("🚀 Ejecutar Imputación con Pipeline F-KNN", type="primary", use_container_width=True)

        if ejecutar:
            with st.spinner("Reconstruyendo series de tiempo con F-KNN..."):
                try:
                    X_array = df_preparado.to_numpy(dtype=float)
                    X_imputado = pipeline.transform(X_array)

                    df_imputado = pd.DataFrame(
                        X_imputado,
                        index=df_preparado.index,
                        columns=variables_modelo
                    )
                    st.session_state["df_imputado"] = df_imputado
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error durante la imputación: {ex}")

    # ==========================================
    # SECCIÓN 4 Y 5: VISUALIZACIÓN Y DESCARGA
    # ==========================================
    if st.session_state["df_imputado"] is not None:
        df_res = st.session_state["df_imputado"]
        st.success(f"✅ ¡Imputación completada con éxito! Se reconstruyeron {total_faltantes:,} celdas.")

        seccion("🌈", "4 · Inspección visual interactiva", "Los puntos amarillos son los valores estimados por F-KNN")

        variable_seleccionada = st.selectbox(
            "Selecciona la variable que deseas inspeccionar:",
            variables_modelo,
            format_func=lambda v: f"{ICONOS.get(v, '📈')}  {v}"
        )

        serie_original = df_preparado[variable_seleccionada]
        serie_imputada = df_res[variable_seleccionada]
        mascara_faltantes = serie_original.isna()

        if PLOTLY_INSTALLED:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=serie_imputada.index,
                y=serie_imputada.values,
                mode="lines",
                name="Serie completa (F-KNN)",
                line=dict(color=VIRIDIS_GREEN, width=2),
                hovertemplate="<b>Fecha:</b> %{x|%d/%m/%Y %H:%M}<br><b>Valor:</b> %{y:.2f}<extra></extra>"
            ))

            if mascara_faltantes.any():
                fig.add_trace(go.Scatter(
                    x=serie_imputada.index[mascara_faltantes],
                    y=serie_imputada.values[mascara_faltantes],
                    mode="markers",
                    name=f"Valores imputados ({int(mascara_faltantes.sum())} celdas)",
                    marker=dict(
                        color=VIRIDIS_YELLOW,
                        size=8,
                        symbol="circle",
                        line=dict(color=VIRIDIS_DARK_PURPLE, width=1.5)
                    ),
                    hovertemplate="<b>⚠️ Imputado F-KNN</b><br><b>Fecha:</b> %{x|%d/%m/%Y %H:%M}<br><b>Valor estimado:</b> %{y:.2f}<extra></extra>"
                ))

            estilo_plotly(fig, 520)
            fig.update_layout(
                title=dict(
                    text=f"{ICONOS.get(variable_seleccionada, '📈')} Reconstrucción temporal de <b>{variable_seleccionada}</b>",
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Fecha y hora",
                    rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(255,255,255,0.05)"),
                    type="date"
                ),
                yaxis=dict(title=f"Valor ({variable_seleccionada})"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se pudo renderizar la gráfica interactiva porque `plotly` no está instalado en el entorno.")

        # Descarga
        seccion("🌱", "5 · Descargar datos limpios e imputados", "Archivo CSV con la serie completa reconstruida")
        csv_bytes = df_res.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Descargar CSV Imputado",
            data=csv_bytes,
            file_name="dataset_meteorologico_imputado_fknn.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("👆 Por favor sube un archivo CSV o haz clic en **'Cargar datos de ejemplo'** en la barra lateral para comenzar.")

# ==========================================
# PIE DE PÁGINA
# ==========================================
paisaje_plantas()
st.markdown(
    '<div class="pie">🌿 Universidad Externado de Colombia · Ciencia de Datos · Machine Learning 1 · 2026</div>',
    unsafe_allow_html=True
)