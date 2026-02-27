import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 | Análisis Inicial de la Pandemia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Estilos globales ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1200px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #cbd5e0 !important; font-size: 0.9rem; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    /* ── Banners de sección ── */
    .banner {
        background: linear-gradient(135deg, #c0392b 0%, #922b21 100%);
        color: white !important;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(192,57,43,0.35);
    }
    .banner h1 { color: white !important; margin: 0 0 0.4rem 0; font-size: 1.75rem; font-weight: 700; }
    .banner p  { color: rgba(255,255,255,0.88) !important; margin: 0; font-size: 0.95rem; line-height: 1.55; }
    .banner-blue   { background: linear-gradient(135deg, #1a6bb5 0%, #0e4f8b 100%); box-shadow: 0 4px 20px rgba(26,107,181,0.35); }
    .banner-green  { background: linear-gradient(135deg, #1e8449 0%, #145a32 100%); box-shadow: 0 4px 20px rgba(30,132,73,0.35); }
    .banner-purple { background: linear-gradient(135deg, #6c3483 0%, #4a235a 100%); box-shadow: 0 4px 20px rgba(108,52,131,0.35); }
    .banner-dark   { background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%); box-shadow: 0 4px 20px rgba(44,62,80,0.35); }

    /* ── Tarjeta métrica ── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.09);
        border-top: 4px solid #c0392b;
        text-align: center;
        height: 100%;
    }
    .metric-card.blue  { border-top-color: #2980b9; }
    .metric-card.green { border-top-color: #27ae60; }
    .metric-card.gray  { border-top-color: #7f8c8d; }
    .metric-value { font-size: 2.1rem; font-weight: 700; color: #1a1a2e; margin: 0.3rem 0 0.2rem; }
    .metric-label { font-size: 0.82rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

    /* ── Tarjeta de contenido ── */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.07);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.2rem;
    }

    /* ── Caja de interpretación ── */
    .interpretacion {
        background: #fdfaf9;
        border-left: 5px solid #c0392b;
        padding: 1.1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1.4rem;
        font-size: 0.93rem;
        line-height: 1.65;
        color: #374151;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .interpretacion b { color: #c0392b; }
    .interpretacion.blue   { border-left-color: #2980b9; background: #f0f7ff; }
    .interpretacion.blue b { color: #2980b9; }
    .interpretacion.green  { border-left-color: #27ae60; background: #f0fff6; }
    .interpretacion.green b { color: #27ae60; }
    .interpretacion.purple { border-left-color: #8e44ad; background: #faf0ff; }
    .interpretacion.purple b { color: #8e44ad; }

    /* ── Tarjeta de pregunta ── */
    .question-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border: 1px solid #f0f0f0;
        border-left: 4px solid #c0392b;
        height: 100%;
    }
    .question-card.q2 { border-left-color: #2980b9; }
    .question-card.q3 { border-left-color: #27ae60; }
    .question-card.q4 { border-left-color: #8e44ad; }
    .question-card h4 { margin: 0 0 0.5rem; font-size: 0.98rem; font-weight: 600; color: #1a1a2e; }
    .question-card p  { margin: 0; font-size: 0.87rem; color: #6b7280; line-height: 1.5; }

    /* ── Nota/advertencia ── */
    .nota {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.88rem;
        color: #78350f;
        margin-top: 0.8rem;
    }

    /* ── Encabezado de filtros ── */
    .filtros-header {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f3f4f6;
    }

    /* ── Ajustes Streamlit nativos ── */
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    }
    .stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers visuales ─────────────────────────────────────────────────────────
def banner(titulo, subtitulo="", color=""):
    cls = f"banner {color}"
    st.markdown(f"""
    <div class="{cls}">
        <h1>{titulo}</h1>
        {"<p>" + subtitulo + "</p>" if subtitulo else ""}
    </div>
    """, unsafe_allow_html=True)

def metric_card(valor, etiqueta, color=""):
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">{etiqueta}</div>
        <div class="metric-value">{valor}</div>
    </div>
    """, unsafe_allow_html=True)

def interpretacion(texto, color=""):
    st.markdown(f'<div class="interpretacion {color}">{texto}</div>', unsafe_allow_html=True)

def nota(texto):
    st.markdown(f'<div class="nota">{texto}</div>', unsafe_allow_html=True)


# ─── Carga y limpieza de datos ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def cargar_datos():
    ruta = os.path.join(BASE_DIR, "COVID19_line_list_data.csv")
    if not os.path.exists(ruta):
        return None, f"Archivo no encontrado en: {ruta}"

    df = pd.read_csv(ruta)

    # Eliminar columnas sin nombre
    unnamed = df.columns[df.columns.str.startswith('Unnamed')]
    df = df.drop(columns=unnamed)

    # Reemplazar 'NA' en texto por NaN real
    df.replace('NA', np.nan, inplace=True)

    # Variables binarias de desenlace
    df['death_binary'] = df['death'].astype(str).apply(lambda x: 0 if x == '0' else 1)
    df['recovered_binary'] = df['recovered'].astype(str).apply(lambda x: 0 if x == '0' else 1)

    # Corregir edades < 1 (bebés) → 1
    df.loc[(df['age'] > 0) & (df['age'] < 1), 'age'] = 1

    # Parseo de fechas
    df['symptom_onset'] = pd.to_datetime(df['symptom_onset'], errors='coerce')
    df['hosp_visit_date'] = pd.to_datetime(df['hosp_visit_date'], errors='coerce')

    # Días desde inicio de síntomas hasta hospitalización
    df['dias_hasta_hosp'] = (df['hosp_visit_date'] - df['symptom_onset']).dt.days

    # Etiquetas de género en español
    df['genero'] = df['gender'].astype(str).str.lower().replace({'male': 'Hombre', 'female': 'Mujer'})

    # Clasificación vínculo con Wuhan
    df['visiting Wuhan'] = pd.to_numeric(df['visiting Wuhan'], errors='coerce')
    df['from Wuhan'] = pd.to_numeric(df['from Wuhan'], errors='coerce')

    def clasificar_wuhan(row):
        if row['from Wuhan'] == 1:
            return 'De Wuhan'
        elif row['visiting Wuhan'] == 1:
            return 'Visitó Wuhan'
        elif row['from Wuhan'] == 0 and row['visiting Wuhan'] == 0:
            return 'Sin vínculo con Wuhan'
        else:
            return 'No registrado'

    df['vinculo_wuhan'] = df.apply(clasificar_wuhan, axis=1)

    return df, None


df, error_msg = cargar_datos()


# ─── BARRA LATERAL ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0.5rem 0.5rem;">
        <div style="font-size:1.5rem;font-weight:700;color:#ff6b6b;letter-spacing:-0.5px;">COVID-19</div>
        <div style="font-size:0.82rem;color:#a0aec0;margin-top:2px;">Los primeros meses de la pandemia</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.8rem 0;border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#718096;padding:0 0.3rem;'>Navegacion</p>", unsafe_allow_html=True)

    seccion = st.radio(
        label="Secciones",
        options=[
            "Inicio",
            "1 — Edad y género",
            "2 — Tiempo hasta atención médica",
            "3 — Comparativa por país",
            "4 — Propagación en el tiempo",
            "Acerca de los datos",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:0.8rem 0;border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 0.3rem;font-size:0.78rem;color:#718096;line-height:1.8;">
        Fuente: Kaggle (pratik1235)<br>
        Datos: Ene–Feb 2020<br>
        Acceso: 26 feb 2026
    </div>
    """, unsafe_allow_html=True)


# ─── Error si no carga el archivo ────────────────────────────────────────────
if df is None:
    st.error(f"No se pudo cargar la base de datos. {error_msg}")
    st.info(
        "Asegúrate de que el archivo **COVID19_line_list_data.csv** "
        "se encuentre en la misma carpeta que este script."
    )
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# INICIO
# ═════════════════════════════════════════════════════════════════════════════
if seccion == "Inicio":
    banner(
        "COVID-19: ¿Qué nos dicen los datos de los primeros meses?",
        "Un análisis exploratorio de los registros de enero y febrero de 2020, al inicio del brote mundial."
    )

    st.markdown("""
    <div class="card">
    A finales de 2019, en la ciudad china de <strong>Wuhan</strong>, se detectó un nuevo virus respiratorio
    que luego sería conocido mundialmente como COVID-19. En pocas semanas comenzó a expandirse a otros países,
    y en marzo de 2020 fue declarado pandemia por la Organización Mundial de la Salud.<br><br>
    Este tablero analiza la información registrada durante <strong>enero y febrero de 2020</strong>,
    los primeros meses en que los datos empezaban a recopilarse. A través de cuatro preguntas concretas,
    buscamos entender quiénes se vieron más afectados, qué países concentraron los primeros casos
    y cómo se expandió el virus desde su epicentro.
    </div>
    """, unsafe_allow_html=True)

    total       = len(df)
    fallecidos  = int(df['death_binary'].sum())
    recuperados = int(df['recovered_binary'].sum())
    paises      = df['country'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{total:,}", "Casos registrados")
    with c2: metric_card(f"{fallecidos:,}", "Fallecidos reportados", "blue")
    with c3: metric_card(f"{recuperados:,}", "Recuperados reportados", "green")
    with c4: metric_card(f"{paises}", "Países en el dataset", "gray")

    nota("La mayoría de los casos aún no tenían un desenlace definitivo al momento de capturar los datos, ya que muchos pacientes seguían en tratamiento.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem;font-weight:600;color:#1a1a2e;margin-bottom:1rem;'>Las cuatro preguntas que responde este análisis</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="question-card">
            <h4>1. Edad y género</h4>
            <p>¿La edad o el género de una persona influyeron en su probabilidad de morir?</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="question-card q2">
            <h4>2. Atención médica</h4>
            <p>¿Llegar más rápido al hospital estuvo relacionado con una mayor probabilidad de sobrevivir?</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="question-card q3">
            <h4>3. Comparativa por país</h4>
            <p>¿Cuáles fueron los países con más casos, más muertes y más recuperados?</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="question-card q4">
            <h4>4. Propagación en el tiempo</h4>
            <p>¿Cómo y dónde se extendió el virus primero? ¿Qué papel jugó Wuhan?</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Usa el menu de la izquierda para navegar entre las secciones.")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — EDAD Y GÉNERO
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "1 — Edad y género":
    banner(
        "¿La edad y el género influyen en quiénes mueren?",
        "Comparamos la distribución de edades entre pacientes que sobrevivieron y quienes fallecieron, separado por género.",
        "banner-blue"
    )

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        rango_edad = st.slider(
            "Rango de edad a analizar",
            min_value=1, max_value=100,
            value=(1, 100)
        )
    with col_f2:
        generos_sel = st.multiselect(
            "Género",
            options=["Hombre", "Mujer"],
            default=["Hombre", "Mujer"]
        )
    with col_f3:
        tipo_grafico = st.radio(
            "Tipo de gráfico",
            options=["Violin", "Caja"],
            horizontal=True
        )

    # Preparar datos
    df_h1 = df.dropna(subset=['age']).copy()
    df_h1 = df_h1[
        (df_h1['age'] >= rango_edad[0]) &
        (df_h1['age'] <= rango_edad[1]) &
        (df_h1['genero'].isin(generos_sel))
    ]
    df_h1['death_binary'] = df_h1['death_binary'].astype(int)
    df_h1['Desenlace'] = df_h1['death_binary'].map({0: 'Sobrevivió', 1: 'Falleció'})

    if df_h1.empty:
        st.warning("No hay registros con los filtros seleccionados. Ajusta el rango de edad o el género.")
    else:
        if tipo_grafico == "Violin":
            fig = px.violin(
                df_h1, x='genero', y='age', color='Desenlace', box=True, points=False,
                color_discrete_map={'Sobrevivió': '#2980b9', 'Falleció': '#c0392b'},
                labels={'genero': 'Género', 'age': 'Edad'},
                title="Distribución de edad por género y desenlace"
            )
        else:
            fig = px.box(
                df_h1, x='genero', y='age', color='Desenlace', points='outliers',
                color_discrete_map={'Sobrevivió': '#2980b9', 'Falleció': '#c0392b'},
                labels={'genero': 'Género', 'age': 'Edad'},
                title="Distribución de edad por género y desenlace"
            )

        fig.update_layout(
            xaxis_title="Género", yaxis_title="Edad", legend_title="Desenlace",
            template="plotly_white", height=520,
            font=dict(family="Inter, sans-serif", size=13),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")

        med_fall_h = df_h1[(df_h1['death_binary']==1) & (df_h1['genero']=='Hombre')]['age'].median()
        med_fall_m = df_h1[(df_h1['death_binary']==1) & (df_h1['genero']=='Mujer')]['age'].median()
        n_fall     = int((df_h1['death_binary']==1).sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            val = f"{med_fall_h:.0f} años" if not np.isnan(med_fall_h) else "Sin datos"
            metric_card(val, "Edad mediana fallecidos — Hombres", "blue")
        with c2:
            val = f"{med_fall_m:.0f} años" if not np.isnan(med_fall_m) else "Sin datos"
            metric_card(val, "Edad mediana fallecidas — Mujeres")
        with c3:
            metric_card(f"{n_fall}", "Fallecidos en la selección", "gray")

        interpretacion(
            "<b>Como leer este grafico:</b> Cada figura muestra la distribucion de edades. "
            "Cuanto mas amplia sea la forma, mas personas tienen esa edad. La linea del medio indica la edad tipica del grupo.<br><br>"
            "<b>Lo que encontramos:</b> En ambos generos, las personas que fallecieron tenian en promedio mas edad que quienes sobrevivieron. "
            "En hombres, los fallecidos se concentran entre los 60 y 80 anos. Las mujeres muestran un patron similar, "
            "con una edad promedio de fallecimiento ligeramente mayor, lo que podria sugerir una mayor longevidad.",
            "blue"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — TIEMPO HASTA ATENCIÓN MÉDICA
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "2 — Tiempo hasta atención médica":
    banner(
        "¿Llegar antes al hospital marca la diferencia?",
        "Analizamos cuántos días tardaron los pacientes en ir al hospital desde sus primeros síntomas, y si esto influyó en su desenlace.",
        "banner-green"
    )

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        max_dias = st.slider(
            "Máximo de días a mostrar en el gráfico",
            min_value=5, max_value=60,
            value=30,
            help="Elimina valores extremos para una mejor visualización."
        )
    with col_f2:
        paises_disp = sorted(df['country'].dropna().unique().tolist())
        paises_h2 = st.multiselect(
            "Filtrar por país (opcional — vacío muestra todos)",
            options=paises_disp,
            default=[],
            placeholder="Todos los países",
        )

    # Preparar datos
    df_h2 = df.dropna(subset=['dias_hasta_hosp', 'death_binary']).copy()
    df_h2['death_binary'] = df_h2['death_binary'].astype(int)
    df_h2 = df_h2[(df_h2['dias_hasta_hosp'] >= 0) & (df_h2['dias_hasta_hosp'] <= max_dias)]
    if paises_h2:
        df_h2 = df_h2[df_h2['country'].isin(paises_h2)]
    df_h2['Desenlace'] = df_h2['death_binary'].map({0: 'Sobrevivió', 1: 'Falleció'})

    if df_h2.empty:
        st.warning("No hay datos con los filtros seleccionados.")
    else:
        fig = px.box(
            df_h2, x='Desenlace', y='dias_hasta_hosp', color='Desenlace', points='all',
            color_discrete_map={'Sobrevivió': '#27ae60', 'Falleció': '#c0392b'},
            labels={'dias_hasta_hosp': 'Días desde síntomas hasta hospital', 'Desenlace': 'Desenlace'},
            title="Tiempo hasta la primera atención médica según el desenlace del paciente"
        )
        fig.update_layout(
            xaxis_title="Desenlace del paciente",
            yaxis_title="Días desde primeros síntomas hasta hospital",
            template="plotly_white", showlegend=False, height=520,
            font=dict(family="Inter, sans-serif", size=13),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")

        med_vivos   = df_h2[df_h2['death_binary']==0]['dias_hasta_hosp'].median()
        med_muertos = df_h2[df_h2['death_binary']==1]['dias_hasta_hosp'].median()
        diff = med_muertos - med_vivos

        c1, c2, c3 = st.columns(3)
        with c1: metric_card(f"{med_vivos:.0f} días", "Días medianos — Sobrevivientes", "green")
        with c2: metric_card(f"{med_muertos:.0f} días", "Días medianos — Fallecidos")
        with c3: metric_card(f"+{diff:.0f} días" if diff >= 0 else f"{diff:.0f} días", "Diferencia entre grupos", "gray")

        interpretacion(
            "<b>Como leer este grafico:</b> Cada caja muestra la distribucion de dias que tardaron los pacientes en ir al hospital. "
            "La linea del centro es el valor tipico (mediana) y los puntos son casos individuales.<br><br>"
            "<b>Lo que encontramos:</b> Los pacientes que fallecieron tardaron en promedio mas dias en llegar al hospital "
            "desde que empezaron con sintomas. Esto sugiere que una atencion medica mas oportuna pudo estar relacionada "
            "con mayores posibilidades de sobrevivir. Sin embargo, la edad y las condiciones previas de salud "
            "tambien son factores determinantes en el desenlace.",
            "green"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — COMPARATIVA POR PAÍS
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "3 — Comparativa por país":
    banner(
        "¿Qué países tuvieron más casos, muertes y recuperados?",
        "Comparativa de los países con mayor número de casos registrados durante enero y febrero de 2020."
    )


    df_h3 = df.copy()
    df_h3['death_binary'] = df_h3['death_binary'].astype(int)
    df_h3['recovered_binary'] = df_h3['recovered_binary'].astype(int)
    df_h3 = df_h3.dropna(subset=['country'])

    agg = (
        df_h3.groupby('country')
        .agg(
            Casos=('country', 'count'),
            Muertes=('death_binary', 'sum'),
            Recuperados=('recovered_binary', 'sum'),
        )
        .reset_index()
        .sort_values('Casos', ascending=False)
    )

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        top_n = st.slider("Número de países a mostrar", min_value=3, max_value=20, value=10)
    with col_f2:
        metrica = st.radio(
            "¿Qué quieres ver?",
            options=["Casos, muertes y recuperados", "Solo muertes", "Solo recuperados"],
            horizontal=True,
        )

    agg_top = agg.head(top_n)

    if metrica == "Casos, muertes y recuperados":
        df_melt = agg_top.melt(
            id_vars='country',
            value_vars=['Casos', 'Recuperados', 'Muertes'],
            var_name='Categoría',
            value_name='Personas',
        )
        color_map = {'Casos': '#3498db', 'Recuperados': '#2ecc71', 'Muertes': '#e74c3c'}
        fig = px.bar(
            df_melt,
            x='country',
            y='Personas',
            color='Categoría',
            barmode='group',
            color_discrete_map=color_map,
            labels={'country': 'País', 'Personas': 'Número de personas', 'Categoría': 'Categoría'},
            title=f"Casos, muertes y recuperados — Top {top_n} países",
        )
        fig.update_layout(xaxis_title="País", yaxis_title="Número de personas")
    else:
        col = 'Muertes' if metrica == "Solo muertes" else 'Recuperados'
        color = '#e74c3c' if col == 'Muertes' else '#2ecc71'
        fig = px.bar(
            agg_top.sort_values(col, ascending=True),
            x=col,
            y='country',
            orientation='h',
            color_discrete_sequence=[color],
            labels={'country': 'País', col: col},
            title=f"{col} por país — Top {top_n}",
        )
        fig.update_layout(xaxis_title=col, yaxis_title="País")

    fig.update_layout(
        template="plotly_white", legend_title="Categoría", height=520,
        font=dict(family="Inter, sans-serif", size=13),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig, width="stretch")

    total_casos   = int(agg_top['Casos'].sum())
    total_muertes = int(agg_top['Muertes'].sum())
    total_recup   = int(agg_top['Recuperados'].sum())
    c1, c2, c3 = st.columns(3)
    with c1: metric_card(f"{total_casos:,}", f"Total casos — Top {top_n}", "blue")
    with c2: metric_card(f"{total_muertes:,}", f"Total muertes — Top {top_n}")
    with c3: metric_card(f"{total_recup:,}", f"Total recuperados — Top {top_n}", "green")

    interpretacion(
        "<b>Lo que encontramos:</b> China fue el pais con mas casos confirmados y fallecidos en este periodo, "
        "lo cual es consistente con ser el origen del brote. Singapur presento un numero considerable de casos "
        "pero con mas recuperados que fallecidos. Alemania, Tailandia y Espana no registraron muertes en este corte de tiempo.<br><br>"
        "Es importante recordar que muchos de estos casos aun estaban activos cuando se recopilaron los datos, "
        "por lo que las cifras no son definitivas."
    )

    with st.expander("Ver tabla de datos completa"):
        st.dataframe(
            agg.head(top_n).rename(columns={'country': 'País'}).reset_index(drop=True),
            width="stretch", hide_index=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — PROPAGACIÓN EN EL TIEMPO
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "4 — Propagación en el tiempo":
    banner(
        "¿Cómo y dónde se propagó el virus primero?",
        "Evolución de los casos por país desde el inicio de síntomas, con trazabilidad al epicentro en Wuhan.",
        "banner-purple"
    )

    df_h4 = df.dropna(subset=['symptom_onset', 'country']).copy()

    df_grouped = (
        df_h4.groupby(['country', 'symptom_onset', 'vinculo_wuhan'])
        .size()
        .reset_index(name='casos')
        .sort_values(['country', 'symptom_onset'])
    )
    df_grouped['casos_acumulados'] = (
        df_grouped.groupby(['country', 'vinculo_wuhan'])['casos'].cumsum()
    )

    todos_paises = sorted(df_grouped['country'].unique().tolist())
    top5_default = (
        df_grouped.groupby('country')['casos']
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    # Filtros
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        paises_sel = st.multiselect(
            "Países a mostrar",
            options=todos_paises,
            default=top5_default,
        )
    with col_f2:
        tipo_casos = st.radio("Tipo de casos", options=["Acumulados", "Diarios"], horizontal=True)
        mostrar_wuhan = st.checkbox("Distinguir vínculo con Wuhan", value=False)

    # Rango de fechas
    fecha_min = df_grouped['symptom_onset'].min().date()
    fecha_max = df_grouped['symptom_onset'].max().date()
    rango_fechas = st.slider(
        "Rango de fechas",
        min_value=fecha_min,
        max_value=fecha_max,
        value=(fecha_min, fecha_max),
        format="DD/MM/YYYY",
    )

    # Aplicar filtros
    df_plot = df_grouped[
        (df_grouped['country'].isin(paises_sel)) &
        (df_grouped['symptom_onset'] >= pd.to_datetime(rango_fechas[0])) &
        (df_grouped['symptom_onset'] <= pd.to_datetime(rango_fechas[1]))
    ].copy()

    y_col = 'casos_acumulados' if tipo_casos == "Acumulados" else 'casos'
    y_label = 'Casos acumulados' if tipo_casos == "Acumulados" else 'Casos diarios'

    if df_plot.empty:
        st.warning("No hay datos para los filtros seleccionados. Intenta seleccionar otros países o ampliar el rango de fechas.")
    else:
        if mostrar_wuhan:
            fig = px.line(
                df_plot,
                x='symptom_onset',
                y=y_col,
                color='country',
                line_dash='vinculo_wuhan',
                markers=True,
                labels={
                    'symptom_onset': 'Fecha de inicio de síntomas',
                    y_col: y_label,
                    'country': 'País',
                    'vinculo_wuhan': 'Vínculo con Wuhan',
                },
                title=f"Propagación del COVID-19 — {tipo_casos} (con vínculo a Wuhan)",
            )
            fig.update_layout(legend_title="País / Vínculo con Wuhan")
        else:
            df_total = (
                df_plot.groupby(['country', 'symptom_onset'])['casos']
                .sum()
                .reset_index()
                .sort_values(['country', 'symptom_onset'])
            )
            df_total['casos_acumulados'] = df_total.groupby('country')['casos'].cumsum()
            y_col2 = 'casos_acumulados' if tipo_casos == "Acumulados" else 'casos'

            fig = px.line(
                df_total,
                x='symptom_onset',
                y=y_col2,
                color='country',
                markers=True,
                labels={
                    'symptom_onset': 'Fecha de inicio de síntomas',
                    y_col2: y_label,
                    'country': 'País',
                },
                title=f"Propagación del COVID-19 — {tipo_casos}",
            )
            fig.update_layout(legend_title="País")

        fig.update_layout(
            xaxis_title="Fecha de inicio de síntomas", yaxis_title=y_label,
            hovermode="x unified", template="plotly_white", height=560,
            font=dict(family="Inter, sans-serif", size=13),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")

        interpretacion(
            "<b>Lo que encontramos:</b> Los primeros casos registrados fuera de China llegaron de personas "
            "que habian visitado Wuhan, lo que confirma que el virus salio de ese epicentro a traves de viajeros. "
            "China concentro el mayor numero de casos desde el inicio. "
            "Japon registro un numero significativo de casos, muchos sin vinculo directo con Wuhan, "
            "lo que podria indicar transmision comunitaria temprana.<br><br>"
            "Activa la opcion <em>Distinguir vinculo con Wuhan</em> para ver como cambia el patron "
            "segun la relacion de cada paciente con la ciudad epicentro del brote.",
            "purple"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN: ACERCA DE LOS DATOS
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "Acerca de los datos":
    banner(
        "Acerca de los datos",
        "Origen de la información, fecha de acceso y cómo mantener este tablero actualizado.",
        "banner-dark"
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("""
        <div class="card">
        <p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9ca3af;margin-bottom:0.8rem;">Fuente principal</p>
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
            <tr><td style="padding:5px 0;color:#6b7280;width:45%">Nombre</td><td style="color:#1a1a2e;font-weight:500">COVID-19 Line List Data</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Plataforma</td><td style="color:#1a1a2e;font-weight:500">Kaggle</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Autor</td><td style="color:#1a1a2e;font-weight:500">pratik1235</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Archivo</td><td style="color:#1a1a2e;font-weight:500">COVID19_line_list_data.csv</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Fecha de acceso</td><td style="color:#1a1a2e;font-weight:500">26 de febrero de 2026</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Período cubierto</td><td style="color:#1a1a2e;font-weight:500">Ene 22, 2020 — Feb 2020</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Tipo de datos</td><td style="color:#1a1a2e;font-weight:500">Un registro por paciente</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Referencia</td><td style="color:#1a1a2e;font-weight:500">CDC</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9ca3af;margin-bottom:0.8rem;">Herramientas utilizadas</p>
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
            <tr><td style="padding:5px 0;color:#6b7280;width:45%">Python 3</td><td style="color:#1a1a2e;">Lenguaje de programación principal</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Pandas</td><td style="color:#1a1a2e;">Procesamiento y limpieza de datos</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Plotly</td><td style="color:#1a1a2e;">Visualizaciones interactivas</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">Streamlit</td><td style="color:#1a1a2e;">Construcción y despliegue del tablero</td></tr>
            <tr><td style="padding:5px 0;color:#6b7280;">NumPy</td><td style="color:#1a1a2e;">Cálculos numéricos auxiliares</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
        <p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9ca3af;margin-bottom:0.8rem;">¿Qué contiene la base de datos?</p>
        <p style="font-size:0.9rem;color:#374151;line-height:1.65;">
        El dataset registra informacion a nivel individual para cada caso confirmado durante
        las primeras semanas del brote. Cada fila representa un paciente e incluye:
        </p>
        <ul style="font-size:0.9rem;color:#374151;line-height:1.9;padding-left:1.2rem;">
            <li><strong>Datos personales:</strong> edad, genero y pais.</li>
            <li><strong>Exposicion al virus:</strong> si el paciente vivia en Wuhan o la habia visitado.</li>
            <li><strong>Cronologia clinica:</strong> fechas de inicio de sintomas y de primera visita al hospital.</li>
            <li><strong>Desenlace:</strong> si el paciente fallecio o se recupero al momento del registro.</li>
            <li><strong>Descripcion del caso:</strong> notas en texto sobre el historial del paciente.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9ca3af;margin-bottom:0.8rem;">Limitaciones importantes</p>
        <ul style="font-size:0.9rem;color:#374151;line-height:1.9;padding-left:1.2rem;">
            <li>Cubre unicamente el inicio de la pandemia, no su impacto total.</li>
            <li>Gran parte de los casos no tenian desenlace definido al registrarse.</li>
            <li>La recopilacion de datos era incompleta y puede tener sesgos.</li>
            <li>Este analisis es exploratorio y no reemplaza estudios clinicos formales.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9ca3af;margin-bottom:0.8rem;">Como mantener este tablero actualizado</p>
    <ol style="font-size:0.9rem;color:#374151;line-height:1.9;padding-left:1.2rem;">
        <li><strong>Revisar Kaggle</strong> periodicamente para verificar si existen versiones mas completas del dataset.</li>
        <li><strong>Our World in Data</strong> (ourworldindata.org/covid-deaths) es una fuente complementaria con datos actualizados a nivel de paises.</li>
        <li><strong>Reemplazar el archivo CSV</strong> en la carpeta del proyecto con la versión mas reciente. Las visualizaciones se actualizarán automáticamente al recargar el tablero.</li>
        <li>Si se adoptan datos agregados por país, las secciones de edad y género requerirán ajustes metodológicos.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
