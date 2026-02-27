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

# ─── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #c0392b; }
    h2 { color: #2c3e50; border-bottom: 2px solid #e8e8e8; padding-bottom: 6px; }
    .interpretacion {
        background: #fff8f8;
        border-left: 5px solid #c0392b;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin-top: 1.2rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("## COVID-19")
    st.caption("Los primeros meses de la pandemia")
    st.write("Datos recopilados entre **enero y febrero de 2020**, al inicio del brote mundial.")
    st.divider()

    seccion = st.radio(
        "Ir a sección:",
        options=[
            "Inicio",
            "1 — Edad y género",
            "2 — Tiempo hasta atención médica",
            "3 — Comparativa por país",
            "4 — Propagación en el tiempo",
            "Acerca de los datos",
        ],
        index=0,
    )

    st.divider()
    st.caption("Fuente: Kaggle (pratik1235)")
    st.caption("Fecha de acceso: 26 de febrero de 2026")


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
    st.title("COVID-19: ¿Qué nos dicen los datos de los primeros meses?")

    st.write(
        """
        A finales de 2019, en la ciudad china de Wuhan, se detectó un nuevo virus respiratorio
        que luego sería conocido mundialmente como COVID-19. En pocas semanas, comenzó a extenderse
        a otros países y en marzo de 2020 fue declarado pandemia por la Organización Mundial de la Salud.

        Este tablero analiza la información registrada durante **enero y febrero de 2020**,
        los primeros meses en que los datos empezaban a recopilarse. A través de cuatro preguntas
        concretas, buscamos entender quiénes se vieron más afectados, qué países
        concentraron los primeros casos y cómo se expandió el virus.
        """
    )

    st.divider()

    # Métricas rápidas
    total = len(df)
    fallecidos = int(df['death_binary'].sum())
    recuperados = int(df['recovered_binary'].sum())
    paises = df['country'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos registrados", f"{total:,}")
    c2.metric("Fallecidos reportados", f"{fallecidos:,}")
    c3.metric("Recuperados reportados", f"{recuperados:,}")
    c4.metric("Países en el dataset", f"{paises}")

    st.caption(
        "Nota: la mayoría de los casos registrados no tenían un desenlace definitivo "
        "al momento de capturar los datos, ya que muchos pacientes aún estaban en tratamiento."
    )

    st.divider()
    st.subheader("Las cuatro preguntas que responde este análisis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **1. ¿La edad y el género influyen en quiénes mueren?**
            Exploramos si el perfil del paciente se relaciona con un mayor riesgo de muerte.

            ---

            **2. ¿Llegar antes al hospital marca la diferencia?**
            Analizamos si el tiempo desde los primeros síntomas hasta la atención médica
            tuvo impacto en el desenlace del paciente.
            """
        )
    with col2:
        st.markdown(
            """
            **3. ¿Qué países tuvieron más casos, muertes y recuperados?**
            Comparamos los países con mayor número de casos durante este período.

            ---

            **4. ¿Cómo y dónde se propagó el virus primero?**
            Observamos la evolución de los casos en el tiempo y el papel central de Wuhan.
            """
        )

    st.divider()
    st.info("Usa el menú de la izquierda para navegar entre las secciones.")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — EDAD Y GÉNERO
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "1 — Edad y género":
    st.title("¿La edad y el género influyen en quiénes mueren?")
    st.write(
        """
        Una de las primeras preguntas fue si características personales como la edad o el género
        podían hacer a alguien más vulnerable al virus. Este gráfico compara la distribución de
        edades de quienes sobrevivieron y quienes fallecieron, separado por género.
        """
    )

    st.divider()

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
                df_h1,
                x='genero',
                y='age',
                color='Desenlace',
                box=True,
                points=False,
                color_discrete_map={'Sobrevivió': '#3498db', 'Falleció': '#e74c3c'},
                labels={'genero': 'Género', 'age': 'Edad'},
                title="Distribución de edad por género y desenlace",
            )
        else:
            fig = px.box(
                df_h1,
                x='genero',
                y='age',
                color='Desenlace',
                points='outliers',
                color_discrete_map={'Sobrevivió': '#3498db', 'Falleció': '#e74c3c'},
                labels={'genero': 'Género', 'age': 'Edad'},
                title="Distribución de edad por género y desenlace",
            )

        fig.update_layout(
            xaxis_title="Género",
            yaxis_title="Edad",
            legend_title="Desenlace",
            template="plotly_white",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas de apoyo
        c1, c2 = st.columns(2)
        med_fall_h = df_h1[(df_h1['death_binary'] == 1) & (df_h1['genero'] == 'Hombre')]['age'].median()
        med_fall_m = df_h1[(df_h1['death_binary'] == 1) & (df_h1['genero'] == 'Mujer')]['age'].median()
        c1.metric("Edad mediana de fallecidos — Hombres", f"{med_fall_h:.0f} años" if not np.isnan(med_fall_h) else "Sin datos")
        c2.metric("Edad mediana de fallecidas — Mujeres", f"{med_fall_m:.0f} años" if not np.isnan(med_fall_m) else "Sin datos")

        st.markdown(
            """
            <div class="interpretacion">
            <b>Como leer este grafico</b><br>
            Cada figura muestra la distribucion de edades. Cuanto mas amplia sea la forma,
            mas personas tienen esa edad. La linea del medio indica la edad promedio del grupo.<br><br>
            <b>Lo que encontramos:</b> en ambos generos, las personas que fallecieron tenian
            en promedio mas edad que quienes sobrevivieron. En hombres, los fallecidos se
            concentran entre los 60 y 80 anos. Las mujeres muestran un patron similar,
            pero con una edad promedio de fallecimiento ligeramente mayor. Esto sugiere que
            la edad avanzada fue un factor de riesgo importante ante el COVID-19.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — TIEMPO HASTA ATENCIÓN MÉDICA
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "2 — Tiempo hasta atención médica":
    st.title("¿Llegar antes al hospital marca la diferencia?")
    st.write(
        """
        Esta sección analiza cuántos días pasaron desde que un paciente empezó a sentirse mal
        hasta que fue al hospital. ¿Los que tardaron más en atenderse tuvieron peores resultados?
        """
    )

    st.divider()

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
            df_h2,
            x='Desenlace',
            y='dias_hasta_hosp',
            color='Desenlace',
            points='all',
            color_discrete_map={'Sobrevivió': '#2ecc71', 'Falleció': '#c0392b'},
            labels={
                'dias_hasta_hosp': 'Días desde síntomas hasta hospital',
                'Desenlace': 'Desenlace',
            },
            title="Tiempo hasta la primera atención médica según el desenlace del paciente",
        )
        fig.update_layout(
            xaxis_title="Desenlace del paciente",
            yaxis_title="Días desde primeros síntomas hasta hospital",
            template="plotly_white",
            showlegend=False,
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Métricas de apoyo
        med_vivos = df_h2[df_h2['death_binary'] == 0]['dias_hasta_hosp'].median()
        med_muertos = df_h2[df_h2['death_binary'] == 1]['dias_hasta_hosp'].median()
        c1, c2 = st.columns(2)
        c1.metric("Días medianos hasta hospital — Sobrevivientes", f"{med_vivos:.0f} días")
        c2.metric("Días medianos hasta hospital — Fallecidos", f"{med_muertos:.0f} días")

        st.markdown(
            """
            <div class="interpretacion">
            <b>Como leer este grafico</b><br>
            Cada caja muestra la distribucion de dias que tardaron los pacientes en ir al hospital.
            La linea del centro es el valor tipico (mediana) y los puntos son casos individuales.<br><br>
            <b>Lo que encontramos:</b> las personas que fallecieron tardaron, en promedio, mas dias
            en llegar al hospital desde que empezaron con sintomas. Esto sugiere que una atencion
            medica mas oportuna pudo estar relacionada con mayores posibilidades de sobrevivir.
            Es importante aclarar que la edad y otras condiciones de salud previas tambien juegan
            un papel importante en el desenlace.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — COMPARATIVA POR PAÍS
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "3 — Comparativa por país":
    st.title("¿Qué países tuvieron más casos, muertes y recuperados?")
    st.write(
        """
        Esta sección compara los países con mayor número de casos registrados durante enero
        y febrero de 2020. Permite identificar cuáles concentraron el mayor impacto y cuáles
        lograron mejores resultados en cuanto a recuperaciones.
        """
    )

    st.divider()

    # Preparar datos
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
        template="plotly_white",
        legend_title="Categoría",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="interpretacion">
        <b>Lo que encontramos:</b> China fue el pais con mas casos confirmados y fallecidos
        en este periodo, lo cual es consistente con ser el origen del brote. Singapur
        presento un numero considerable de casos pero con mas recuperados que fallecidos.
        Alemania, Tailandia y Espana no registraron muertes en este corte de tiempo.<br><br>
        Es importante recordar que muchos de estos casos aun estaban activos cuando se
        recopilaron los datos, por lo que las cifras de recuperados y fallecidos no son definitivas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Ver tabla de datos"):
        st.dataframe(
            agg.head(top_n).rename(columns={'country': 'País'}).reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — PROPAGACIÓN EN EL TIEMPO
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "4 — Propagación en el tiempo":
    st.title("¿Cómo y dónde se propagó el virus primero?")
    st.write(
        """
        Esta sección muestra cómo creció el número de casos en distintos países desde
        el inicio de los síntomas. También permite explorar si los primeros pacientes
        de cada país tenían alguna relación con Wuhan, la ciudad epicentro del brote.
        """
    )

    st.divider()

    # Preparar datos
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
            xaxis_title="Fecha de inicio de síntomas",
            yaxis_title=y_label,
            hovermode="x unified",
            template="plotly_white",
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="interpretacion">
            <b>Lo que encontramos:</b> los primeros casos registrados fuera de China llegaron de
            personas que habian visitado Wuhan, lo que confirma que el virus salio de ese epicentro
            a traves de viajeros. China concentro el mayor numero de casos desde el inicio.
            Japon registro un numero significativo de casos, muchos sin vinculo directo con Wuhan,
            lo que podria indicar una transmision comunitaria temprana.<br><br>
            Activa la opcion "Distinguir vinculo con Wuhan" para ver como cambia el patron
            segun la relacion de cada paciente con la ciudad origen del brote.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN: ACERCA DE LOS DATOS
# ═════════════════════════════════════════════════════════════════════════════
elif seccion == "Acerca de los datos":
    st.title("Acerca de los datos")
    st.write(
        "Esta sección documenta el origen de la información utilizada, cuándo fue accedida "
        "y cómo se podría mantener este tablero actualizado en el futuro."
    )

    st.divider()

    st.subheader("Fuente principal")
    st.markdown(
        """
        | Campo | Detalle |
        |---|---|
        | **Nombre del dataset** | COVID-19 Line List Data |
        | **Plataforma** | Kaggle |
        | **Autor** | pratik1235 |
        | **Enlace** | https://www.kaggle.com/datasets/pratik1235/covid19-csea |
        | **Archivo usado** | COVID19_line_list_data.csv |
        | **Fecha de acceso** | 26 de febrero de 2026 |
        | **Período cubierto** | 22 de enero de 2020 — febrero de 2020 |
        | **Tipo de datos** | Un registro por cada paciente confirmado |
        | **Organismo citado** | Centers for Disease Control and Prevention (CDC) |
        """
    )

    st.divider()

    st.subheader("¿Qué contiene la base de datos?")
    st.write(
        """
        El dataset registra información a nivel individual para cada caso confirmado de COVID-19
        durante las primeras semanas del brote. Cada fila representa un paciente e incluye:

        - **Datos personales:** edad, género y país.
        - **Exposición al virus:** si el paciente vivía en Wuhan o la había visitado.
        - **Cronología clínica:** fechas de inicio de síntomas y de primera visita al hospital.
        - **Desenlace:** si el paciente falleció o se recuperó al momento del registro.
        - **Descripción del caso:** notas en texto libre sobre el historial del paciente.
        """
    )

    st.divider()

    st.subheader("Limitaciones importantes")
    st.write(
        """
        - Los datos cubren únicamente el inicio de la pandemia. No reflejan el impacto total del COVID-19.
        - Gran parte de los casos no tenía un desenlace definido al registrarse, porque los pacientes
          aún estaban en tratamiento.
        - La recopilación de datos en ese momento era incompleta y puede contener sesgos de registro.
        - Este análisis es de tipo exploratorio y no reemplaza estudios clínicos formales.
        """
    )

    st.divider()

    st.subheader("¿Cómo mantener este tablero actualizado?")
    st.write(
        """
        Para extender o actualizar este análisis en el futuro, se recomienda:

        1. **Revisar la fuente original en Kaggle** para verificar si existen versiones
           más completas del dataset.
        2. **Considerar Our World in Data** (https://ourworldindata.org/covid-deaths) como fuente
           complementaria. Actualiza sus datos con regularidad y cubre la pandemia completa
           a nivel de países.
        3. **Reemplazar el archivo CSV** `COVID19_line_list_data.csv` en la carpeta del proyecto
           con la versión más reciente y volver a ejecutar el dashboard. Las visualizaciones
           se actualizarán automáticamente.
        4. Si se adoptan datos agregados por país (en lugar de individuales), las secciones de
           edad y género requerirán ajustes en la metodología, ya que esas variables no están
           disponibles a nivel agregado.
        """
    )

    st.divider()

    st.subheader("Herramientas utilizadas")
    st.markdown(
        """
        | Herramienta | Uso |
        |---|---|
        | Python 3 | Lenguaje de programación principal |
        | Pandas | Procesamiento y limpieza de datos |
        | Plotly | Visualizaciones interactivas |
        | Streamlit | Construcción y despliegue del tablero |
        | NumPy | Cálculos numéricos auxiliares |
        """
    )
