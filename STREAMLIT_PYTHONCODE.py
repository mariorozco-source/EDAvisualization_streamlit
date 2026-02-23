# ============================================================
# EDA OWID COVID-19 — Streamlit Interactive Dashboard
# Marco QUEST: Question → Understand → Explore → Study → Tell
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 OWID Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────
C1 = "#2C5F2D"
C2 = "#97BC62"
C3 = "#B8860B"
C4 = "#8B3A3A"
CONT_COLORS = ["#2C5F2D", "#97BC62", "#B8860B", "#8B3A3A", "#5B7FA6", "#7B5EA7"]

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv(base_dir / "owid-covid-data.csv", parse_dates=["date"])

    non_countries = [
        "World", "Africa", "Asia", "Europe", "European Union",
        "High income", "Low income", "Lower middle income",
        "North America", "Oceania", "South America", "Upper middle income",
        "International",
    ]
    df = df[~df["location"].isin(non_countries) & df["continent"].notna()].copy()

    snap = df.sort_values("date").groupby("location").last().reset_index()
    snap["cfr"] = (snap["total_deaths"] / snap["total_cases"] * 100).round(3)
    snap["log_deaths_pm"] = np.log1p(snap["total_deaths_per_million"])
    snap["log_gdp"] = np.log10(snap["gdp_per_capita"].clip(lower=1))
    return df, snap

df, snap = load_data()
CONTINENTS = sorted(snap["continent"].dropna().unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🦠 COVID-19 Dashboard")
st.sidebar.markdown("**Marco QUEST — OWID Dataset**")

page = st.sidebar.radio(
    "Navegar a:",
    [
        "🏠 Introducción",
        "📊 Q1: Demografía y Mortalidad",
        "🏥 Q2: Capacidad Sanitaria",
        "🌍 Q3: Comparativa por País",
        "📈 Q4: Propagación Temporal",
        "📂 Fuente de Datos",
    ],
)

selected_continents = st.sidebar.multiselect(
    "Filtrar continentes:", options=CONTINENTS, default=CONTINENTS
)
min_pop = st.sidebar.slider(
    "Población mínima del país (millones):", 0, 100, 1, 1
)

snap_f = snap[
    snap["continent"].isin(selected_continents) &
    (snap["population"] >= min_pop * 1e6)
].copy()

cont_palette = dict(zip(CONTINENTS, CONT_COLORS))
st.sidebar.markdown("---")
st.sidebar.caption(f"Países en vista: **{len(snap_f)}** de {len(snap)}")

# =============================================================
# PAGE 0 — INTRODUCCIÓN
# =============================================================
if page == "🏠 Introducción":
    st.title("Análisis Exploratorio de Datos: COVID-19 (OWID)")
    st.markdown("""
## Marco QUEST aplicado al dataset global de COVID-19

Este dashboard presenta el EDA del dataset **Our World in Data (OWID)** sobre COVID-19,
con estadísticas diarias por país: casos, muertes, y más de 60 indicadores socioeconómicos
y de salud pública.

---
### Preguntas Analíticas

| # | Pregunta |
|---|----------|
| Q1 | ¿Existe relación entre la demografía (edad mediana, mayores de 65) y la mortalidad? |
| Q2 | ¿Cómo influye la capacidad hospitalaria y el PIB per cápita en las muertes? |
| Q3 | ¿Qué países y continentes presentan mayor/menor mortalidad proporcional? |
| Q4 | ¿Qué patrones temporales se observan en la propagación del virus? |

---
### Cómo usar este dashboard
- Navega con el menú de la izquierda.
- Los **filtros de continente y población mínima** se aplican a Q1–Q3.
- Cada sección incluye interpretación de resultados.
""")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Países analizados", len(snap))
    c2.metric("Registros totales", f"{len(df):,}")
    c3.metric("Fecha inicio", str(df["date"].min().date()))
    c4.metric("Fecha fin", str(df["date"].max().date()))

    st.markdown("---")
    st.subheader("Vista previa del dataset")
    st.dataframe(
        df[["location", "continent", "date", "total_cases", "total_deaths",
            "total_cases_per_million", "total_deaths_per_million"]].head(10),
        use_container_width=True,
    )

# =============================================================
# PAGE 1 — Q1: DEMOGRAFÍA Y MORTALIDAD
# =============================================================
elif page == "📊 Q1: Demografía y Mortalidad":
    st.title("Q1: ¿Impacta la demografía en la mortalidad por COVID-19?")
    st.markdown("""
**Hipótesis:** Los países con poblaciones más envejecidas (`median_age`, `aged_65_older`)
presentan tasas de mortalidad por COVID-19 más altas por millón de habitantes.
""")

    demo_df = snap_f.dropna(
        subset=["median_age", "aged_65_older", "total_deaths_per_million"]
    ).copy()

    col_a, col_b = st.columns([3, 1])
    with col_b:
        log_y = st.checkbox("Escala log (eje Y)", value=True)
        show_trend = st.checkbox("Línea de tendencia", value=True)

    y_col = "log_deaths_pm" if log_y else "total_deaths_per_million"
    y_lbl = "log(1 + Muertes/Millón)" if log_y else "Muertes por Millón"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cont in demo_df["continent"].unique():
        sub = demo_df[demo_df["continent"] == cont]
        axes[0].scatter(sub["median_age"], sub[y_col],
                        label=cont, alpha=0.7, s=40, color=cont_palette.get(cont))
        axes[1].scatter(sub["aged_65_older"], sub[y_col],
                        label=cont, alpha=0.7, s=40, color=cont_palette.get(cont))

    for ax, xcol, xlbl in zip(
        axes,
        ["median_age", "aged_65_older"],
        ["Edad Mediana", "Mayores de 65 (%)"]
    ):
        if show_trend:
            vd = demo_df[[xcol, y_col]].dropna()
            z = np.polyfit(vd[xcol], vd[y_col], 1)
            xs = np.linspace(vd[xcol].min(), vd[xcol].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), "k--", lw=1.5, label="Tendencia")
        ax.set_xlabel(xlbl, fontsize=11)
        ax.set_ylabel(y_lbl, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Edad Mediana vs Mortalidad", fontsize=12)
    axes[1].set_title("% Mayores 65 vs Mortalidad", fontsize=12)
    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    with col_a:
        st.pyplot(fig)
    plt.close()

    st.info("""
**Interpretación:** Se observa correlación positiva entre envejecimiento poblacional
y mortalidad proporcional. Los países europeos concentran los valores más altos,
consistente con que la edad es el mayor factor de riesgo individual para mortalidad
por SARS-CoV-2.
""")

# =============================================================
# PAGE 2 — Q2: CAPACIDAD SANITARIA
# =============================================================
elif page == "🏥 Q2: Capacidad Sanitaria":
    st.title("Q2: ¿Protege la capacidad hospitalaria contra la mortalidad?")
    st.markdown("""
**Hipótesis:** Países con mayor `hospital_beds_per_thousand` y `gdp_per_capita`
presentarán menor mortalidad proporcional por COVID-19.
""")

    health_df = snap_f.dropna(
        subset=["hospital_beds_per_thousand", "gdp_per_capita", "total_deaths_per_million"]
    ).copy()

    color_var = st.selectbox(
        "Colorear puntos por:",
        ["continent", "median_age", "aged_65_older"],
        format_func=lambda x: {"continent": "Continente",
                                "median_age": "Edad Mediana",
                                "aged_65_older": "% Mayores 65"}[x]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xcol, xlbl, title in zip(
        axes,
        ["hospital_beds_per_thousand", "log_gdp"],
        ["Camas Hosp. / 1,000 hab.", "log₁₀(PIB per cápita)"],
        ["Infraestructura Hospitalaria vs Mortalidad",
         "Riqueza Nacional vs Mortalidad"]
    ):
        if color_var == "continent":
            for cont in health_df["continent"].unique():
                sub = health_df[health_df["continent"] == cont]
                ax.scatter(sub[xcol], sub["log_deaths_pm"],
                           label=cont, alpha=0.7, s=40,
                           color=cont_palette.get(cont))
            ax.legend(fontsize=7, ncol=2)
        else:
            sc = ax.scatter(health_df[xcol], health_df["log_deaths_pm"],
                            c=health_df[color_var], cmap="YlGn",
                            alpha=0.7, s=40)
            plt.colorbar(sc, ax=ax, label=color_var)

        ax.set_xlabel(xlbl, fontsize=11)
        ax.set_ylabel("log(1 + Muertes/Millón)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.subheader("Correlación de Spearman — Variables Estructurales vs Mortalidad")
    corr_cols = ["total_deaths_per_million", "median_age", "aged_65_older",
                 "gdp_per_capita", "hospital_beds_per_thousand",
                 "life_expectancy", "extreme_poverty",
                 "cardiovasc_death_rate", "diabetes_prevalence"]
    corr_mat = snap_f[corr_cols].dropna().corr(method="spearman")
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="YlGn",
                center=0, vmin=-0.7, vmax=0.7, ax=ax2, linewidths=0.5)
    ax2.set_title("Correlación Spearman: Mortalidad vs Indicadores Estructurales", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.info("""
**Interpretación:** La relación no es lineal: Europa tiene muchas camas hospitaliarias
pero también poblaciones envejecidas. La matriz de Spearman confirma que `median_age`
y `life_expectancy` son los correlatos más fuertes de la mortalidad proporcional.
""")

# =============================================================
# PAGE 3 — Q3: COMPARATIVA POR PAÍS
# =============================================================
elif page == "🌍 Q3: Comparativa por País":
    st.title("Q3: Análisis Comparativo de Mortalidad por País y Continente")
    st.markdown("""
Identifica los países con mayor carga de enfermedad y descubre grupos de países
con perfiles epidemiológicos similares mediante **K-Means clustering**.
""")

    col1, col2 = st.columns([2, 1])
    with col2:
        top_n = st.slider("Top N países:", 5, 30, 15, 5)
        metric_choice = st.selectbox(
            "Métrica:",
            ["total_deaths_per_million", "total_cases_per_million", "cfr"],
            format_func=lambda x: {
                "total_deaths_per_million": "Muertes por Millón",
                "total_cases_per_million": "Casos por Millón",
                "cfr": "Tasa de Letalidad (%)"
            }[x]
        )
    metric_lbl = {"total_deaths_per_million": "Muertes por Millón",
                  "total_cases_per_million": "Casos por Millón",
                  "cfr": "Tasa de Letalidad (%)"}[metric_choice]

    top_data = (snap_f.dropna(subset=[metric_choice])
                .nlargest(top_n, metric_choice)
                [["location", "continent", metric_choice]])

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(top_data["location"], top_data[metric_choice],
            color=[cont_palette.get(c, "#888") for c in top_data["continent"]],
            edgecolor="white")
    ax.set_xlabel(metric_lbl, fontsize=11)
    ax.set_title(f"Top {top_n} Países — {metric_lbl}", fontsize=12)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=cont_palette.get(c, "#888"))
               for c in selected_continents if c in cont_palette]
    ax.legend(handles, [c for c in selected_continents if c in cont_palette],
              fontsize=8, loc="lower right")
    plt.tight_layout()
    with col1:
        st.pyplot(fig)
    plt.close()

    # Boxplot por continente
    st.subheader("Distribución de Muertes por Millón por Continente")
    cont_df = snap_f.dropna(subset=["total_deaths_per_million"])
    cont_order = (cont_df.groupby("continent")["total_deaths_per_million"]
                  .median().sort_values(ascending=False).index.tolist())
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cont_df, x="continent", y="total_deaths_per_million",
                order=cont_order,
                palette=CONT_COLORS[:len(cont_order)], ax=ax2, fliersize=3)
    ax2.set_xlabel("")
    ax2.set_ylabel("Muertes por Millón")
    ax2.set_title("Muertes por Millón por Continente", fontsize=12)
    ax2.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # K-Means
    st.subheader("K-Means: Grupos de Países por Perfil Epidemiológico")
    cluster_cols = ["total_deaths_per_million", "median_age",
                    "gdp_per_capita", "hospital_beds_per_thousand"]
    cluster_df = snap_f.dropna(subset=cluster_cols).copy()
    k_val = st.slider("Número de clusters (k):", 2, 6, 3, 1)

    scaler = StandardScaler()
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    cluster_df["cluster"] = km.fit_predict(scaler.fit_transform(cluster_df[cluster_cols]))

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for cl in range(k_val):
        sub = cluster_df[cluster_df["cluster"] == cl]
        ax3.scatter(sub["median_age"], sub["log_deaths_pm"],
                    label=f"Cluster {cl} (n={len(sub)})",
                    alpha=0.75, s=50,
                    color=CONT_COLORS[cl % len(CONT_COLORS)])
    ax3.set_xlabel("Edad Mediana", fontsize=11)
    ax3.set_ylabel("log(1 + Muertes/Millón)", fontsize=11)
    ax3.set_title("K-Means: Agrupación de Países", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    with st.expander("Ver medias por cluster"):
        st.dataframe(cluster_df.groupby("cluster")[cluster_cols].mean().round(1),
                     use_container_width=True)

    st.info("""
**Interpretación:** Los clusters no se alinean perfectamente con continentes —
el envejecimiento y la riqueza combinados explican mejor los grupos que la geografía sola.
Europa occidental y América del Norte forman clusters de alta mortalidad con poblaciones
envejecidas; África subsahariana agrupa países de baja mortalidad reportada y menor edad mediana.
""")

# =============================================================
# PAGE 4 — Q4: PROPAGACIÓN TEMPORAL
# =============================================================
elif page == "📈 Q4: Propagación Temporal":
    st.title("Q4: Dinámica de Propagación Temporal")
    st.markdown("""
Evolución de nuevos casos y muertes a nivel global y por país,
identificando las distintas oleadas epidémicas.
""")

    available = sorted(
        df[df["continent"].isin(selected_continents)]["location"].unique().tolist()
    )
    defaults = [c for c in
                ["United States", "Brazil", "India", "Germany", "Colombia", "Peru"]
                if c in available][:5]

    selected_countries = st.multiselect(
        "Seleccionar países:", options=available, default=defaults
    )

    if selected_countries:
        ts = df[df["location"].isin(selected_countries)].copy()
        cmap = plt.cm.get_cmap("tab10", len(selected_countries))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, country in enumerate(selected_countries):
            sub = ts[ts["location"] == country]
            s1 = sub.dropna(subset=["new_cases_smoothed_per_million"])
            s2 = sub.dropna(subset=["new_deaths_smoothed_per_million"])
            axes[0].plot(s1["date"], s1["new_cases_smoothed_per_million"],
                         label=country, lw=1.8, color=cmap(i), alpha=0.85)
            axes[1].plot(s2["date"], s2["new_deaths_smoothed_per_million"],
                         label=country, lw=1.8, color=cmap(i), alpha=0.85)

        axes[0].set_title("Nuevos Casos Suavizados por Millón (7 días)", fontsize=12)
        axes[0].set_ylabel("Casos / Millón")
        axes[0].legend(fontsize=9, ncol=3)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Nuevas Muertes Suavizadas por Millón (7 días)", fontsize=12)
        axes[1].set_ylabel("Muertes / Millón")
        axes[1].set_xlabel("Fecha")
        axes[1].legend(fontsize=9, ncol=3)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Selecciona al menos un país.")

    # Evolución por continente
    st.subheader("Evolución Mensual de Nuevos Casos por Continente (millones)")
    monthly = (
        df[df["continent"].isin(selected_continents)]
        .assign(ym=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
        .dropna(subset=["new_cases"])
        .groupby(["ym", "continent"])["new_cases"].sum()
        .reset_index()
    )
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for cont in monthly["continent"].unique():
        sub = monthly[monthly["continent"] == cont]
        ax2.plot(sub["ym"], sub["new_cases"] / 1e6, label=cont,
                 lw=1.8, color=cont_palette.get(cont, "#888"))
    ax2.set_ylabel("Nuevos Casos (millones)")
    ax2.set_xlabel("Fecha")
    ax2.set_title("Nuevos Casos Mensuales por Continente", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.info("""
**Interpretación:** Las series temporales revelan claramente las oleadas epidémicas.
Ómicron (finales 2021 – inicios 2022) generó el mayor pico de casos globales.
La mortalidad sigue con rezago a los nuevos casos, patrón esperado dado el tiempo
entre infección y desenlace clínico.
""")

# =============================================================
# PAGE 5 — FUENTE DE DATOS
# =============================================================
elif page == "📂 Fuente de Datos":
    st.title("📂 Documentación de la Fuente de Datos")
    st.markdown(f"""
## Dataset: Our World in Data — COVID-19

| Campo | Detalle |
|-------|---------|
| **Nombre** | Our World in Data COVID-19 Dataset |
| **URL** | https://github.com/owid/covid-19-data |
| **Archivo** | `owid-covid-data.csv` |
| **Fecha de acceso** | Febrero 2026 |
| **Granularidad** | Registro diario por país |
| **Última fecha disponible** | {df["date"].max().date()} |
| **Fuentes originales** | Johns Hopkins CSSE, CDC, ECDC, OMS |
| **Licencia** | CC BY 4.0 |

---

## Cómo Actualizar el Dashboard

**Descarga manual del CSV:**
```bash
curl -L https://covid.ourworldindata.org/data/owid-covid-data.csv -o owid-covid-data.csv
```

**Script Python:**
```python
import urllib.request
urllib.request.urlretrieve(
    "https://covid.ourworldindata.org/data/owid-covid-data.csv",
    "owid-covid-data.csv"
)
```

Reemplaza el archivo `owid-covid-data.csv` en el repositorio de GitHub y
Streamlit Cloud se actualizará automáticamente en el siguiente redeploy.

---

## Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub (incluir `STREAMLIT_PYTHONCODE.py`,
   `owid-covid-data.csv` y `requirements.txt`).
2. Ir a [share.streamlit.io](https://share.streamlit.io) → iniciar sesión con GitHub.
3. Seleccionar repo → archivo principal `STREAMLIT_PYTHONCODE.py` → **Deploy**.
4. La URL pública queda disponible en minutos.

---

## Limitaciones

- Los **casos confirmados** no son directamente comparables entre países con
  distinta capacidad de testeo. Las **muertes por millón** son más robustas.
- Los indicadores estructurales (`median_age`, `gdp_per_capita`) son pre-pandemia
  y se asumen constantes.
- Los agregados regionales de OWID (continentes, grupos de ingreso) fueron
  excluidos del análisis país-a-país.
""")
    st.success(f"✅ Dataset cargado correctamente. "
               f"Última fecha disponible: **{df['date'].max().date()}**")



