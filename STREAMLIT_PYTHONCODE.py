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

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2C5F2D;
        box-shadow: 0 4px 15px rgba(44, 95, 45, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #97BC62;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 5px;
    }
    .sidebar-section {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 10px;
        margin: 8px 0;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2C5F2D;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────
C1 = "#2C5F2D"
C2 = "#97BC62"
C3 = "#B8860B"
C4 = "#8B3A3A"
CONT_COLORS = ["#2C5F2D", "#97BC62", "#B8860B", "#8B3A3A", "#5B7FA6", "#7B5EA7"]

# ── Load data ─────────────────────────────────────────────────
OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    local_path = base_dir / "owid-covid-data.csv"

    if local_path.exists():
        df = pd.read_csv(local_path, parse_dates=["date"])
    else:
        df = pd.read_csv(OWID_URL, parse_dates=["date"])

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
st.sidebar.title("COVID-19 Dashboard")
st.sidebar.markdown("**Marco QUEST — OWID Dataset**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegar a:",
    [
        "Introduccion",
        "Q1: Demografia y Mortalidad",
        "Q2: Capacidad Sanitaria",
        "Q3: Comparativa por Pais",
        "Q4: Propagacion Temporal",
        "Fuente de Datos",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Globales")

# Continent filter with select all / none
col_all, col_none = st.sidebar.columns(2)
if col_all.button("Todos", use_container_width=True):
    st.session_state["continents"] = CONTINENTS
if col_none.button("Ninguno", use_container_width=True):
    st.session_state["continents"] = []

if "continents" not in st.session_state:
    st.session_state["continents"] = CONTINENTS

selected_continents = st.sidebar.multiselect(
    "Continentes:", options=CONTINENTS, default=st.session_state["continents"],
    key="continent_filter"
)

min_pop = st.sidebar.slider(
    "Poblacion minima (millones):", 0, 100, 1, 1
)

min_cases = st.sidebar.number_input(
    "Casos minimos confirmados:", min_value=0, value=0, step=1000
)

snap_f = snap[
    snap["continent"].isin(selected_continents) &
    (snap["population"] >= min_pop * 1e6) &
    (snap["total_cases"].fillna(0) >= min_cases)
].copy()

cont_palette = dict(zip(CONTINENTS, CONT_COLORS))
st.sidebar.markdown("---")
st.sidebar.caption(f"Paises en vista: **{len(snap_f)}** de {len(snap)}")

# Live global counters in sidebar
total_cases_global = snap["total_cases"].sum()
total_deaths_global = snap["total_deaths"].sum()
st.sidebar.markdown("**Totales globales (acumulado)**")
st.sidebar.metric("Casos totales", f"{total_cases_global/1e6:.1f}M")
st.sidebar.metric("Muertes totales", f"{total_deaths_global/1e6:.2f}M")
st.sidebar.metric("CFR global", f"{(total_deaths_global/total_cases_global*100):.2f}%")

# =============================================================
# PAGE 0 — INTRODUCCION
# =============================================================
if page == "Introduccion":
    st.title("Analisis Exploratorio de Datos: COVID-19 (OWID)")

    st.markdown("""
## Marco QUEST aplicado al dataset global de COVID-19

Este dashboard presenta el EDA del dataset **Our World in Data (OWID)** sobre COVID-19,
con estadisticas diarias por pais: casos, muertes, y mas de 60 indicadores socioeconómicos
y de salud publica.
""")

    # Animated-style KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paises analizados", len(snap), help="Total de paises con datos validos")
    c2.metric("Registros totales", f"{len(df):,}", help="Filas en el dataset")
    c3.metric("Fecha inicio", str(df["date"].min().date()))
    c4.metric("Fecha fin", str(df["date"].max().date()))

    st.markdown("---")

    # Interactive table of questions
    st.subheader("Preguntas Analiticas")
    questions = {
        "Q1 — Tiene la estructura demografica de un pais influencia en su tasa de mortalidad?": "Q1: Demografia y Mortalidad",
        "Q2 — Reduce la capacidad hospitalaria la mortalidad proporcional?": "Q2: Capacidad Sanitaria",
        "Q3 — Que paises y continentes presentan mayor carga de enfermedad?": "Q3: Comparativa por Pais",
        "Q4 — Que patrones temporales se observan en la propagacion del virus?": "Q4: Propagacion Temporal",
    }
    for q, dest in questions.items():
        with st.expander(q):
            st.markdown(f"Navega a **{dest}** en el menu lateral para explorar esta pregunta.")

    st.markdown("---")

    # Top 5 snapshot cards
    st.subheader("Ranking Rapido")
    tab1, tab2, tab3 = st.tabs(["Mas muertes/millon", "Mas casos/millon", "Mayor CFR"])
    with tab1:
        top5d = snap.dropna(subset=["total_deaths_per_million"]).nlargest(5, "total_deaths_per_million")[["location", "continent", "total_deaths_per_million"]]
        st.dataframe(top5d.reset_index(drop=True), use_container_width=True)
    with tab2:
        top5c = snap.dropna(subset=["total_cases_per_million"]).nlargest(5, "total_cases_per_million")[["location", "continent", "total_cases_per_million"]]
        st.dataframe(top5c.reset_index(drop=True), use_container_width=True)
    with tab3:
        top5cfr = snap.dropna(subset=["cfr"]).nlargest(5, "cfr")[["location", "continent", "cfr"]]
        st.dataframe(top5cfr.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("Vista previa del dataset")

    # Interactive column selector
    all_cols = df.columns.tolist()
    default_cols = ["location", "continent", "date", "total_cases", "total_deaths",
                    "total_cases_per_million", "total_deaths_per_million"]
    chosen_cols = st.multiselect("Seleccionar columnas a mostrar:", options=all_cols, default=default_cols)
    n_rows = st.slider("Filas a mostrar:", 5, 50, 10)
    st.dataframe(df[chosen_cols].head(n_rows), use_container_width=True)

    # Download button
    csv = df[chosen_cols].head(n_rows).to_csv(index=False).encode("utf-8")
    st.download_button("Descargar seleccion como CSV", data=csv,
                       file_name="covid_preview.csv", mime="text/csv")

# =============================================================
# PAGE 1 — Q1: DEMOGRAFIA Y MORTALIDAD
# =============================================================
elif page == "Q1: Demografia y Mortalidad":
    st.title("Q1: Impacta la demografia en la mortalidad por COVID-19?")
    st.markdown("""
**Hipotesis:** Los paises con poblaciones mas envejecidas (`median_age`, `aged_65_older`)
presentan tasas de mortalidad por COVID-19 mas altas por millon de habitantes.
""")

    demo_df = snap_f.dropna(
        subset=["median_age", "aged_65_older", "total_deaths_per_million"]
    ).copy()

    # Controls row
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        log_y = st.toggle("Escala logaritmica (Y)", value=True)
    with ctrl2:
        show_trend = st.toggle("Linea de tendencia", value=True)
    with ctrl3:
        show_labels = st.toggle("Mostrar nombres de paises", value=False)
    with ctrl4:
        size_by = st.selectbox("Tamanio de puntos por:", ["Fijo", "population", "total_cases"])

    y_col = "log_deaths_pm" if log_y else "total_deaths_per_million"
    y_lbl = "log(1 + Muertes/Millon)" if log_y else "Muertes por Millon"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    for cont in demo_df["continent"].unique():
        sub = demo_df[demo_df["continent"] == cont]
        sz = 40
        if size_by == "population":
            sz = (sub["population"].fillna(1e6) / 5e6).clip(10, 300)
        elif size_by == "total_cases":
            sz = (sub["total_cases"].fillna(1e4) / 5e5).clip(10, 300)

        axes[0].scatter(sub["median_age"], sub[y_col],
                        label=cont, alpha=0.8, s=sz, color=cont_palette.get(cont))
        axes[1].scatter(sub["aged_65_older"], sub[y_col],
                        label=cont, alpha=0.8, s=sz, color=cont_palette.get(cont))

        if show_labels:
            for _, row in sub.iterrows():
                axes[0].annotate(row["location"], (row["median_age"], row[y_col]),
                                 fontsize=5, color="white", alpha=0.7)
                axes[1].annotate(row["location"], (row["aged_65_older"], row[y_col]),
                                 fontsize=5, color="white", alpha=0.7)

    for ax, xcol, xlbl in zip(
        axes,
        ["median_age", "aged_65_older"],
        ["Edad Mediana", "Mayores de 65 (%)"]
    ):
        if show_trend:
            vd = demo_df[[xcol, y_col]].dropna()
            z = np.polyfit(vd[xcol], vd[y_col], 1)
            xs = np.linspace(vd[xcol].min(), vd[xcol].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), "--", lw=1.5, color="#FFD700", label="Tendencia")
        ax.set_xlabel(xlbl, fontsize=11)
        ax.set_ylabel(y_lbl, fontsize=11)
        ax.grid(True, alpha=0.15, color="white")

    axes[0].set_title("Edad Mediana vs Mortalidad", fontsize=12)
    axes[1].set_title("% Mayores 65 vs Mortalidad", fontsize=12)
    axes[0].legend(fontsize=7, ncol=2, facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlation sliders — interactive Pearson r display
    st.subheader("Correlacion en tiempo real")
    col_x, col_y2 = st.columns(2)
    with col_x:
        x_var = st.selectbox("Variable X:", ["median_age", "aged_65_older",
                                             "life_expectancy", "gdp_per_capita",
                                             "extreme_poverty", "diabetes_prevalence",
                                             "cardiovasc_death_rate"])
    with col_y2:
        y_var = st.selectbox("Variable Y:", ["total_deaths_per_million",
                                             "total_cases_per_million", "cfr"],
                             format_func=lambda v: {
                                 "total_deaths_per_million": "Muertes/Millon",
                                 "total_cases_per_million": "Casos/Millon",
                                 "cfr": "CFR (%)"
                             }[v])

    corr_data = snap_f[[x_var, y_var]].dropna()
    if len(corr_data) > 5:
        r = corr_data.corr(method="pearson").iloc[0, 1]
        rho = corr_data.corr(method="spearman").iloc[0, 1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson r", f"{r:.3f}")
        m2.metric("Spearman rho", f"{rho:.3f}")
        m3.metric("N paises", len(corr_data))

        fig_c, ax_c = plt.subplots(figsize=(7, 4), facecolor="#0e1117")
        ax_c.set_facecolor("#0e1117")
        ax_c.scatter(corr_data[x_var], corr_data[y_var], alpha=0.6, color=C2, s=40)
        z = np.polyfit(corr_data[x_var], corr_data[y_var], 1)
        xs = np.linspace(corr_data[x_var].min(), corr_data[x_var].max(), 100)
        ax_c.plot(xs, np.poly1d(z)(xs), "--", color="#FFD700", lw=2)
        ax_c.set_xlabel(x_var, color="white")
        ax_c.set_ylabel(y_var, color="white")
        ax_c.tick_params(colors="white")
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#444")
        ax_c.grid(True, alpha=0.15, color="white")
        ax_c.set_title(f"{x_var} vs {y_var}  |  r = {r:.3f}", color="white")
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close()

    st.info("""
**Interpretacion:** Se observa correlacion positiva entre envejecimiento poblacional
y mortalidad proporcional. Los paises europeos concentran los valores mas altos,
consistente con que la edad es el mayor factor de riesgo individual para mortalidad
por SARS-CoV-2.
""")

# =============================================================
# PAGE 2 — Q2: CAPACIDAD SANITARIA
# =============================================================
elif page == "Q2: Capacidad Sanitaria":
    st.title("Q2: Protege la capacidad hospitalaria contra la mortalidad?")
    st.markdown("""
**Hipotesis:** Paises con mayor `hospital_beds_per_thousand` y `gdp_per_capita`
presentaran menor mortalidad proporcional por COVID-19.
""")

    health_df = snap_f.dropna(
        subset=["hospital_beds_per_thousand", "gdp_per_capita", "total_deaths_per_million"]
    ).copy()

    tab_scatter, tab_heatmap, tab_rank = st.tabs(["Dispersion", "Correlacion Spearman", "Ranking por indicador"])

    with tab_scatter:
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            color_var = st.selectbox(
                "Colorear puntos por:",
                ["continent", "median_age", "aged_65_older"],
                format_func=lambda x: {"continent": "Continente",
                                       "median_age": "Edad Mediana",
                                       "aged_65_older": "% Mayores 65"}[x]
            )
        with col_ctrl2:
            show_annotations = st.toggle("Mostrar paises extremos", value=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0e1117")
        for ax in axes:
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        for ax, xcol, xlbl, title in zip(
            axes,
            ["hospital_beds_per_thousand", "log_gdp"],
            ["Camas Hosp. / 1,000 hab.", "log10(PIB per capita)"],
            ["Infraestructura Hospitalaria vs Mortalidad",
             "Riqueza Nacional vs Mortalidad"]
        ):
            if color_var == "continent":
                for cont in health_df["continent"].unique():
                    sub = health_df[health_df["continent"] == cont]
                    ax.scatter(sub[xcol], sub["log_deaths_pm"],
                               label=cont, alpha=0.75, s=45,
                               color=cont_palette.get(cont))
                ax.legend(fontsize=7, ncol=2, facecolor="#1a1a2e", labelcolor="white")
            else:
                sc = ax.scatter(health_df[xcol], health_df["log_deaths_pm"],
                                c=health_df[color_var], cmap="YlGn",
                                alpha=0.75, s=45)
                plt.colorbar(sc, ax=ax, label=color_var)

            if show_annotations:
                top3 = health_df.nlargest(3, "log_deaths_pm")
                for _, row in top3.iterrows():
                    ax.annotate(row["location"], (row[xcol], row["log_deaths_pm"]),
                                fontsize=7, color="#FFD700", alpha=0.9)

            ax.set_xlabel(xlbl, fontsize=11, color="white")
            ax.set_ylabel("log(1 + Muertes/Millon)", fontsize=11, color="white")
            ax.set_title(title, fontsize=11, color="white")
            ax.grid(True, alpha=0.15, color="white")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_heatmap:
        st.subheader("Correlacion de Spearman — Variables Estructurales vs Mortalidad")
        corr_cols_options = {
            "Muertes/Millon": "total_deaths_per_million",
            "Edad Mediana": "median_age",
            "Mayores 65": "aged_65_older",
            "PIB per capita": "gdp_per_capita",
            "Camas Hosp.": "hospital_beds_per_thousand",
            "Esperanza de vida": "life_expectancy",
            "Pobreza extrema": "extreme_poverty",
            "Cardiovascular": "cardiovasc_death_rate",
            "Diabetes": "diabetes_prevalence",
            "Vacunados (%)": "people_fully_vaccinated_per_hundred",
        }
        selected_corr = st.multiselect(
            "Variables para la matriz:",
            options=list(corr_cols_options.keys()),
            default=list(corr_cols_options.keys())[:8]
        )
        if len(selected_corr) > 1:
            corr_cols = [corr_cols_options[k] for k in selected_corr]
            corr_mat = snap_f[corr_cols].dropna().corr(method="spearman")
            corr_mat.index = selected_corr
            corr_mat.columns = selected_corr
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
            ax2.set_facecolor("#0e1117")
            sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                        center=0, vmin=-1, vmax=1, ax=ax2, linewidths=0.5,
                        annot_kws={"size": 9})
            ax2.tick_params(colors="white")
            ax2.set_title("Correlacion Spearman", fontsize=12, color="white")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    with tab_rank:
        rank_var = st.selectbox("Ordenar paises por:", list(corr_cols_options.values()),
                                format_func=lambda v: [k for k, val in corr_cols_options.items() if val == v][0])
        n_rank = st.slider("Cantidad de paises:", 5, 30, 15)
        ascending = st.toggle("Orden ascendente", value=False)
        rank_df = snap_f.dropna(subset=[rank_var]).sort_values(rank_var, ascending=ascending).head(n_rank)
        fig3, ax3 = plt.subplots(figsize=(9, max(4, n_rank * 0.38)), facecolor="#0e1117")
        ax3.set_facecolor("#0e1117")
        bars = ax3.barh(rank_df["location"], rank_df[rank_var],
                        color=[cont_palette.get(c, "#888") for c in rank_df["continent"]],
                        edgecolor="#0e1117")
        ax3.tick_params(colors="white")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#444")
        ax3.set_xlabel(rank_var, color="white")
        ax3.set_title(f"Ranking por {rank_var}", color="white")
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.15, color="white", axis="x")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        csv_r = rank_df[["location", "continent", rank_var]].to_csv(index=False).encode("utf-8")
        st.download_button("Descargar ranking CSV", data=csv_r,
                           file_name=f"ranking_{rank_var}.csv", mime="text/csv")

    st.info("""
**Interpretacion:** La relacion no es lineal: Europa tiene muchas camas hospitalarias
pero tambien poblaciones envejecidas. La matriz de Spearman confirma que `median_age`
y `life_expectancy` son los correlatos mas fuertes de la mortalidad proporcional.
""")

# =============================================================
# PAGE 3 — Q3: COMPARATIVA POR PAIS
# =============================================================
elif page == "Q3: Comparativa por Pais":
    st.title("Q3: Analisis Comparativo de Mortalidad por Pais y Continente")
    st.markdown("""
Identifica los paises con mayor carga de enfermedad y descubre grupos de paises
con perfiles epidemiologicos similares mediante **K-Means clustering**.
""")

    tab_bar, tab_box, tab_cluster, tab_search = st.tabs(
        ["Top Paises", "Distribucion Continentes", "K-Means Clustering", "Buscar Pais"]
    )

    with tab_bar:
        col1, col2 = st.columns([2, 1])
        with col2:
            top_n = st.slider("Top N paises:", 5, 30, 15, 5, key="topn_bar")
            metric_choice = st.selectbox(
                "Metrica:",
                ["total_deaths_per_million", "total_cases_per_million", "cfr"],
                format_func=lambda x: {
                    "total_deaths_per_million": "Muertes por Millon",
                    "total_cases_per_million": "Casos por Millon",
                    "cfr": "Tasa de Letalidad (%)"
                }[x]
            )
            show_bottom = st.toggle("Mostrar bottom N tambien", value=False)

        metric_lbl = {"total_deaths_per_million": "Muertes por Millon",
                      "total_cases_per_million": "Casos por Millon",
                      "cfr": "Tasa de Letalidad (%)"}[metric_choice]

        top_data = (snap_f.dropna(subset=[metric_choice])
                    .nlargest(top_n, metric_choice)
                    [["location", "continent", metric_choice]])

        if show_bottom:
            bottom_data = (snap_f.dropna(subset=[metric_choice])
                           .nsmallest(top_n, metric_choice)
                           [["location", "continent", metric_choice]])
            fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.38)),
                                                  facecolor="#0e1117")
            axes_list = [ax_top, ax_bot]
            data_list = [top_data, bottom_data]
            titles = [f"Top {top_n} — {metric_lbl}", f"Bottom {top_n} — {metric_lbl}"]
        else:
            fig, ax_top = plt.subplots(figsize=(9, max(5, top_n * 0.38)), facecolor="#0e1117")
            axes_list = [ax_top]
            data_list = [top_data]
            titles = [f"Top {top_n} — {metric_lbl}"]

        for ax, data, title in zip(axes_list, data_list, titles):
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.barh(data["location"], data[metric_choice],
                    color=[cont_palette.get(c, "#888") for c in data["continent"]],
                    edgecolor="#0e1117")
            ax.set_xlabel(metric_lbl, fontsize=10, color="white")
            ax.set_title(title, fontsize=11, color="white")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.15, color="white", axis="x")

        plt.tight_layout()
        with col1:
            st.pyplot(fig)
        plt.close()

        csv_top = top_data.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar top paises CSV", data=csv_top,
                           file_name="top_paises.csv", mime="text/csv")

    with tab_box:
        metric_box = st.selectbox("Metrica distribucion:",
                                  ["total_deaths_per_million", "total_cases_per_million", "cfr"],
                                  format_func=lambda x: {
                                      "total_deaths_per_million": "Muertes por Millon",
                                      "total_cases_per_million": "Casos por Millon",
                                      "cfr": "CFR (%)"
                                  }[x], key="metric_box")
        show_violin = st.toggle("Violin en lugar de boxplot", value=False)

        cont_df = snap_f.dropna(subset=[metric_box])
        cont_order = (cont_df.groupby("continent")[metric_box]
                      .median().sort_values(ascending=False).index.tolist())

        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
        ax2.set_facecolor("#0e1117")
        ax2.tick_params(colors="white", axis="both")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")

        if show_violin:
            sns.violinplot(data=cont_df, x="continent", y=metric_box,
                           order=cont_order, palette=CONT_COLORS[:len(cont_order)],
                           ax=ax2, inner="quartile")
        else:
            sns.boxplot(data=cont_df, x="continent", y=metric_box,
                        order=cont_order, palette=CONT_COLORS[:len(cont_order)],
                        ax=ax2, fliersize=3)

        ax2.set_xlabel("", color="white")
        ax2.set_ylabel(metric_box, color="white")
        ax2.set_title(f"{metric_box} por Continente", fontsize=12, color="white")
        ax2.tick_params(axis="x", rotation=15, colors="white")
        ax2.grid(True, alpha=0.15, color="white", axis="y")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab_cluster:
        cluster_cols_options = {
            "Muertes/Millon": "total_deaths_per_million",
            "Edad Mediana": "median_age",
            "PIB per capita": "gdp_per_capita",
            "Camas Hosp.": "hospital_beds_per_thousand",
            "Casos/Millon": "total_cases_per_million",
            "Esperanza vida": "life_expectancy",
        }
        sel_cluster_vars = st.multiselect(
            "Variables para clustering:",
            options=list(cluster_cols_options.keys()),
            default=["Muertes/Millon", "Edad Mediana", "PIB per capita", "Camas Hosp."]
        )
        if len(sel_cluster_vars) < 2:
            st.warning("Selecciona al menos 2 variables.")
        else:
            cluster_cols = [cluster_cols_options[k] for k in sel_cluster_vars]
            cluster_df = snap_f.dropna(subset=cluster_cols).copy()

            k_val = st.slider("Numero de clusters (k):", 2, 6, 3, 1)

            x_axis_raw = st.selectbox("Eje X del grafico:", sel_cluster_vars, index=1)
            y_axis_raw = st.selectbox("Eje Y del grafico:", sel_cluster_vars, index=0)
            x_axis = cluster_cols_options[x_axis_raw]
            y_axis = cluster_cols_options[y_axis_raw]

            scaler = StandardScaler()
            km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            cluster_df["cluster"] = km.fit_predict(scaler.fit_transform(cluster_df[cluster_cols]))

            fig3, ax3 = plt.subplots(figsize=(9, 5), facecolor="#0e1117")
            ax3.set_facecolor("#0e1117")
            ax3.tick_params(colors="white")
            for spine in ax3.spines.values():
                spine.set_edgecolor("#444")

            for cl in range(k_val):
                sub = cluster_df[cluster_df["cluster"] == cl]
                ax3.scatter(sub[x_axis], sub[y_axis],
                            label=f"Cluster {cl} (n={len(sub)})",
                            alpha=0.8, s=55,
                            color=CONT_COLORS[cl % len(CONT_COLORS)])

            ax3.set_xlabel(x_axis_raw, fontsize=11, color="white")
            ax3.set_ylabel(y_axis_raw, fontsize=11, color="white")
            ax3.set_title("K-Means: Agrupacion de Paises", fontsize=12, color="white")
            ax3.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
            ax3.grid(True, alpha=0.15, color="white")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            with st.expander("Ver medias por cluster"):
                st.dataframe(cluster_df.groupby("cluster")[cluster_cols].mean().round(1),
                             use_container_width=True)

            with st.expander("Ver paises por cluster"):
                for cl in range(k_val):
                    countries_in = cluster_df[cluster_df["cluster"] == cl]["location"].sort_values().tolist()
                    st.markdown(f"**Cluster {cl}:** {', '.join(countries_in)}")

    with tab_search:
        st.subheader("Ficha por pais")
        all_countries = sorted(snap["location"].unique().tolist())
        selected_country = st.selectbox("Seleccionar pais:", all_countries)
        compare_countries = st.multiselect("Comparar con:", all_countries,
                                           default=[], max_selections=4)

        profile_cols = ["total_cases_per_million", "total_deaths_per_million",
                        "cfr", "median_age", "gdp_per_capita",
                        "hospital_beds_per_thousand", "life_expectancy",
                        "people_fully_vaccinated_per_hundred"]
        profile_labels = ["Casos/Millon", "Muertes/Millon", "CFR (%)",
                          "Edad Mediana", "PIB per capita",
                          "Camas/1000", "Esp. Vida", "Vacunados (%)"]

        countries_to_show = [selected_country] + compare_countries
        profile_df = snap[snap["location"].isin(countries_to_show)][
            ["location"] + profile_cols
        ].set_index("location")
        profile_df.columns = profile_labels

        st.dataframe(profile_df.T, use_container_width=True)

        # Mini time series for the searched country
        country_ts = df[df["location"] == selected_country].dropna(
            subset=["new_cases_smoothed_per_million"]
        )
        if not country_ts.empty:
            fig_ts, ax_ts = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
            ax_ts.set_facecolor("#0e1117")
            ax_ts.fill_between(country_ts["date"],
                               country_ts["new_cases_smoothed_per_million"],
                               alpha=0.4, color=C2)
            ax_ts.plot(country_ts["date"],
                       country_ts["new_cases_smoothed_per_million"],
                       color=C2, lw=1.5)
            ax_ts.tick_params(colors="white")
            for spine in ax_ts.spines.values():
                spine.set_edgecolor("#444")
            ax_ts.set_title(f"Nuevos casos/millon — {selected_country}",
                            color="white", fontsize=11)
            ax_ts.grid(True, alpha=0.15, color="white")
            plt.tight_layout()
            st.pyplot(fig_ts)
            plt.close()

    st.info("""
**Interpretacion:** Los clusters no se alinean perfectamente con continentes —
el envejecimiento y la riqueza combinados explican mejor los grupos que la geografia sola.
""")

# =============================================================
# PAGE 4 — Q4: PROPAGACION TEMPORAL
# =============================================================
elif page == "Q4: Propagacion Temporal":
    st.title("Q4: Dinamica de Propagacion Temporal")
    st.markdown("""
Evolucion de nuevos casos y muertes a nivel global y por pais,
identificando las distintas oleadas epidemicas.
""")

    tab_countries, tab_continents, tab_heatmap_t = st.tabs(
        ["Por pais", "Por continente", "Heatmap temporal"]
    )

    available = sorted(
        df[df["continent"].isin(selected_continents)]["location"].unique().tolist()
    )

    with tab_countries:
        defaults = [c for c in
                    ["United States", "Brazil", "India", "Germany", "Colombia", "Peru"]
                    if c in available][:5]
        selected_countries = st.multiselect(
            "Seleccionar paises:", options=available, default=defaults
        )

        metric_ts = st.selectbox("Metrica temporal:",
                                 ["new_cases_smoothed_per_million",
                                  "new_deaths_smoothed_per_million",
                                  "new_vaccinations_smoothed_per_million"],
                                 format_func=lambda v: {
                                     "new_cases_smoothed_per_million": "Nuevos Casos/Millon (7d)",
                                     "new_deaths_smoothed_per_million": "Nuevas Muertes/Millon (7d)",
                                     "new_vaccinations_smoothed_per_million": "Nuevas Vacunaciones/Millon (7d)"
                                 }[v])

        date_range = st.date_input(
            "Rango de fechas:",
            value=[df["date"].min().date(), df["date"].max().date()],
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date()
        )

        if selected_countries and len(date_range) == 2:
            ts = df[
                df["location"].isin(selected_countries) &
                (df["date"].dt.date >= date_range[0]) &
                (df["date"].dt.date <= date_range[1])
            ].copy()

            show_both = st.toggle("Mostrar casos Y muertes simultaneamente", value=False)
            cmap_ts = plt.cm.get_cmap("tab10", len(selected_countries))

            if show_both:
                fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, facecolor="#0e1117")
                pairs = [
                    ("new_cases_smoothed_per_million", "Casos / Millon"),
                    ("new_deaths_smoothed_per_million", "Muertes / Millon"),
                ]
                for ax, (col, ylabel) in zip(axes, pairs):
                    ax.set_facecolor("#0e1117")
                    ax.tick_params(colors="white")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#444")
                    for i, country in enumerate(selected_countries):
                        sub = ts[ts["location"] == country].dropna(subset=[col])
                        ax.plot(sub["date"], sub[col], label=country,
                                lw=1.8, color=cmap_ts(i), alpha=0.85)
                    ax.set_ylabel(ylabel, color="white")
                    ax.legend(fontsize=9, ncol=3, facecolor="#1a1a2e", labelcolor="white")
                    ax.grid(True, alpha=0.15, color="white")
                axes[0].set_title("Casos y Muertes Suavizados", color="white", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0e1117")
                ax.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")
                for i, country in enumerate(selected_countries):
                    sub = ts[ts["location"] == country].dropna(subset=[metric_ts])
                    ax.fill_between(sub["date"], sub[metric_ts],
                                    alpha=0.12, color=cmap_ts(i))
                    ax.plot(sub["date"], sub[metric_ts], label=country,
                            lw=1.8, color=cmap_ts(i), alpha=0.9)
                ax.set_ylabel(metric_ts, color="white")
                ax.set_xlabel("Fecha", color="white")
                ax.legend(fontsize=9, ncol=3, facecolor="#1a1a2e", labelcolor="white")
                ax.grid(True, alpha=0.15, color="white")
                ax.set_title(metric_ts, color="white", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("Selecciona al menos un pais y un rango de fechas valido.")

    with tab_continents:
        agg_metric = st.selectbox("Metrica:", ["new_cases", "new_deaths"],
                                  format_func=lambda v: {"new_cases": "Nuevos Casos",
                                                         "new_deaths": "Nuevas Muertes"}[v])
        agg_freq = st.selectbox("Frecuencia:", ["Mensual", "Semanal"],
                                format_func=str)
        freq_code = "M" if agg_freq == "Mensual" else "W"

        monthly = (
            df[df["continent"].isin(selected_continents)]
            .assign(period=lambda x: x["date"].dt.to_period(freq_code).dt.to_timestamp())
            .dropna(subset=[agg_metric])
            .groupby(["period", "continent"])[agg_metric].sum()
            .reset_index()
        )

        normalize_cont = st.toggle("Normalizar por casos totales del continente", value=False)

        fig2, ax2 = plt.subplots(figsize=(12, 5), facecolor="#0e1117")
        ax2.set_facecolor("#0e1117")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")

        for cont in monthly["continent"].unique():
            sub = monthly[monthly["continent"] == cont]
            y_vals = sub[agg_metric] / 1e6
            if normalize_cont:
                total_cont = sub[agg_metric].sum()
                y_vals = sub[agg_metric] / total_cont * 100 if total_cont > 0 else y_vals
            ax2.plot(sub["period"], y_vals, label=cont,
                     lw=1.8, color=cont_palette.get(cont, "#888"))
            ax2.fill_between(sub["period"], y_vals, alpha=0.05,
                             color=cont_palette.get(cont, "#888"))

        ylabel = "% del total" if normalize_cont else "Millones"
        ax2.set_ylabel(ylabel, color="white")
        ax2.set_xlabel("Fecha", color="white")
        ax2.set_title(f"{agg_metric} — {agg_freq} por Continente", color="white", fontsize=12)
        ax2.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        ax2.grid(True, alpha=0.15, color="white")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab_heatmap_t:
        st.subheader("Heatmap de intensidad por pais y mes")
        hm_countries_options = sorted(
            df[df["continent"].isin(selected_continents)]["location"].unique().tolist()
        )
        hm_defaults = [c for c in
                       ["United States", "Brazil", "India", "Germany",
                        "United Kingdom", "France", "Colombia", "Mexico"]
                       if c in hm_countries_options][:8]
        hm_countries = st.multiselect("Paises para el heatmap:", hm_countries_options,
                                      default=hm_defaults)
        hm_metric = st.selectbox("Metrica heatmap:",
                                 ["new_cases_smoothed_per_million",
                                  "new_deaths_smoothed_per_million"],
                                 format_func=lambda v: {
                                     "new_cases_smoothed_per_million": "Casos/Millon",
                                     "new_deaths_smoothed_per_million": "Muertes/Millon"
                                 }[v])

        if hm_countries:
            hm_data = (
                df[df["location"].isin(hm_countries)]
                .assign(month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
                .groupby(["location", "month"])[hm_metric].mean()
                .reset_index()
                .pivot(index="location", columns="month", values=hm_metric)
            )
            hm_data.columns = [str(c.date())[:7] for c in hm_data.columns]

            fig_hm, ax_hm = plt.subplots(
                figsize=(max(12, len(hm_data.columns) * 0.35), max(4, len(hm_countries) * 0.5)),
                facecolor="#0e1117"
            )
            ax_hm.set_facecolor("#0e1117")
            sns.heatmap(hm_data, ax=ax_hm, cmap="YlOrRd",
                        linewidths=0.2, linecolor="#222",
                        cbar_kws={"label": hm_metric})
            ax_hm.tick_params(colors="white", axis="both")
            ax_hm.set_xlabel("Mes", color="white")
            ax_hm.set_ylabel("")
            ax_hm.set_title(f"Intensidad mensual — {hm_metric}", color="white", fontsize=11)
            plt.xticks(rotation=90, fontsize=6)
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close()

    st.info("""
**Interpretacion:** Las series temporales revelan claramente las oleadas epidemicas.
Omicron (finales 2021 - inicios 2022) genero el mayor pico de casos globales.
La mortalidad sigue con rezago a los nuevos casos.
""")

# =============================================================
# PAGE 5 — FUENTE DE DATOS
# =============================================================
elif page == "📂 Fuente de Datos":
    st.title("📂 Documentacion de la Fuente de Datos")

    st.markdown(f"""
## Dataset: Our World in Data — COVID-19

| Campo | Detalle |
|-------|---------|
| **Nombre** | Our World in Data COVID-19 Dataset |
| **URL** | https://github.com/owid/covid-19-data |
| **Archivo** | `owid-covid-data.csv` |
| **Frecuencia** | Registro diario por pais |
| **Ultima fecha disponible** | {df["date"].max().date()} |
| **Fuentes originales** | Johns Hopkins CSSE, CDC, ECDC, OMS |
| **Licencia** | CC BY 4.0 |

---

## Como Actualizar el Dashboard

**Descarga manual del CSV:**
""")

    st.code(
        "curl -L https://covid.ourworldindata.org/data/owid-covid-data.csv -o owid-covid-data.csv",
        language="bash"
    )

    st.markdown("**Script Python para actualizacion automatica:**")
    st.code(
        'import urllib.request\n'
        'urllib.request.urlretrieve(\n'
        '    "https://covid.ourworldindata.org/data/owid-covid-data.csv",\n'
        '    "owid-covid-data.csv"\n'
        ')',
        language="python"
    )

    st.markdown("""
Reemplaza el archivo `owid-covid-data.csv` en tu repositorio de GitHub y
Streamlit Cloud se actualizara automaticamente.

---

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (incluye `STREAMLIT_PYTHONCODE.py` y `requirements.txt`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) → inicia sesion con GitHub.
3. Selecciona el repo → archivo principal `STREAMLIT_PYTHONCODE.py` → **Deploy**.
4. La URL publica queda disponible en minutos.

---

## Limitaciones del Dataset

- Los **casos confirmados** no son directamente comparables entre paises con
  distinta capacidad de testeo. Las **muertes por millon** son mas robustas.
- Los indicadores estructurales (`median_age`, `gdp_per_capita`) son pre-pandemia
  y se asumen constantes durante el periodo de analisis.
- Los agregados regionales de OWID (continentes, grupos de ingreso) fueron
  excluidos del analisis pais-a-pais para evitar doble conteo.
""")

    st.success(
        f"Dataset cargado correctamente. "
        f"Ultima fecha disponible: **{df['date'].max().date()}**"
    )
