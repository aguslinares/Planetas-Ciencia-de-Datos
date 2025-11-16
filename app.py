import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# =======================
# CONFIGURACIÓN GENERAL
# =======================
st.set_page_config(page_title="KOI / Exoplanetas - Grupo 4", layout="wide")

# VegaFusion para Altair (mejor rendimiento)
alt.data_transformers.enable("vegafusion")

# Estilos básicos (tema oscuro + tarjetas métricas)
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #111827 0, #020617 45%, #000000 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    section.main > div {
        padding-top: 1rem;
    }
    h1, h2, h3, h4 {
        color: #f9fafb;
    }
    .stSidebar {
        background-color: #020617 !important;
    }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 1rem;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.35);
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================
# CARGA DE DATOS
# =======================


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


df = load_data("Planetas_2025-10-21.csv")

# =======================
# HELPERS DE COLUMNAS
# =======================


def norm(s: str) -> str:
    """Normaliza un nombre de columna para facilitar coincidencias."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


colmap = {norm(c): c for c in df.columns}
cols_norm = set(colmap.keys())


def find_col(candidates) -> str | None:
    """
    Busca la primera columna existente entre una lista de candidatos.
    1) Coincidencia exacta por nombre normalizado.
    2) Coincidencia por substring (evitando columnas fp_*).
    """
    cands = [norm(c) for c in candidates]
    # Coincidencia exacta
    for c in cands:
        if c in cols_norm:
            return colmap[c]
    # Coincidencia por substring
    for c in cands:
        for k in cols_norm:
            if c in k and not k.startswith("fp_"):
                return colmap[k]
    return None


# Detectar columnas relevantes (planetarias / sistema)
name_col = find_col(["pl_name", "planet_name", "name", "planet", "nombre_kepler"])
mass_col = find_col(["pl_mass", "mass", "planet_mass", "mass_earth", "mass_mj", "m_sin_i"])
radius_col = find_col(
    [
        "radio_planeta_radios_tierra",
        "pl_rade",
        "radius",
        "planet_radius",
        "radius_earth",
        "rad_re",
        "rad",
        "radio_estelar_radios_solares",
        "radio_estelar",
    ]
)
period_col = find_col(["pl_orbper", "orbital_period", "period", "per", "periodo_orbital_dias"])
sma_col = find_col(["pl_orbsmax", "semi_major_axis", "sma", "a", "semieje_mayor_au"])
temp_col = find_col(
    ["pl_eqt", "teq", "temperature", "equilibrium_temperature", "temp", "temperatura_equilibrio_k"]
)
dist_col = find_col(["sy_dist", "st_dist", "distance", "star_distance", "system_distance"])
method_col = find_col(["discovery_method", "disc_method", "method"])
year_col = find_col(["disc_year", "discovery_year", "year"])
disp_col = find_col(["disposicion_final"])
koi_cnt = find_col(["conteo_koi"])
prof_col = find_col(["profundidad_transito_ppm"])
dur_col = find_col(["duracion_transito_horas"])
name_koi = find_col(["nombre_koi"])

# Convertir algunas columnas a numérico si vienen como texto
for c in [period_col, sma_col, temp_col, dist_col, radius_col, mass_col, prof_col, dur_col]:
    if c and df[c].dtype == "O":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =======================
# MENÚ LATERAL
# =======================

menu = st.sidebar.radio(
    "Secciones",
    ["Introducción","Columnas del dataset", "Visualizaciones", "Predicción del modelo" ],
    index=0,
)

# =======================
# SECCIÓN INTRODUCCIÓN
# =======================


def intro_section() -> None:
    st.title("Introducción - Análisis de Exoplanetas")

    st.markdown(
        """
        ### ¿Qué es un exoplaneta?

        Un **exoplaneta** es un planeta que orbita una estrella distinta del Sol.
        Desde mediados de los 90 se han confirmado miles: algunos son *gigantes gaseosos calientes*,
        otros *mini-Neptunos* y unos pocos tienen tamaños y temperaturas parecidos a la Tierra.

        ### ¿Qué es un KOI?

        En la misión **Kepler** de la NASA, un **KOI** (*Kepler Object of Interest*)
        es un objeto cuyo brillo sugiere la posible presencia de un planeta.
        Cada KOI pasa por etapas de detección automática, vetting y clasificación
        como `CONFIRMED`, `CANDIDATE` o `FALSE POSITIVE`.

        ### Método de tránsito (el que usamos)

        Kepler usa principalmente el **método de tránsito**:

        - Si un planeta pasa por delante de su estrella (desde nuestro punto de vista),
          la luz medida baja un poquito.
        - La **profundidad del tránsito** se relaciona con el **tamaño del planeta**.
        - El tiempo entre tránsitos permite estimar el **período orbital**.
        - Con modelos físicos y parámetros estelares se estiman radio, masa y temperatura de equilibrio.

        En este trabajo, **todo el dataset y las visualizaciones se basan en observaciones de tránsito**,
        igual que la misión Kepler. Por eso, muchas de nuestras columnas están ligadas directamente
        a este método (profundidad del tránsito, duración, período orbital, etc.).

        ### Otros métodos para detectar exoplanetas

        Aunque aquí nos enfocamos en el tránsito, existen otros métodos muy usados:

        - **Velocidad radial**: se mide el pequeño movimiento "de vaivén" de la estrella
          causado por la gravedad del planeta. Ese bamboleo deja una firma en el espectro de la luz.
        - **Imágenes directas**: en algunos casos se puede bloquear el brillo de la estrella
          (por ejemplo con un coronógrafo) y obtener imágenes directas de planetas muy luminosos y lejanos.
        - **Microlentes gravitacionales**: cuando una estrella pasa por delante de otra,
          su gravedad actúa como una lente; la presencia de un planeta modifica brevemente esa señal.

        Estos métodos son complementarios, pero **el catálogo KOI de Kepler y nuestro análisis
        están dominados por el método de tránsito**, que es especialmente eficiente
        para encontrar muchos planetas a la vez.
        """,
    )

    # Métricas rápidas
    c1, c2, c3 = st.columns(3)
    total = len(df)
    n_conf = int(df[disp_col].eq("CONFIRMED").sum()) if disp_col else None
    n_cand = int(df[disp_col].eq("CANDIDATE").sum()) if disp_col else None

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Total de objetos</h4>
                <h2>{total:,}</h2>
                <span>Filas en el catálogo analizado</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        txt = n_conf if n_conf is not None else "-"
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Planetas confirmados</h4>
                <h2>{txt}</h2>
                <span>disposición = CONFIRMED</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        txt = n_cand if n_cand is not None else "-"
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Candidatos</h4>
                <h2>{txt}</h2>
                <span>disposición = CANDIDATE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Ver primeras filas del dataset"):
        st.write(df.head())


# =======================
# SECCIÓN VISUALIZACIONES
# =======================


def viz_section() -> None:
    st.title("Dashboard KOI / Exoplanetas - Grupo 4")

    # =======================
    # A) Período orbital, Radio y Temperatura  (snippet A)
    # =======================
    st.subheader("A) Período orbital, Radio y Temperatura")

    # Redetección flexible (por si algo cambió)
    name_a = name_col or find_col(["pl_name", "planet_name", "name", "planet", "nombre_kepler"])
    mass_a = mass_col or find_col(["pl_mass", "mass", "planet_mass", "mass_earth", "mass_mj", "m_sin_i"])
    radius_a = radius_col or find_col(
        [
            "radio_planeta_radios_tierra",
            "pl_rade",
            "radius",
            "planet_radius",
            "radius_earth",
            "rad_re",
            "rad",
            "radio_estelar_radios_solares",
            "radio_estelar",
        ]
    )
    period_a = period_col or find_col(
        ["pl_orbper", "orbital_period", "period", "per", "periodo_orbital_dias"]
    )
    sma_a = sma_col or find_col(["pl_orbsmax", "semi_major_axis", "sma", "a", "semieje_mayor_au"])
    temp_a = temp_col or find_col(
        [
            "pl_eqt",
            "teq",
            "temperature",
            "equilibrium_temperature",
            "temp",
            "temperatura_equilibrio_k",
        ]
    )
    dist_a = dist_col or find_col(["sy_dist", "st_dist", "distance", "star_distance", "system_distance"])
    method_a = method_col or find_col(["discovery_method", "disc_method", "method"])
    year_a = year_col or find_col(["disc_year", "discovery_year", "year"])

    y_metric = mass_a or radius_a or dist_a
    x_metric = period_a or sma_a or dist_a

    if not x_metric or not y_metric:
        st.warning("No se encontraron columnas suficientes para construir el gráfico A.")
    else:
        cols_to_convert = [x_metric, y_metric, radius_a, mass_a, period_a, temp_a, dist_a]
        for c in cols_to_convert:
            if c is not None and c in df.columns and df[c].dtype == "O":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        data_a = df.copy()
        chart_a = alt.Chart(data_a)

        chart_a = chart_a.transform_filter(alt.datum[x_metric] > 0)
        chart_a = chart_a.transform_filter(alt.datum[y_metric] > 0)

        size_enc = alt.value(80)
        if radius_a and radius_a in df.columns:
            size_enc = alt.Size(
                f"{radius_a}:Q",
                title="Radio",
                scale=alt.Scale(range=[20, 600]),
            )
        elif mass_a and mass_a in df.columns:
            size_enc = alt.Size(
                f"{mass_a}:Q",
                title="Masa",
                scale=alt.Scale(range=[20, 600]),
            )

        color_enc = alt.value("#4C78A8")
        if temp_a and temp_a in df.columns:
            color_enc = alt.Color(
                f"{temp_a}:Q",
                title="Temp. equilibrio (K)",
                scale=alt.Scale(scheme="turbo"),
            )

        tooltip_cols = [
            c
            for c in [
                name_a,
                x_metric,
                y_metric,
                period_a,
                radius_a,
                mass_a,
                temp_a,
                dist_a,
                method_a,
                year_a,
            ]
            if c and c in df.columns
        ]
        tooltips = [
            alt.Tooltip(
                c,
                type="quantitative" if df[c].dtype != object else "nominal",
                title=c,
            )
            for c in tooltip_cols
        ]

        scatter = (
            chart_a.mark_circle(opacity=0.5, stroke="black", strokeWidth=0.2)
            .encode(
                x=alt.X(f"{x_metric}:Q", title=x_metric, scale=alt.Scale(type="log")),
                y=alt.Y(f"{y_metric}:Q", title=y_metric, scale=alt.Scale(type="log")),
                color=color_enc,
                size=size_enc,
                tooltip=tooltips,
            )
            .properties(
                title="A) Período orbital, Radio y Temperatura",
                width=720,
                height=460,
            )
        )

        # Filtro por disposición final usando selection_point (como en tu snippet)
        if "disposicion_final" in df.columns:
            disp_values = sorted(df["disposicion_final"].dropna().unique().tolist())
            disposition_dropdown = alt.binding_select(
                options=disp_values,
                name="Filtrar por disposición",
            )
            disposition_select = alt.selection_point(
                fields=["disposicion_final"],
                bind=disposition_dropdown,
                name="Filtrar por",
            )
            scatter = scatter.add_params(disposition_select).transform_filter(disposition_select)

        st.altair_chart(scatter.interactive(), use_container_width=True)

    # =======================
    # B) Distribución comparativa de períodos orbitales  (snippet B)
    # =======================
    st.subheader("B) Períodos – Candidatos vs Confirmados")

    if {"disposicion_final", "periodo_orbital_dias"}.issubset(df.columns):
        df_b = df.copy()
        df_b["periodo_orbital_dias"] = pd.to_numeric(
            df_b["periodo_orbital_dias"], errors="coerce"
        )
        df_subset = df_b[df_b["disposicion_final"].isin(["CONFIRMED", "CANDIDATE"])]
        df_subset = df_subset[df_subset["periodo_orbital_dias"].between(0.1, 1000)]

        if df_subset.empty:
            st.info("No hay datos suficientes para el gráfico B.")
        else:
            bins = np.logspace(np.log10(0.1), np.log10(1000), 20)
            labels = [
                f"{int(a)}–{int(b)}" if b >= 1 else f"{a:.1f}–{b:.1f}"
                for a, b in zip(bins[:-1], bins[1:])
            ]

            df_subset["rango_periodo"] = pd.cut(
                df_subset["periodo_orbital_dias"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )

            df_grouped = (
                df_subset.groupby(["rango_periodo", "disposicion_final"])
                .size()
                .reset_index(name="cantidad")
            )
            df_grouped["rango_periodo"] = pd.Categorical(
                df_grouped["rango_periodo"], categories=labels, ordered=True
            )

            bars = (
                alt.Chart(df_grouped)
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X(
                        "rango_periodo:N",
                        sort=None,
                        title="Período orbital (días, escala logarítmica aprox.)",
                    ),
                    y=alt.Y("cantidad:Q", title="Cantidad de objetos"),
                    color=alt.Color(
                        "disposicion_final:N",
                        title="Disposición final",
                    ),
                    tooltip=["rango_periodo", "disposicion_final", "cantidad"],
                )
            )

            text = (
                alt.Chart(df_grouped)
                .mark_text(dy=-5, size=11, fontWeight="bold", color="black")
                .encode(
                    x=alt.X("rango_periodo:N", sort=None),
                    y=alt.Y("cantidad:Q"),
                    detail="disposicion_final:N",
                    text=alt.Text("cantidad:Q", format=".0f"),
                )
            )

            chart_b = (
                (bars + text)
                .properties(
                    title="Distribución comparativa de períodos orbitales (Candidatos vs Confirmados)",
                    width=700,
                    height=400,
                )
                .configure_axisX(labelAngle=-40, labelFontSize=11)
                .configure_title(fontSize=16, fontWeight="bold")
            )

            st.altair_chart(chart_b, use_container_width=True)
    else:
        st.warning(
            "No se encontraron las columnas 'disposicion_final' y 'periodo_orbital_dias' para el gráfico B."
        )

    # =======================
    # C) Conteo de KOI por disposición final  (snippet C)
    # =======================
    st.subheader("C) Conteo KOI por disposición")

    koi_count_col = koi_cnt or find_col(["conteo_koi"])
    disposition_col = disp_col or find_col(["disposicion_final"])

    if (
        koi_count_col
        and disposition_col
        and koi_count_col in df.columns
        and disposition_col in df.columns
    ):
        koi_counts_disposition = (
            df.groupby([koi_count_col, disposition_col])
            .size()
            .reset_index(name="number_of_objects")
        )

        chart_c = (
            alt.Chart(koi_counts_disposition)
            .mark_bar()
            .encode(
                x=alt.X(f"{koi_count_col}:O", title="Conteo de KOI"),
                y=alt.Y("number_of_objects:Q", title="Número de objetos"),
                color=alt.Color(
                    f"{disposition_col}:N",
                    title="Disposición Final",
                ),
                tooltip=[koi_count_col, disposition_col, "number_of_objects"],
            )
            .properties(
                title="Distribución del Conteo de KOI por Disposición Final",
                width=600,
                height=400,
            )
        )

        st.altair_chart(chart_c, use_container_width=True)
    else:
        st.warning(
            "No se encontraron las columnas 'conteo_koi' y 'disposicion_final' para el gráfico C."
        )

    # =======================
    # D) Tránsitos: profundidad vs duración + histograma linkeado  (snippet D)
    # =======================
    st.subheader("D) Tránsitos (scatter + histograma linkeado)")

    required_cols_d = {
        "profundidad_transito_ppm",
        "duracion_transito_horas",
        "disposicion_final",
    }

    if required_cols_d.issubset(df.columns):
        df_d = df.copy()
        df_d["profundidad_transito_ppm"] = pd.to_numeric(
            df_d["profundidad_transito_ppm"], errors="coerce"
        )
        df_d["duracion_transito_horas"] = pd.to_numeric(
            df_d["duracion_transito_horas"], errors="coerce"
        )
        df_clean = df_d.dropna(
            subset=[
                "profundidad_transito_ppm",
                "duracion_transito_horas",
                "disposicion_final",
            ]
        )

        if df_clean.empty:
            st.info("No hay datos válidos para el gráfico D.")
        else:
            brush = alt.selection_interval(encodings=["x", "y"])

            scatter_d = (
                alt.Chart(df_clean)
                .mark_circle(size=65, opacity=0.5)
                .encode(
                    x=alt.X(
                        "duracion_transito_horas:Q",
                        title="Duración del tránsito (horas)",
                    ),
                    y=alt.Y(
                        "profundidad_transito_ppm:Q",
                        title="Profundidad del tránsito (ppm)",
                    ),
                    color=alt.condition(
                        brush, "disposicion_final:N", alt.value("lightgray")
                    ),
                    tooltip=[
                        "nombre_koi:N",
                        "disposicion_final:N",
                        alt.Tooltip(
                            "duracion_transito_horas:Q",
                            format=".2f",
                            title="Duración (h)",
                        ),
                        alt.Tooltip(
                            "profundidad_transito_ppm:Q",
                            format=".0f",
                            title="Profundidad (ppm)",
                        ),
                    ],
                )
                .add_params(brush)
                .properties(
                    width=700,
                    height=400,
                    title="Relación entre profundidad y duración del tránsito",
                )
            )

            bars_d = (
                alt.Chart(df_clean)
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X(
                        "disposicion_final:N",
                        title="Disposición final",
                    ),
                    y=alt.Y("count():Q", title="Cantidad de planetas"),
                    color=alt.Color("disposicion_final:N", legend=None),
                )
                .transform_filter(brush)
            )

            labels_d = (
                alt.Chart(df_clean)
                .mark_text(dy=-5, size=11, fontWeight="bold", color="black")
                .encode(
                    x=alt.X("disposicion_final:N"),
                    y=alt.Y("count():Q"),
                    text=alt.Text("count():Q", format=".0f"),
                )
                .transform_filter(brush)
            )

            hist_d = (
                (bars_d + labels_d)
                .properties(
                    width=700,
                    height=180,
                    title="Distribución de disposición final en los planetas seleccionados",
                )
            )

            chart_d = (scatter_d & hist_d).configure_title(
                fontSize=16,
                fontWeight="bold",
            )

            st.altair_chart(chart_d, use_container_width=True)
    else:
        st.warning(
            "No se encontraron columnas suficientes para el gráfico D."
        )

    st.caption("Fuente: catálogo KOI/Kepler. Columnas detectadas automáticamente por nombre.")


# =======================
# SECCIÓN PREDICCIÓN (placeholder)
# =======================


def model_section() -> None:
    st.title("Predicción del modelo (borrador)")
    st.markdown(
        """
        En esta sección se podría integrar un modelo de **Machine Learning** que,
        a partir de variables físicas (período, radio, profundidad del tránsito, etc.),
        estime por ejemplo:

        - La probabilidad de que un KOI sea **planeta confirmado**.
        - O la **clase** de disposición esperada (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`).

        Flujo sugerido:

        1. Entrenar un modelo (por ejemplo `RandomForestClassifier` o `XGBoost`) en un cuaderno aparte.
        2. Guardar el modelo con `joblib`.
        3. Cargarlo aquí y exponer un formulario para ingresar valores o seleccionar filas del dataset.
        """
    )


def columns_section() -> None:
    st.title("Columnas del dataset")

    st.markdown(
        """
        ### Cómo está armado el catálogo

        Cada **fila** del dataset corresponde a un objeto de interés de Kepler (**KOI**)
        y reúne información sobre:

        - Su identificación en los catálogos de Kepler.
        - Cómo fue clasificado (candidato, confirmado, falso positivo).
        - La geometría del tránsito observado.
        - Las propiedades orbitales del planeta.
        - Las propiedades físicas de la estrella.
        - Flags y metadatos del proceso de *vetting*.

        A continuación se muestra un **diccionario de datos** organizado por temas.
        """
    )

    column_info = [
        # Identificación y estado del KOI
        {
            "name": "id_kepler",
            "category": "Identificación y estado del KOI",
            "description": "Identificador numérico del objetivo observado por Kepler (KIC).",
            "unit": "entero (ID)",
            "context": "Sirve para cruzar este KOI con otros catálogos de Kepler.",
        },
        {
            "name": "nombre_koi",
            "category": "Identificación y estado del KOI",
            "description": "Nombre del objeto de interés Kepler (por ejemplo K00001.01).",
            "unit": "string",
            "context": "Identificador principal del KOI dentro de este catálogo.",
        },
        {
            "name": "nombre_kepler",
            "category": "Identificación y estado del KOI",
            "description": "Nombre oficial de planeta Kepler si el KOI fue confirmado.",
            "unit": "string",
            "context": "Suele tomar valores como Kepler-22 b; puede estar vacío para candidatos.",
        },
        {
            "name": "disposicion_final",
            "category": "Identificación y estado del KOI",
            "description": "Clasificación final del KOI (CONFIRMED, CANDIDATE, FALSE POSITIVE).",
            "unit": "categoría",
            "context": "Es uno de los campos clave para análisis supervisados.",
        },
        {
            "name": "disposicion_pipeline",
            "category": "Identificación y estado del KOI",
            "description": "Disposición asignada automáticamente por el pipeline de Kepler.",
            "unit": "categoría",
            "context": "Permite comparar la decisión automática con la revisión humana.",
        },
        {
            "name": "puntaje_confianza",
            "category": "Identificación y estado del KOI",
            "description": "Puntaje de confianza numérico asociado a la disposición del KOI.",
            "unit": "0–1 (adimensional)",
            "context": "Valores altos indican mayor confianza en la clasificación.",
        },
        {
            "name": "estado_vetting",
            "category": "Identificación y estado del KOI",
            "description": "Estado del proceso de vetting (por ejemplo Done).",
            "unit": "categoría",
            "context": "Indica si el KOI ya fue revisado por el equipo científico.",
        },
        {
            "name": "fecha_vetting",
            "category": "Identificación y estado del KOI",
            "description": "Fecha en la que se completó el vetting del KOI.",
            "unit": "fecha (AAAA-MM-DD)",
            "context": "Útil para analizar la evolución temporal del catálogo.",
        },
        # Parámetros de tránsito y órbita
        {
            "name": "periodo_orbital_dias",
            "category": "Parámetros de tránsito y órbita",
            "description": "Periodo orbital del planeta alrededor de su estrella.",
            "unit": "días",
            "context": "Se estima a partir de la repetición de los tránsitos.",
        },
        {
            "name": "epoca_transito_bkjd",
            "category": "Parámetros de tránsito y órbita",
            "description": "Época de referencia del tránsito en tiempo BKJD.",
            "unit": "BKJD (BJD – 2454833)",
            "context": "Sirve como punto de anclaje temporal para los modelos de tránsito.",
        },
        {
            "name": "parametro_impacto",
            "category": "Parámetros de tránsito y órbita",
            "description": "Parámetro de impacto del tránsito, relacionado con qué tan centrado es el tránsito sobre el disco estelar.",
            "unit": "adimensional",
            "context": "Valores cercanos a 0 indican un tránsito casi central; cercanos a 1, rasantes.",
        },
        {
            "name": "duracion_transito_horas",
            "category": "Parámetros de tránsito y órbita",
            "description": "Duración total del tránsito observado.",
            "unit": "horas",
            "context": "Depende del tamaño de la estrella, del planeta y de la órbita.",
        },
        {
            "name": "profundidad_transito_ppm",
            "category": "Parámetros de tránsito y órbita",
            "description": "Disminución relativa de brillo durante el tránsito.",
            "unit": "ppm (partes por millón)",
            "context": "Está directamente relacionada con el área aparente del planeta frente a la estrella.",
        },
        {
            "name": "semieje_mayor_au",
            "category": "Parámetros de tránsito y órbita",
            "description": "Semieje mayor de la órbita del planeta.",
            "unit": "UA (unidades astronómicas)",
            "context": "Permite comparar la distancia a la estrella con la distancia Tierra–Sol.",
        },
        {
            "name": "inclinacion_grados",
            "category": "Parámetros de tránsito y órbita",
            "description": "Inclinación de la órbita respecto a la línea de visión.",
            "unit": "grados",
            "context": "Para que haya tránsito, la inclinación debe estar muy cerca de 90°.",
        },
        {
            "name": "excentricidad",
            "category": "Parámetros de tránsito y órbita",
            "description": "Excentricidad orbital del planeta.",
            "unit": "0–1 (adimensional)",
            "context": "0 indica órbita circular; valores mayores, órbitas más elípticas.",
        },
        {
            "name": "numero_transitos",
            "category": "Parámetros de tránsito y órbita",
            "description": "Cantidad de tránsitos observados en los datos de Kepler.",
            "unit": "entero",
            "context": "Más tránsitos implican mejores estimaciones de los parámetros orbitales.",
        },
        # Propiedades físicas del planeta
        {
            "name": "radio_relativo_rp_rs",
            "category": "Propiedades físicas del planeta",
            "description": "Relación entre el radio del planeta y el radio de la estrella.",
            "unit": "adimensional",
            "context": "Es una medida directa que surge del ajuste del modelo de tránsito.",
        },
        {
            "name": "radio_planeta_radios_tierra",
            "category": "Propiedades físicas del planeta",
            "description": "Radio estimado del planeta en radios terrestres.",
            "unit": "R⊕",
            "context": "Permite clasificar el planeta como tipo Tierra, Neptuno, Júpiter, etc.",
        },
        {
            "name": "flujo_insolacion_tierra",
            "category": "Propiedades físicas del planeta",
            "description": "Flujo de radiación que recibe el planeta comparado con la Tierra.",
            "unit": "S⊕ (adimensional)",
            "context": "Suele usarse para discutir habitabilidad y zona habitable.",
        },
        {
            "name": "temperatura_equilibrio_k",
            "category": "Propiedades físicas del planeta",
            "description": "Temperatura de equilibrio teórica del planeta.",
            "unit": "Kelvin",
            "context": "Se calcula asumiendo balance radiativo ideal y depende de la distancia y de la estrella.",
        },
        # Propiedades físicas de la estrella
        {
            "name": "temperatura_efectiva_estrella_k",
            "category": "Propiedades físicas de la estrella",
            "description": "Temperatura efectiva de la estrella anfitriona.",
            "unit": "Kelvin",
            "context": "Ayuda a clasificar la estrella (tipo espectral) y el color.",
        },
        {
            "name": "radio_estelar_radios_solares",
            "category": "Propiedades físicas de la estrella",
            "description": "Radio de la estrella en radios solares.",
            "unit": "R☉",
            "context": "Impacta directamente en el tamaño estimado del planeta.",
        },
        {
            "name": "logg_estelar_cgs",
            "category": "Propiedades físicas de la estrella",
            "description": "Logaritmo de la gravedad superficial de la estrella en cgs.",
            "unit": "log10(cm/s²)",
            "context": "Distingue entre enanas (log g alto) y gigantes (log g bajo).",
        },
        {
            "name": "metallicidad_estelar_dex",
            "category": "Propiedades físicas de la estrella",
            "description": "Metallicidad estelar, típicamente [Fe/H] en dex.",
            "unit": "dex",
            "context": "Estrellas más metálicas tienden a albergar más planetas gigantes.",
        },
        {
            "name": "masa_estelar_masas_solares",
            "category": "Propiedades físicas de la estrella",
            "description": "Masa de la estrella en unidades de masa solar.",
            "unit": "M☉",
            "context": "Se usa para derivar parámetros orbitales y condiciones del sistema.",
        },
        {
            "name": "densidad_estelar",
            "category": "Propiedades físicas de la estrella",
            "description": "Densidad promedio de la estrella.",
            "unit": "g/cm³ (aprox.)",
            "context": "Se puede inferir a partir de la forma detallada del tránsito.",
        },
        {
            "name": "edad_estelar_gyr",
            "category": "Propiedades físicas de la estrella",
            "description": "Edad estimada de la estrella.",
            "unit": "Gyr (miles de millones de años)",
            "context": "Aporta contexto evolutivo al sistema planetario.",
        },
        # Posición en el cielo y fotometría
        {
            "name": "ascension_recta_grados",
            "category": "Posición en el cielo y fotometría",
            "description": "Ascensión recta de la estrella en coordenadas ecuatoriales.",
            "unit": "grados",
            "context": "Coordenada similar a la longitud, medida sobre el cielo.",
        },
        {
            "name": "declinacion_grados",
            "category": "Posición en el cielo y fotometría",
            "description": "Declinación de la estrella en coordenadas ecuatoriales.",
            "unit": "grados",
            "context": "Coordenada similar a la latitud celeste.",
        },
        {
            "name": "magnitud_kepler",
            "category": "Posición en el cielo y fotometría",
            "description": "Magnitud aparente en el sistema fotométrico de Kepler.",
            "unit": "magnitudes",
            "context": "Indica cuán brillante es el objetivo en la banda de Kepler.",
        },
        {
            "name": "snr_modelo",
            "category": "Posición en el cielo y fotometría",
            "description": "Relación señal-ruido del modelo de tránsito ajustado.",
            "unit": "adimensional",
            "context": "Valores altos indican tránsitos detectados con mayor claridad.",
        },
        # Flags de falso positivo y multiplicidad
        {
            "name": "conteo_koi",
            "category": "Flags de falso positivo y multiplicidad",
            "description": "Cantidad de KOI asociados a la misma estrella.",
            "unit": "entero",
            "context": "Sistemas con conteo alto tienen múltiples planetas candidatos.",
        },
        {
            "name": "fp_no_transito",
            "category": "Flags de falso positivo y multiplicidad",
            "description": "Flag que indica evidencia de que la señal no proviene de un tránsito planetario.",
            "unit": "0/1",
            "context": "Cuando vale 1, sugiere que la variabilidad no es compatible con un tránsito.",
        },
        {
            "name": "fp_senal_estelar",
            "category": "Flags de falso positivo y multiplicidad",
            "description": "Flag de señal causada por la propia estrella (actividad, manchas, etc.).",
            "unit": "0/1",
            "context": "Ayuda a descartar variaciones intrínsecas como planetas.",
        },
        {
            "name": "fp_desplazamiento_centroide",
            "category": "Flags de falso positivo y multiplicidad",
            "description": "Flag por desplazamiento del centroide de luz durante el tránsito.",
            "unit": "0/1",
            "context": "Si el centro de luz se mueve, puede indicar una fuente contaminante cercana.",
        },
        {
            "name": "fp_contaminacion_binaria_eclipsante",
            "category": "Flags de falso positivo y multiplicidad",
            "description": "Flag de posible contaminación por una binaria eclipsante vecina.",
            "unit": "0/1",
            "context": "Es una de las principales fuentes de falsos positivos en catálogos de tránsitos.",
        },
        # Metadatos de origen y procesamiento
        {
            "name": "origen_parametros",
            "category": "Metadatos de origen y procesamiento",
            "description": "Fuente de los parámetros físicos adoptados.",
            "unit": "string",
            "context": "Permite rastrear de qué versión o estudio provienen los parámetros.",
        },
        {
            "name": "origen_disposicion",
            "category": "Metadatos de origen y procesamiento",
            "description": "Fuente de la disposición final del KOI.",
            "unit": "string",
            "context": "Diferencia, por ejemplo, entre catálogos oficiales y actualizaciones posteriores.",
        },
        {
            "name": "fecha_procesamiento",
            "category": "Metadatos de origen y procesamiento",
            "description": "Fecha de procesamiento o generación de este registro.",
            "unit": "fecha (AAAA-MM-DD)",
            "context": "Útil para controlar versiones del catálogo y reproducibilidad.",
        },
    ]

    by_category: dict[str, list[dict]] = {}
    for info in column_info:
        if info["name"] not in df.columns:
            continue
        by_category.setdefault(info["category"], []).append(info)

    for category, items in by_category.items():
        st.subheader(category)
        cols = st.columns(2)
        for i, info in enumerate(items):
            unit_html = (
                f'<p style="margin-bottom:0.25rem;"><strong>Unidad:</strong> {info["unit"]}</p>'
                if info.get("unit")
                else ""
            )
            context_html = (
                f'<p style="font-size:0.9rem;color:#9ca3af;">{info["context"]}</p>'
                if info.get("context")
                else ""
            )
            card_html = f"""
            <div class="metric-card">
                <h4><code>{info["name"]}</code></h4>
                <p style="margin-bottom:0.25rem;"><strong>Qué mide:</strong> {info["description"]}</p>
                {unit_html}
                {context_html}
            </div>
            """
            with cols[i % 2]:
                st.markdown(card_html, unsafe_allow_html=True)

    st.caption("Diccionario de variables basado en la documentación pública de la misión Kepler.")


# show_columns_only = st.sidebar.checkbox("Ver solo columnas del dataset")

# if show_columns_only:
#     columns_section()
#     st.stop()


# =======================
# DESPACHO DE SECCIONES
# =======================

if menu == "Introducción":
    intro_section()
elif menu == "Columnas del dataset":
    columns_section()
elif menu == "Visualizaciones":
    viz_section()
else:
    model_section()
