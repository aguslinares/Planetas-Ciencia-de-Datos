import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# ---- Setup
st.set_page_config(page_title="KOI / Exoplanetas – Grupo 4", layout="wide")
# Habilitar el transformador de datos VegaFusion para mejor rendimiento
alt.data_transformers.enable("vegafusion")

# Función para normalizar nombres de columnas
def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

# Crear un mapeo de nombres de columna normalizados a nombres originales
colmap = {norm(c): c for c in df.columns}
cols_norm = set(colmap.keys())

# Función para encontrar la primera columna existente entre una lista de candidatos
def find_col(candidates):
    """
    Busca la primera columna existente entre una lista de candidatos.
    - Coincidencia exacta por nombre normalizado
    - Luego 'contains' por substring
    Retorna None si no encuentra.
    """
    cands = [norm(c) for c in candidates]
    # Coincidencia exacta
    for c in cands:
        if c in cols_norm:
            return colmap[c]
    # Coincidencia por substring (evitando columnas de falsos positivos)
    for c in cands:
        for k in cols_norm:
            if c in k and not k.startswith("fp_"):
                return colmap[k]
    return None

# ---------------------------------------------------------------------
# Selección flexible de columnas típicas en un dataset de exoplanetas
# (usa lo que encuentre en TU archivo; si falta alguna, se adapta)
# ---------------------------------------------------------------------
name_col     = find_col(["pl_name","planet_name","name","planet", "nombre_kepler"])
mass_col     = find_col(["pl_mass","mass","planet_mass","mass_earth","mass_mj","m_sin_i"])
radius_col   = find_col(["pl_rade","radius","planet_radius","radius_earth","rad_re","rad", "radio_estelar"])
period_col   = find_col(["pl_orbper","orbital_period","period","per", "periodo_orbital_dias"])
sma_col      = find_col(["pl_orbsmax","semi_major_axis","sma","a"])
temp_col     = find_col(["pl_eqt","teq","temperature","equilibrium_temperature","temp", "temperatura_equilibrio_k"])
dist_col     = find_col(["sy_dist","st_dist","distance","star_distance","system_distance"])
method_col   = find_col(["discovery_method","disc_method","method"])
year_col     = find_col(["disc_year","discovery_year","year"])
host_teff    = find_col(["st_teff","host_star_temperature","star_temp", "temperatura_efectiva_estelar"])
host_metal   = find_col(["st_met","host_star_metallicity","metallicity"])

# Definir métricas para los ejes X e Y del gráfico A
y_metric = mass_col or radius_col or dist_col
x_metric = period_col or sma_col or dist_col

# Convertir columnas a tipo numérico si es necesario y aplicar filtros básicos
cols_to_convert = [x_metric, y_metric, radius_col, mass_col, period_col, temp_col, dist_col]
for c in cols_to_convert:
    if c is not None and c in df.columns and df[c].dtype == "O":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------------------------------------------------------------
# Chart A: Interactive bubble scatter (Period vs Mass/Radius)
# - X = period (or SMA / distance) in log
# - Y = mass (or radius / distance)
# - color = equilibrium temperature (if exists)
# - size = radius (if exists) or mass
# ---------------------------------------------------------------------
data_a = df.copy()

# Base + filtros
base_a = alt.Chart(data_a)

# Aplicar filtros solo si las columnas existen
if x_metric:
    base_a = base_a.transform_filter(alt.datum[x_metric] > 0)
if y_metric:
    base_a = base_a.transform_filter(alt.datum[y_metric] > 0)

# Codificación de tamaño (Radio o Masa)
size_enc = alt.value(80) # Tamaño por defecto
if radius_col and radius_col in df.columns:
    size_enc = alt.Size(radius_col + ":Q", title="Radio", scale=alt.Scale(range=[20, 600]))
elif mass_col and mass_col in df.columns:
    size_enc = alt.Size(mass_col + ":Q", title="Masa", scale=alt.Scale(range=[20, 600]))

# Codificación de color (Temperatura de equilibrio)
color_enc = alt.value("#4C78A8") # Color por defecto
if temp_col and temp_col in df.columns:
    color_enc = alt.Color(temp_col + ":Q", title="Temp. equilibrio (K)", scale=alt.Scale(scheme="turbo"))

# Tooltips
tooltip_cols = [c for c in [name_col, x_metric, y_metric, period_col, radius_col, mass_col, temp_col, dist_col, method_col, year_col] if c and c in df.columns]
tooltips = [alt.Tooltip(c, type="quantitative" if (df[c].dtype!=object) else "nominal", title=c) for c in tooltip_cols]

# Create a dropdown filter for 'disposicion_final'
disposition_dropdown = alt.binding_select(options=df['disposicion_final'].unique().tolist())
disposition_select = alt.selection_point(fields=['disposicion_final'], bind=disposition_dropdown, name="Filtrar por")

# Creación del gráfico de dispersión
scatter = (
    base_a.mark_circle(opacity=0.5, stroke="black", strokeWidth=0.2)
    .encode(
        x=alt.X(f"{x_metric}:Q", title=x_metric, scale=alt.Scale(type="log")) if x_metric else alt.X(""),
        y=alt.Y(f"{y_metric}:Q", title=y_metric, scale=alt.Scale(type="log")) if y_metric else alt.Y(""),
        color=color_enc,
        size=size_enc,
        tooltip=tooltips
    )
    .properties(
        title="A) Período orbital, Radio y Temperatura",
        width=720, height=460
    )
    .add_params(disposition_select) # Add the filter to the chart
    .transform_filter(disposition_select) # Apply the filter
    .interactive()  # pan/zoom
)

chart_a = scatter

chart_a
    
    # =======================
    # Sección B – Períodos: CANDIDATE vs CONFIRMED
    # =======================
    if disp_col and period_col:
        df_subset = df[df[disp_col].isin(["CONFIRMED", "CANDIDATE"])].copy()
        df_subset = df_subset[df_subset[period_col].between(0.1, 1000)]
        bins = np.logspace(np.log10(0.1), np.log10(1000), 20)
        labels = [f"{int(a)}–{int(b)}" if b >= 1 else f"{a:.1f}–{b:.1f}" for a, b in zip(bins[:-1], bins[1:])]
        df_subset["rango_periodo"] = pd.cut(df_subset[period_col], bins=bins, labels=labels, include_lowest=True)
        grouped = (df_subset.groupby(["rango_periodo", disp_col]).size().reset_index(name="cantidad"))
        grouped["rango_periodo"] = pd.Categorical(grouped["rango_periodo"], categories=labels, ordered=True)
    
        bars = (
            alt.Chart(grouped)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("rango_periodo:O", sort=labels, title="Período (días, aprox. log)"),
                y=alt.Y("cantidad:Q", title="Cantidad"),
                color=alt.Color(f"{disp_col}:N", title="Disposición"),
                tooltip=["rango_periodo", disp_col, "cantidad"]
            )
            .properties(title="B) Distribución comparativa de períodos orbitales", width=900, height=380)
        )
        text = (
            alt.Chart(grouped)
            .mark_text(dy=-5, size=10, fontWeight="bold")
            .encode(
                x=alt.X("rango_periodo:O", sort=labels),
                y="cantidad:Q",
                detail=f"{disp_col}:N",
                text=alt.Text("cantidad:Q", format=".0f")
            )
        )
        st.subheader("B) Períodos – Candidatos vs Confirmados")
        st.altair_chart((bars + text).configure_axisX(labelAngle=-35), use_container_width=True)
    
    # =======================
    # Sección C – KOI por disposición
    # =======================
    if koi_cnt and disp_col:
        koi_counts = df.groupby([koi_cnt, disp_col]).size().reset_index(name="number_of_objects")
        chart_koi = (
            alt.Chart(koi_counts)
            .mark_bar()
            .encode(
                x=alt.X(f"{koi_cnt}:O", title="Conteo de KOI"),
                y=alt.Y("number_of_objects:Q", title="Número de objetos"),
                color=alt.Color(f"{disp_col}:N", title="Disposición Final"),
                tooltip=[koi_cnt, disp_col, "number_of_objects"]
            )
            .properties(title="C) Distribución del Conteo de KOI por Disposición Final", width=900, height=380)
            .add_params(p_dispo)
        )
        if dispo_expr:
            chart_koi = chart_koi.transform_filter(dispo_expr)
        st.subheader("C) Conteo KOI por Disposición")
        st.altair_chart(chart_koi, use_container_width=True)
    
    # =======================
    # Sección D – Tránsito: profundidad vs duración + histograma linkeado
    # =======================
    prof_col = find_col(["profundidad_transito_ppm"])
    dur_col  = find_col(["duracion_transito_horas"])
    name_koi = find_col(["nombre_koi"])
    
    if prof_col and dur_col and disp_col:
        # num
        df[prof_col] = pd.to_numeric(df[prof_col], errors="coerce")
        df[dur_col]  = pd.to_numeric(df[dur_col], errors="coerce")
        df_clean = df.dropna(subset=[prof_col, dur_col, disp_col]).copy()
    
        brush = alt.selection_interval(encodings=["x", "y"])
    
        scatter_t = (
            alt.Chart(df_clean)
            .mark_circle(size=60, opacity=0.5)
            .encode(
                x=alt.X(f"{dur_col}:Q", title="Duración del tránsito (h)"),
                y=alt.Y(f"{prof_col}:Q", title="Profundidad del tránsito (ppm)"),
                color=alt.condition(brush, f"{disp_col}:N", alt.value("lightgray")),
                tooltip=[name_koi or name_col or "index", disp_col,
                         alt.Tooltip(f"{dur_col}:Q", format=".2f", title="Duración (h)"),
                         alt.Tooltip(f"{prof_col}:Q", format=".0f", title="Profundidad (ppm)")]
            )
            .add_params(brush, p_dispo)
            .properties(width=900, height=420, title="D) Profundidad vs Duración del tránsito")
        )
        if dispo_expr:
            scatter_t = scatter_t.transform_filter(dispo_expr)
    
        bars = (
            alt.Chart(df_clean)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X(f"{disp_col}:N", title="Disposición final"),
                y=alt.Y("count():Q", title="Cantidad"),
                color=alt.Color(f"{disp_col}:N", legend=None)
            )
            .transform_filter(brush)
            .add_params(p_dispo)
        )
        if dispo_expr:
            bars = bars.transform_filter(dispo_expr)
    
        labels = (
            alt.Chart(df_clean)
            .mark_text(dy=-5, size=11, fontWeight="bold", color="black")
            .encode(x=alt.X(f"{disp_col}:N"), y=alt.Y("count():Q"), text=alt.Text("count():Q", format=".0f"))
            .transform_filter(brush)
            .add_params(p_dispo)
        )
        if dispo_expr:
            labels = labels.transform_filter(dispo_expr)
    
        st.subheader("D) Tránsitos (scatter + histograma linkeado)")
        st.altair_chart((scatter_t & (bars + labels)).configure_title(fontSize=16, fontWeight="bold"),
                        use_container_width=True)
    
    st.caption("Fuente: Catálogo Kepler / columnas detectadas automáticamente por nombre.")

if menu == "Predicción del modelo":
    st.title("Predicción del modelo")
    





