import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# ---- Setup
st.set_page_config(page_title="KOI / Exoplanetas – Grupo 4", layout="wide")
alt.data_transformers.enable("vegafusion")

# ---- Helpers
def norm(s): return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # si existe, úsalo como índice (opcional)
    if "id_kepler" in df.columns:
        df = df.set_index("id_kepler")
    return df

def make_colmap(df):
    colmap = {norm(c): c for c in df.columns}
    cols_norm = set(colmap.keys())
    def find_col(candidates):
        cands = [norm(c) for c in candidates]
        for c in cands:
            if c in cols_norm: return colmap[c]
        for c in cands:
            for k in cols_norm:
                if c in k and not k.startswith("fp_"):
                    return colmap[k]
        return None
    return find_col

# ---- Sidebar (datos + filtros generales)
st.sidebar.header("Datos")
csv_path = st.sidebar.text_input("Ruta del CSV", value="Planetas_2025-10-21.csv")
df = load_data(csv_path)

find_col = make_colmap(df)

name_col   = find_col(["pl_name","planet_name","name","planet","nombre_kepler"])
mass_col   = find_col(["pl_mass","mass","planet_mass","mass_earth","mass_mj","m_sin_i"])
radius_col = find_col(["pl_rade","radius","planet_radius","radius_earth","rad_re","rad","radio_estelar"])
period_col = find_col(["pl_orbper","orbital_period","period","per","periodo_orbital_dias"])
sma_col    = find_col(["pl_orbsmax","semi_major_axis","sma","a"])
temp_col   = find_col(["pl_eqt","teq","temperature","equilibrium_temperature","temp","temperatura_equilibrio_k"])
dist_col   = find_col(["sy_dist","st_dist","distance","star_distance","system_distance"])
method_col = find_col(["discovery_method","disc_method","method"])
year_col   = find_col(["disc_year","discovery_year","year"])
disp_col   = find_col(["disposicion_final"])
koi_cnt    = find_col(["conteo_koi"])

# num casts usados por los charts
for c in [period_col, sma_col, temp_col, dist_col, radius_col, mass_col]:
    if c and df[c].dtype == "O":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Dropdown global de Disposición (con “(Todas)”)
opts = ["(Todas)"]
if disp_col:
    opts += sorted(df[disp_col].dropna().unique().tolist())
p_dispo = alt.param(name="p_dispo", value="(Todas)",
                    bind=alt.binding_select(options=opts, name="Disposición: "))
dispo_expr = f"(p_dispo == '(Todas)') || (datum.{disp_col} == p_dispo)" if disp_col else None

st.title("Dashboard KOI / Exoplanetas – Grupo 4")

# =======================
# Sección A – Scatter Periodo vs Masa/Radio
# =======================
x_metric = period_col or sma_col or dist_col
y_metric = mass_col or radius_col or dist_col

base_a = alt.Chart(df)
if x_metric:
    base_a = base_a.transform_filter(alt.datum[x_metric] > 0)
if y_metric:
    base_a = base_a.transform_filter(alt.datum[y_metric] > 0)

size_enc = alt.value(80)
if radius_col:
    size_enc = alt.Size(f"{radius_col}:Q", title="Radio", scale=alt.Scale(range=[20, 600]))
elif mass_col:
    size_enc = alt.Size(f"{mass_col}:Q", title="Masa", scale=alt.Scale(range=[20, 600]))

color_enc = alt.value("#4C78A8")
if temp_col:
    color_enc = alt.Color(f"{temp_col}:Q", title="Temp. equilibrio (K)", scale=alt.Scale(scheme="turbo"))

tt_cols = [c for c in [name_col, x_metric, y_metric, period_col, radius_col, mass_col, temp_col, dist_col, method_col, year_col, disp_col] if c]
tooltips = [alt.Tooltip(c, type=("quantitative" if (c and df[c].dtype != object) else "nominal"), title=c) for c in tt_cols]

scatter = (
    base_a.mark_circle(opacity=0.55, stroke="black", strokeWidth=0.2)
    .encode(
        x=alt.X(f"{x_metric}:Q", title=x_metric, scale=alt.Scale(type="log")) if x_metric else alt.X(""),
        y=alt.Y(f"{y_metric}:Q", title=y_metric, scale=alt.Scale(type="log")) if y_metric else alt.Y(""),
        color=color_enc,
        size=size_enc,
        tooltip=tooltips
    )
    .properties(title="A) Período orbital vs Masa/Radio (bubble)", width=800, height=480)
    .add_params(p_dispo)
)

if dispo_expr:
    scatter = scatter.transform_filter(dispo_expr)

st.subheader("A) Período orbital vs Masa/Radio")
st.altair_chart(scatter.interactive(), use_container_width=True)

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


