# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# Config
# =========================
st.set_page_config(
    page_title="LLMs Timeline Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE = "LLMs cronología.xlsx"
COL_MODEL = "Model"
COL_PARAMS = "Parameters \n(B)"
COL_DATE = "Announced\n▼"

CAND_BENCH = ["MMLU", "MMLU\n-Pro", "GPQA", "HLE"]
CAND_CATS  = ["Lab", "Arch", "Public?"]

# =========================
# Load
# =========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    hdr = raw.iloc[0]
    df = raw[1:].copy()
    df.columns = hdr

    # Type
    df[COL_PARAMS] = pd.to_numeric(df[COL_PARAMS], errors="coerce")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", format="mixed")

    # Drop essential nulls
    df = df.dropna(subset=[COL_MODEL, COL_PARAMS, COL_DATE]).copy()

    # Bench numeric
    for c in CAND_BENCH:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep 2023+
    df = df[df[COL_DATE].dt.year >= 2023].copy()

    # Clean categories
    for c in CAND_CATS:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": np.nan})

    df = df.sort_values(COL_DATE)
    return df

df = load_data(FILE)

bench_cols = [c for c in CAND_BENCH if c in df.columns]
cat_cols = [c for c in CAND_CATS if c in df.columns]

if len(df) == 0:
    st.error("No hay datos disponibles. Verifica que el archivo Excel contenga fechas, parámetros y modelos válidos.")
    st.stop()

# =========================
# Sidebar - look & feel
# =========================
st.sidebar.title("Configuración")
st.sidebar.markdown("---")
with st.sidebar.expander("Apariencia", expanded=True):
    theme = st.radio("Tema", ["Claro", "Oscuro"], index=0, horizontal=True)
    
    color_mode = st.selectbox(
        "Colorear por",
        ["Ninguno"] + cat_cols + (["Benchmark (score)"] if bench_cols else []),
        index=0
    )
    
    col1, col2 = st.columns(2)
    with col1:
        marker_size = st.slider("Tamaño", 4, 18, 8, 1)
    with col2:
        marker_opacity = st.slider("Opacidad", 0.10, 1.00, 0.45, 0.05)
    
    label_mode = st.selectbox(
        "Etiquetas",
        ["Solo TopN", "Ninguna (solo hover)", "TopN + extremos"],
        index=0
    )

# =========================
# Sidebar - filters
# =========================
st.sidebar.markdown("---")
with st.sidebar.expander("Filtros", expanded=True):
    min_date = df[COL_DATE].min().date()
    max_date = df[COL_DATE].max().date()
    date_from, date_to = st.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    pmin = float(df[COL_PARAMS].min())
    pmax = float(df[COL_PARAMS].max())
    params_min, params_max = st.slider(
        "Parámetros (B)",
        min_value=float(np.floor(pmin*100)/100),
        max_value=float(np.ceil(pmax*100)/100),
        value=(float(pmin), float(pmax)),
        step=max((pmax - pmin)/300, 0.01)
    )
    
    # Text search
    search = st.text_input(
        "Buscar modelo",
        value="",
        placeholder="Ej: GPT, Claude, Llama..."
    ).strip()
    use_regex = st.checkbox("Usar expresión regular", value=False)

    # Benchmark & Y
    if bench_cols:
        bench_options = ["Ninguno"] + bench_cols
        bench_selection = st.selectbox(
            "Benchmark",
            bench_options,
            index=0
        )
        bench = None if bench_selection == "Ninguno" else bench_selection
        only_with_score = st.checkbox(
            "Solo con score disponible",
            value=False,
            disabled=(bench is None)
        )
        if bench is None:
            only_with_score = False
    else:
        bench, only_with_score = None, False
        st.info("No se detectaron benchmarks en el dataset")

    y_mode = st.radio(
        "Eje Y",
        ["Tamaño (params)", "Score (benchmark)"],
        index=0,
        horizontal=True
    )
    if y_mode == "Score (benchmark)" and bench is None:
        st.warning("No hay benchmark disponible; se usará Params")
        y_mode = "Tamaño (params)"

st.sidebar.markdown("---")
with st.sidebar.expander("Visualización", expanded=True):
    top_n = st.slider(
        "Top N (destacados)",
        5, 100, 20, 5
    )
    show_n = st.slider(
        "Modelos a mostrar",
        50, min(4000, len(df)), min(600, len(df)), 50
    )
    
    # Y scale control (only makes sense for params; scores often linear)
    y_scale = st.selectbox(
        "Escala eje Y",
        ["Log", "Lineal"],
        index=0
    )
    if y_mode == "Score (benchmark)":
        y_scale = "Lineal"
    
    # Plot size
    plot_height = st.slider(
        "Altura del gráfico (px)",
        600, 1400, 950, 50
    )

# =========================
# Apply filters
# =========================
d = df.copy()
d = d[(d[COL_DATE].dt.date >= date_from) & (d[COL_DATE].dt.date <= date_to)]
d = d[(d[COL_PARAMS] >= params_min) & (d[COL_PARAMS] <= params_max)]

if bench and only_with_score:
    d = d.dropna(subset=[bench])

if search:
    if use_regex:
        try:
            rgx = re.compile(search, re.IGNORECASE)
            d = d[d[COL_MODEL].astype(str).apply(lambda x: bool(rgx.search(x)))]
        except re.error as e:
            st.error(f"Expresión regular inválida: {e}. Desactiva 'Usar expresión regular' o corrige el patrón.")
            st.stop()
    else:
        d = d[d[COL_MODEL].astype(str).str.contains(search, case=False, na=False)]

if len(d) == 0:
    st.warning("No hay modelos que coincidan con los filtros actuales. Ajusta los parámetros en la barra lateral.")
    st.stop()

# Density sampling preserving time
d = d.sort_values(COL_DATE)
if len(d) > show_n:
    idx = np.linspace(0, len(d)-1, show_n).astype(int)
    d_show = d.iloc[idx].copy()
else:
    d_show = d.copy()

# Determine Top set
if y_mode == "Score (benchmark)" and bench:
    d_top = d.dropna(subset=[bench]).nlargest(top_n, bench).copy()
else:
    # fallback relevance: largest params
    d_top = d.nlargest(top_n, COL_PARAMS).copy()

# Ensure top included
d_show = pd.concat([d_show, d_top]).drop_duplicates(subset=[COL_MODEL, COL_DATE]).copy()

# Choose Y
if y_mode == "Tamaño (params)":
    ycol = COL_PARAMS
    y_title = "Parameters (B)"
else:
    ycol = bench
    y_title = bench

# =========================
# Header & KPIs
# =========================
# Apply theme CSS
if theme == "Oscuro":
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stMarkdown, .stText {
            color: #fafafa;
        }
        /* Tabla oscura */
        [data-testid="stDataFrame"] {
            background-color: #1e2130;
        }
        [data-testid="stDataFrame"] * {
            color: #fafafa !important;
            border-color: #3d4251 !important;
        }
        /* Métricas oscuras */
        [data-testid="stMetricValue"] {
            color: #fafafa;
        }
        [data-testid="stMetricLabel"] {
            color: #c4c4c4;
        }
        /* Info boxes oscuros */
        .stAlert {
            background-color: #1e2130;
            color: #fafafa;
        }
        /* Sidebar oscuro */
        [data-testid="stSidebar"] {
            background-color: #0e1117;
        }
        [data-testid="stSidebar"] * {
            color: #fafafa;
        }
        /* Selectbox/Dropdown oscuro */
        [data-baseweb="select"] {
            background-color: #1e2130 !important;
        }
        [data-baseweb="select"] * {
            color: #fafafa !important;
            background-color: #1e2130 !important;
        }
        [data-baseweb="popover"] {
            background-color: #1e2130 !important;
        }
        [role="listbox"] {
            background-color: #1e2130 !important;
        }
        [role="option"] {
            background-color: #1e2130 !important;
            color: #fafafa !important;
        }
        [role="option"]:hover {
            background-color: #3d4251 !important;
        }
        /* Inputs oscuros */
        input, textarea {
            background-color: #1e2130 !important;
            color: #fafafa !important;
            border-color: #3d4251 !important;
        }
        /* Expanders oscuros */
        [data-testid="stExpander"] {
            background-color: #1e2130;
            border-color: #3d4251;
        }
        /* Sliders oscuros */
        [data-testid="stSlider"] * {
            color: #fafafa !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #31333F;
        }
        .stMarkdown, .stText {
            color: #31333F;
        }
        /* Tabla clara */
        [data-testid="stDataFrame"] {
            background-color: #ffffff;
        }
        [data-testid="stDataFrame"] * {
            color: #31333F !important;
            border-color: #e0e0e0 !important;
        }
        /* Métricas claras */
        [data-testid="stMetricValue"] {
            color: #31333F;
        }
        [data-testid="stMetricLabel"] {
            color: #6c757d;
        }
        /* Info boxes claros */
        .stAlert {
            background-color: #f0f2f6;
            color: #31333F;
        }
        /* Sidebar claro */
        [data-testid="stSidebar"] {
            background-color: #f7f7f7;
        }
        [data-testid="stSidebar"] * {
            color: #31333F;
        }
        /* Selectbox/Dropdown claro */
        [data-baseweb="select"] {
            background-color: #ffffff !important;
        }
        [data-baseweb="select"] * {
            color: #31333F !important;
            background-color: #ffffff !important;
        }
        [data-baseweb="popover"] {
            background-color: #ffffff !important;
        }
        [role="listbox"] {
            background-color: #ffffff !important;
        }
        [role="option"] {
            background-color: #ffffff !important;
            color: #31333F !important;
        }
        [role="option"]:hover {
            background-color: #f0f2f6 !important;
        }
        /* Inputs claros */
        input, textarea {
            background-color: #ffffff !important;
            color: #31333F !important;
            border-color: #e0e0e0 !important;
        }
        /* Expanders claros */
        [data-testid="stExpander"] {
            background-color: #ffffff;
            border-color: #e0e0e0;
        }
        /* Sliders claros */
        [data-testid="stSlider"] * {
            color: #31333F !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("LLMs Timeline Dashboard")
info_bg = "#1e2130" if theme == "Oscuro" else "#f0f2f6"
info_color = "#fafafa" if theme == "Oscuro" else "#31333F"
st.markdown(f"""
<div style='padding: 10px; background-color: {info_bg}; border-radius: 5px; margin-bottom: 20px;'>
    <p style='margin: 0; color: {info_color};'>
        Visualización de la evolución de modelos LLM desde 2023.<br>
        Usa los filtros de la barra lateral para explorar. El eje X muestra la línea temporal.
    </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Modelos filtrados", f"{len(d):,}")
k2.metric("Mostrados", f"{len(d_show):,}")
k3.metric("Mediana Params", f"{d[COL_PARAMS].median():.2f}B")
k4.metric("Máx Params", f"{d[COL_PARAMS].max():.2f}B")

# % con score
if bench:
    pct = 100.0 * (d[bench].notna().mean())
    st.info(f"**Cobertura de {bench}:** {pct:.1f}% de los modelos tienen score.")

# =========================
# Colors
# =========================
def make_color_series(data: pd.DataFrame):
    # returns: (mode, values) for marker coloring
    if color_mode == "Ninguno":
        return None, None

    if color_mode in cat_cols:
        # categorical coloring
        vals = data[color_mode].fillna("Unknown").astype(str)
        return "cat", vals

    if color_mode == "Benchmark (score)" and bench:
        vals = pd.to_numeric(data[bench], errors="coerce")
        return "num", vals

    return None, None

color_type, color_vals = make_color_series(d_show)

# =========================
# Build figure
# =========================
fig = go.Figure()

# Hover content
hover_lines = [
    "<b>%{text}</b>",
    "Announced=%{x|%Y-%m-%d}",
    "Params(B)=%{customdata[0]:.3g}",
]
custom_cols = [COL_PARAMS]
if bench:
    custom_cols.append(bench)
    hover_lines.append(f"{bench}=%{{customdata[1]:.3g}}")

hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

customdata = np.stack(
    [pd.to_numeric(d_show[c], errors="coerce").to_numpy() for c in custom_cols],
    axis=1
)

# Base points
marker = dict(size=marker_size, opacity=marker_opacity)

# Apply coloring
if color_type == "num":
    marker.update(color=color_vals, showscale=True, colorbar=dict(title=color_mode))
elif color_type == "cat":
    # Plotly needs numeric color for single trace; easiest: split per category
    # We'll draw multiple traces for categories (still fast with Scattergl)
    for cat, sub in d_show.assign(_cat=color_vals).groupby("_cat", dropna=False):
        sub_custom = np.stack([pd.to_numeric(sub[c], errors="coerce").to_numpy() for c in custom_cols], axis=1)
        fig.add_trace(go.Scattergl(
            x=sub[COL_DATE],
            y=pd.to_numeric(sub[ycol], errors="coerce"),
            mode="markers",
            name=str(cat),
            text=sub[COL_MODEL],
            customdata=sub_custom,
            hovertemplate=hovertemplate,
            marker=dict(size=marker_size, opacity=marker_opacity),
        ))
else:
    fig.add_trace(go.Scattergl(
        x=d_show[COL_DATE],
        y=pd.to_numeric(d_show[ycol], errors="coerce"),
        mode="markers",
        name="Modelos",
        text=d_show[COL_MODEL],
        customdata=customdata,
        hovertemplate=hovertemplate,
        marker=marker,
    ))

# Top highlight
d_top = d_top.sort_values(COL_DATE)
top_labels = [f"{m}" for m in d_top[COL_MODEL]]  # Simplified to avoid overlap

top_y = pd.to_numeric(d_top[ycol], errors="coerce")
fig.add_trace(go.Scatter(
    x=d_top[COL_DATE],
    y=top_y,
    mode="markers+text" if label_mode != "Ninguna (solo hover)" else "markers",
    name=f"Top {top_n}" + (f" ({bench})" if (bench and y_mode == "Score (benchmark)") else ""),
    text=top_labels if label_mode != "Ninguna (solo hover)" else None,
    textposition="top center",
    textfont=dict(
        size=9,
        color="#fafafa" if theme == "Oscuro" else "#31333F"
    ),
    marker=dict(
        size=max(marker_size+3, 10),
        opacity=0.95,
        line=dict(width=1, color="#fafafa" if theme == "Oscuro" else "#31333F")
    ),
    hovertemplate="<b>%{text}</b><br>Announced=%{x|%Y-%m-%d}<br>Y=%{y:.3g}<extra></extra>",
))

# Optional: label extreme max params point too (if selected)
if label_mode == "TopN + extremos":
    extreme = d.loc[d[COL_PARAMS].idxmax()]
    fig.add_trace(go.Scatter(
        x=[extreme[COL_DATE]],
        y=[extreme[ycol] if ycol in extreme.index else extreme[COL_PARAMS]],
        mode="markers+text",
        name="Máx params",
        text=[f"MAX: {extreme[COL_MODEL]}"],
        textposition="bottom right",
        textfont=dict(
            size=10,
            color="#fafafa" if theme == "Oscuro" else "#31333F"
        ),
        marker=dict(
            size=max(marker_size+6, 12),
            opacity=0.95,
            symbol="diamond",
            line=dict(width=2, color="#fafafa" if theme == "Oscuro" else "#31333F")
        ),
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

# Layout style
template = "plotly_dark" if theme == "Oscuro" else "plotly_white"

# Theme-specific colors
title_color = "#fafafa" if theme == "Oscuro" else "#31333F"
grid_color = "#3d4251" if theme == "Oscuro" else "#e0e0e0"
paper_bgcolor = "#0e1117" if theme == "Oscuro" else "#ffffff"
plot_bgcolor = "#0e1117" if theme == "Oscuro" else "#ffffff"

fig.update_layout(
    template=template,
    height=plot_height,
    margin=dict(l=70, r=30, t=85, b=60),
    title=dict(
        text=f"LLMs (2023+) — Timeline con filtros<br><sup>Mostrando {len(d_show):,} modelos | Top {top_n} resaltado</sup>",
        font=dict(color=title_color)
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.01,
        font=dict(color=title_color)
    ),
    paper_bgcolor=paper_bgcolor,
    plot_bgcolor=plot_bgcolor,
    font=dict(color=title_color),
)

fig.update_xaxes(
    title="Announced",
    title_font=dict(color=title_color),
    tickfont=dict(color=title_color),
    rangeslider=dict(
        visible=True,
        thickness=0.02,
        bgcolor="rgba(0,0,0,0)",
        bordercolor=grid_color,
        borderwidth=1,
        yaxis=dict(rangemode="match")
    ),
    showgrid=True,
    gridcolor=grid_color,
)

# Ocultar los datos en el rangeslider
for trace in fig.data:
    trace.update(xaxis="x")

fig.update_yaxes(
    title=y_title + (" [log]" if (y_scale == "Log") else ""),
    title_font=dict(color=title_color),
    tickfont=dict(color=title_color),
    type="log" if (y_scale == "Log") else "linear",
    showgrid=True,
    gridcolor=grid_color,
)

# =========================
# Render
# =========================
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

# =========================
# Exports
# =========================
st.markdown("---")
st.subheader("Exportar Datos y Gráficos")

colA, colB = st.columns([1, 1])

with colA:
    html_bytes = fig.to_html(include_plotlyjs=True, full_html=True).encode("utf-8")
    st.download_button(
        "Descargar HTML interactivo",
        data=html_bytes,
        file_name="llms_timeline_filtrado.html",
        mime="text/html"
    )

with colB:
    st.info("**Exportar imágenes (PNG/SVG/PDF):** Instala kaleido con pip install kaleido")

# Optional image exports if kaleido is available
try:
    import kaleido  # noqa: F401
    img_format = st.selectbox("Exportar imagen", ["(no)", "png", "svg", "pdf"], index=0)
    if img_format != "(no)":
        img_bytes = fig.to_image(format=img_format, width=2000, height=1100, scale=2)
        st.download_button(
            f"Descargar {img_format.upper()}",
            data=img_bytes,
            file_name=f"llms_timeline.{img_format}",
            mime="application/octet-stream"
        )
except Exception:
    pass

# =========================
# Tables (Top + filtered)
# =========================
st.markdown("---")
st.subheader(f"Top {top_n} Modelos (criterio actual)")
top_display = d_top[[COL_DATE, COL_MODEL, COL_PARAMS] + ([bench] if bench else [])].sort_values(COL_DATE, ascending=False)
st.dataframe(top_display, width="stretch", height=400)

st.markdown("---")
st.subheader("Datos Filtrados")
show_cols = [COL_DATE, COL_MODEL, COL_PARAMS] + ([bench] if bench else []) + [c for c in cat_cols if c in d.columns]
show_cols = [c for c in show_cols if c in d.columns]

with st.expander(f"Ver tabla completa ({len(d)} registros)", expanded=False):
    st.dataframe(d[show_cols].sort_values(COL_DATE, ascending=False), width="stretch", height=400)

csv = d[show_cols].sort_values(COL_DATE).to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV (datos filtrados)",
    data=csv,
    file_name="llms_filtrado.csv",
    mime="text/csv"
)
