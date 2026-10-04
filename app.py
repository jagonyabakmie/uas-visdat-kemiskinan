"""
Dashboard: Bukan Seberapa Banyak, Tapi Seberapa Dalam
Kedalaman (P1) dan Keparahan (P2) Kemiskinan Kabupaten/Kota di Indonesia, 2025

Topik visualisasi (Soal UAS, Lampiran A):
  (a) Data berdimensi tinggi : halaman Multivariat & Profil Klaster (biplot PCA, parallel coordinates, heatmap terklaster)
  (b) Data berjaringan       : halaman Jaringan (graf force-directed, adjacency matrix, sentralitas, komunitas)
  (c) Data berhierarki       : halaman Hierarki (treemap, sunburst, drill-down, breadcrumb)
  (e) Data geospasial        : halaman Peta Sebaran & Autokorelasi Spasial (choropleth, simbol proporsional, LISA, Moran)

Jalankan:  streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.cluster.hierarchy import linkage, leaves_list

st.set_page_config(page_title="Kedalaman Kemiskinan Kabupaten/Kota 2025", layout="wide",
                   initial_sidebar_state="auto")

# =====================================================================================
# 0. KONSTANTA
# =====================================================================================
SUMBER = "Sumber: BPS (2025), diolah"
SOURCES = [
    ("Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota",
     "https://www.bps.go.id/id/statistics-table/2/NjIxIzI=/persentase-penduduk-miskin-menurut-kabupaten-kota.html"),
    ("Indeks Kedalaman Kemiskinan (P1) Menurut Kabupaten/Kota",
     "https://www.bps.go.id/id/statistics-table/2/NjIyIzI=/indeks-kedalaman-kemiskinan--p1--menurut-kabupaten-kota.html"),
    ("Indeks Keparahan Kemiskinan (P2) Menurut Kabupaten/Kota",
     "https://www.bps.go.id/id/statistics-table/2/NjIzIzI=/indeks-keparahan-kemiskinan-p2-menurut-kabupaten-kota.html"),
    ("Jumlah Penduduk Miskin (Ribu Jiwa) Menurut Kabupaten/Kota",
     "https://www.bps.go.id/en/statistics-table/2/NjE5IzI=/jumlah-penduduk-miskin--ribu-jiwa--menurut-kabupaten-kota.html"),
    ("Indeks Pembangunan Manusia 2025",
     "https://www.bps.go.id/id/publication/2026/04/24/f96755ab0e48765d028c0462/indeks-pembangunan-manusia-2025.html"),
]

LABEL = {
    "P0": "P0, persentase penduduk miskin (%)", "P1": "P1, indeks kedalaman kemiskinan",
    "P2": "P2, indeks keparahan kemiskinan", "MISKIN": "Jumlah penduduk miskin (ribu jiwa)",
    "UHH": "Umur harapan hidup (tahun)", "HLS": "Harapan lama sekolah (tahun)", "RLS": "Rata-rata lama sekolah (tahun)",
    "PENG": "Pengeluaran per kapita disesuaikan (ribu Rp)", "IPM": "Indeks Pembangunan Manusia",
    "IPM_TUMBUH": "Pertumbuhan IPM (%)", "INTENSITAS": "Intensitas kemiskinan, P1/P0×100 (%)",
}
SHORT = {"P0": "P0", "P1": "P1", "P2": "P2", "MISKIN": "Jml miskin", "UHH": "UHH", "HLS": "HLS", "RLS": "RLS",
         "PENG": "Pengeluaran", "IPM": "IPM", "IPM_TUMBUH": "Tumbuh IPM", "INTENSITAS": "Intensitas"}
PCLAB = {"P0": "P0", "P1": "P1", "P2": "P2", "MISKIN": "Miskin (log)", "UHH": "UHH", "HLS": "HLS", "RLS": "RLS",
         "PENG": "Pengeluaran", "IPM": "IPM", "IPM_TUMBUH": "Tumbuh IPM"}
VARS = ["P0", "P1", "P2", "MISKIN", "UHH", "HLS", "RLS", "PENG", "IPM", "IPM_TUMBUH"]
PULAU_ORDER = ["Sumatera", "Jawa", "Bali–Nusa Tenggara", "Kalimantan", "Sulawesi", "Maluku", "Papua"]

# ---- satu palet untuk seluruh dashboard -----------------------------------------------
# Divergen biru–merah (ColorBrewer RdBu, aman bagi buta warna): biru = rendah, merah = tinggi,
# warna netral tepat di median kab/kota. IPM dibalik (IPM tinggi = biru).
B2, B1, MID, R1, R2 = "#2166ac", "#92c5de", "#e4e0da", "#f4a582", "#b2182b"
RDBU = [[0, B2], [0.25, B1], [0.5, MID], [0.75, R1], [1, R2]]
ORD5 = [B2, B1, "#fddbc7", R1, R2]  # kelas berurutan choropleth
KLASTER_C = {"K1 · Maju & ringan": B2, "K2 · Menengah": B1, "K3 · Rentan": R1, "K4 · Miskin dalam": R2}
KLASTER_S = {"K1 · Maju & ringan": "circle", "K2 · Menengah": "square", "K3 · Rentan": "diamond", "K4 · Miskin dalam": "triangle-up"}
LISA_C = {"Tinggi–Tinggi": R2, "Tinggi–Rendah": R1, "Rendah–Tinggi": B1, "Rendah–Rendah": B2, "Tidak signifikan": "#dedcd7"}
COMM = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]  # komunitas: sengaja bukan biru/merah
GREYS = [[0, "#eef0f3"], [1, "#1f2937"]]
INK, INK2, MUTED, GRID = "#1f2937", "#4b5563", "#9ca3af", "#eceef1"
NAVY = "#1f3a5f"
FONT = "Source Sans, Source Sans Pro, sans-serif"
MAP_STYLE = "white-bg"  # gaya bawaan Plotly tanpa ubin; tetap tampil setelah mode layar penuh

# =====================================================================================
# 1. CSS (font memakai Source Sans bawaan Streamlit agar tampilan sama di semua komputer)
# =====================================================================================
st.markdown("""
<style>
.stApp {background:#f4f5f7;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {overflow:hidden !important; height:100vh;}
[data-testid="stMainBlockContainer"], .block-container {padding:14px 22px 0 22px !important; max-width:100% !important;}
[data-testid="stDecoration"], footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"],
[data-testid="stStatusWidget"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent !important; height:0 !important; min-height:0 !important; overflow:visible !important;}
.stElementContainer:has(style) {display:none !important;}
.st-key-vh {position:absolute !important; top:0; left:0; width:2px !important; height:2px !important; overflow:hidden; opacity:0; pointer-events:none;}
div[data-testid="stVerticalBlock"] {gap:10px;}
/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #e3e6eb;}
[data-testid="stSidebar"][aria-expanded="true"] {min-width:292px !important; max-width:292px !important;}
[data-testid="stSidebarHeader"] {position:relative;}
[data-testid="stSidebarCollapseButton"] {position:absolute !important; right:12px; top:50%; transform:translateY(-50%); z-index:5;}
[data-testid="stSidebarContent"] {padding:0 !important;}
[data-testid="stSidebarHeader"] {padding:12px 12px !important; margin:0 !important; width:100% !important; box-sizing:border-box; height:auto !important;}
[data-testid="stSidebarUserContent"] {padding:6px 18px 14px 18px !important; width:100% !important; box-sizing:border-box;}
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {gap:8px;}
.navlabel {font-size:.72rem; font-weight:700; color:#9ca3af; letter-spacing:.08em; padding:0 4px;}
/* tombol garis tiga untuk menyembunyikan/menampilkan sidebar */
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {display:flex !important; visibility:visible !important; opacity:1 !important;}
[data-testid="stSidebarCollapseButton"] button, button[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] button {
   width:38px !important; height:38px !important; border-radius:8px !important; background:#fff !important; border:1px solid #d9dde3 !important;
   display:flex !important; align-items:center; justify-content:center; padding:0 !important; transition:background .15s, transform .08s;}
[data-testid="stSidebarCollapseButton"] button:hover, button[data-testid="stExpandSidebarButton"]:hover, [data-testid="stExpandSidebarButton"] button:hover {background:#eef2f7 !important;}
[data-testid="stSidebarCollapseButton"] button:active, button[data-testid="stExpandSidebarButton"]:active, [data-testid="stExpandSidebarButton"] button:active {transform:scale(.94);}
[data-testid="stSidebarCollapseButton"] button > *, button[data-testid="stExpandSidebarButton"] > *, [data-testid="stExpandSidebarButton"] button > * {display:none !important;}
[data-testid="stSidebarCollapseButton"] button::before, button[data-testid="stExpandSidebarButton"]::before, [data-testid="stExpandSidebarButton"] button::before {
   content:""; display:block; width:16px; height:2px; background:#1f3a5f; border-radius:2px; box-shadow:0 -5px 0 #1f3a5f, 0 5px 0 #1f3a5f;}
[data-testid="stExpandSidebarButton"] {position:fixed !important; top:12px; left:12px; z-index:1001;}
.stApp:has([data-testid="stSidebar"][aria-expanded="false"]) .ph {padding-left:52px;}
/* menu navigasi */
[data-testid="stSidebar"] .stElementContainer:has([data-testid="stRadio"]), [data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadio"] > div, [data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stWidgetLabel"] + div {width:100% !important; max-width:100% !important;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:3px !important; width:100% !important; display:flex; flex-direction:column; align-items:stretch !important;}
[data-testid="stRadioOption"] > div > div:first-child {display:none !important;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {width:100% !important; box-sizing:border-box; margin:0; padding:9px 12px; border-radius:7px;
   cursor:pointer; transition:background .15s, color .15s;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {font-size:.92rem; color:#374151; font-weight:500; white-space:nowrap;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {background:#f1f3f6;}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background:#1f3a5f;}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {color:#ffffff; font-weight:600;}
[data-testid="stSidebar"] [data-testid="stExpander"] details {border:1px solid #e3e6eb; border-radius:8px;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {font-size:.9rem; font-weight:600; color:#374151;}
.sidemeta {font-size:.75rem; color:#9ca3af; line-height:1.45; padding:4px 4px 0 4px;}
/* ---------- logo & judul di kepala sidebar ---------- */
[data-testid="stHeaderLogo"] {display:none !important;}
[data-testid="stSidebarHeader"] {align-items:center !important;}
[data-testid="stSidebarLogo"], [data-testid="stSidebarHeader"] img {height:44px !important; width:44px !important; max-height:44px !important;}
[data-testid="stSidebarHeader"]::after {content:"Kedalaman dan keparahan kemiskinan kabupaten/kota di Indonesia";
   position:absolute; left:66px; right:58px; top:50%; transform:translateY(-50%);
   font-size:.76rem; font-weight:700; line-height:1.25; color:#1f3a5f;}
[data-testid="stSidebarHeader"] {min-height:74px !important; margin-bottom:6px; border-bottom:1px solid #eceef1;}
/* ---------- kotak angka sorotan ---------- */
.tiles {display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:10px;}
.tile {border-radius:8px; padding:10px 14px; color:#fff; min-height:78px; display:flex; flex-direction:column; justify-content:center;}
.tile .v {font-size:clamp(1.35rem, 3.2vh, 2.1rem); font-weight:700; line-height:1.15; color:#fff;}
.tile .l {font-size:clamp(.72rem, 1.45vh, .9rem); line-height:1.25; color:rgba(255,255,255,.9); margin-top:3px;}
/* ---------- judul halaman ---------- */
.ph {display:flex; justify-content:space-between; align-items:flex-end; height:50px; border-bottom:1px solid #e3e6eb; padding-bottom:8px;}
.ph h2 {font-size:1.35rem; font-weight:700; color:#111827; margin:0; padding:0; line-height:1.2; white-space:nowrap;}
.ph p {font-size:.86rem; color:#6b7280; margin:3px 0 0 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.ph .scope {font-size:.8rem; color:#6b7280; text-align:right; white-space:nowrap;}
.ph .scope b {color:#374151;}
/* ---------- kartu ---------- */
[class*="st-key-card_"] {background:#fff; border:1px solid #e3e6eb !important; border-radius:10px !important; padding:12px 16px 8px 16px !important;}
[class*="st-key-card_"] div[data-testid="stVerticalBlock"] {gap:6px;}
[class*="st-key-card_"] > div[data-testid="stVerticalBlock"] {height:100%;}
[class*="st-key-card_"] .stElementContainer:has(.foot) {margin-top:auto;}
[data-testid="stMarkdownContainer"]:has(.tiles), [data-testid="stMarkdownContainer"]:has(.brandtxt), [data-testid="stMarkdownContainer"]:has(.ctitle), [data-testid="stMarkdownContainer"]:has(.foot), [data-testid="stMarkdownContainer"]:has(.kpis),
[data-testid="stMarkdownContainer"]:has(.ph), [data-testid="stMarkdownContainer"]:has(.grid2), [data-testid="stMarkdownContainer"]:has(.note),
[data-testid="stMarkdownContainer"]:has(.meth), [data-testid="stMarkdownContainer"]:has(.brand), [data-testid="stMarkdownContainer"]:has(.navlabel),
[data-testid="stMarkdownContainer"]:has(.sidemeta) {margin-bottom:0 !important;}
.ctitle {font-size:.98rem; font-weight:700; color:#111827; line-height:1.35; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.csub {font-size:.82rem; color:#6b7280; line-height:1.4; margin-top:2px; height:2.8em; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;}
.csub.free {height:auto; display:block;}
.foot {display:flex; justify-content:space-between; gap:14px; font-size:.75rem; color:#9ca3af; border-top:1px solid #f0f1f3; padding-top:5px; line-height:1.3;}
/* ---------- KPI ---------- */
.kpis {display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:12px;}
.kpi {background:#fff; border:1px solid #e3e6eb; border-radius:10px; padding:11px 16px;}
.kpi .l {font-size:.8rem; color:#6b7280; font-weight:600; line-height:1.3;}
.kpi .v {font-size:1.5rem; font-weight:700; color:#111827; line-height:1.25; margin-top:3px;}
.kpi .s {font-size:.78rem; color:#9ca3af; line-height:1.3; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
/* ---------- teks lain ---------- */
.note {font-size:.84rem; color:#374151; line-height:1.45;}
.note b {color:#111827; font-weight:600;}
.grid2 {display:grid; grid-template-columns:1fr 1fr; gap:3px 16px; font-size:.82rem; color:#374151;}
.grid2 div {display:flex; justify-content:space-between; border-bottom:1px dotted #e5e7eb; padding:2px 0;}
.sw {display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;}
.tags {display:flex; flex-wrap:wrap; gap:5px; margin-top:4px;}
.tag {font-size:.78rem; padding:2px 8px; border-radius:5px; background:#f3f4f6; color:#374151; border:1px solid #e5e7eb; line-height:1.4;}
.meth h4 {font-size:clamp(.8rem, 1.95vh, 1.2rem); font-weight:700; color:#111827; margin:0 0 3px 0; padding:0;}
.meth p, .meth li {font-size:clamp(.72rem, 1.78vh, 1.1rem); color:#374151; line-height:1.4; margin:0;}
.meth ul {margin:0; padding-left:18px;}
.meth {display:flex; flex-direction:column; justify-content:flex-start; gap:clamp(6px, 3.6vh - 12px, 30px); overflow:hidden;}
.meth > div {margin:0;}
.meth a {color:#1f3a5f;}
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stMultiSelect"] label,
div[data-testid="stSegmentedControl"] label, div[data-testid="stTextInput"] label {font-size:.8rem !important; color:#6b7280 !important; font-weight:600;}
[class*="st-key-card_table"] [data-testid="stTextInput"] input {border:1px solid #d9dde3; border-radius:7px; background:#fff;}
[class*="st-key-card_table"] [data-testid="stTextInput"] > div > div {border:none; background:transparent;}
.stDownloadButton button {background:#2166ac !important; color:#fff !important; border:1px solid #2166ac !important; border-radius:7px;
   font-weight:600; font-size:.88rem; transition:background .15s, transform .08s, box-shadow .15s;}
.stDownloadButton button p {color:#fff !important;}
.stDownloadButton button:hover {background:#1b5690 !important; border-color:#1b5690 !important; box-shadow:0 2px 6px rgba(33,102,172,.3);}
.stDownloadButton button:active {transform:scale(.95); background:#16477a !important; box-shadow:inset 0 2px 4px rgba(0,0,0,.25);}
.stDownloadButton button:focus-visible {outline:2px solid #92c5de; outline-offset:2px;}
/* panel pengantar & tabel lima besar */
.intro {display:flex; flex-direction:column; justify-content:flex-start; gap:clamp(6px, 4.6vh - 16px, 40px); overflow:hidden;}
.intro.spread {justify-content:space-evenly;}
.intro .stats {margin:0 !important;}
.intro h1 {font-size:clamp(1.3rem, 3.3vh, 2.3rem); font-weight:700; color:#111827; line-height:1.2; margin:0; padding:0;}
.intro .lead {font-size:clamp(.88rem, 2vh, 1.3rem); color:#1f3a5f; font-weight:600; margin:4px 0 clamp(8px,1.6vh,16px) 0;}
.intro p, .intro li {font-size:clamp(.76rem, 1.74vh, 1.15rem); color:#374151; line-height:1.5; margin:0;}
.intro p b {color:#111827;}
.intro .stats {display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:clamp(6px,1.6vh,18px) 0;}
.intro .stats div {background:#f5f7fa; border:1px solid #e6e9ee; border-radius:8px; padding:clamp(7px,1.6vh,18px) 12px; font-size:clamp(.72rem,1.5vh,1rem); color:#6b7280; line-height:1.3;}
.intro .stats span {display:block; font-size:clamp(1.05rem, 2.8vh, 1.8rem); font-weight:700; color:#111827;}
.intro h3 {font-size:clamp(.86rem,2vh,1.25rem); font-weight:700; color:#111827; margin:0 0 4px 0; padding:0;}
.intro ul {margin:0; padding-left:18px;}
.intro li {margin-bottom:clamp(2px, .6vh, 8px);}
.t5 {margin-top:6px; font-size:.84rem;}
.t5 .r {min-height:24px; display:grid; grid-template-columns:18px minmax(0,1fr) 62px 30% 42px 44px; align-items:center; gap:8px;
        padding:3px 2px; border-bottom:1px solid #f1f2f4; color:#374151; line-height:1.3;}
.t5 .r.h {font-size:.74rem; color:#9ca3af; font-weight:600; border-bottom:1px solid #e5e7eb;}
.t5 .rk {color:#9ca3af;}
.t5 .nm {white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.t5 .nm em {font-style:normal; color:#9ca3af; font-size:.76rem; margin-left:6px;}
.t5 .num {text-align:right; white-space:nowrap;}
.t5 .num.b {font-weight:700; color:#111827;}
.t5 .bw {height:11px; background:#f3f4f6; border-radius:3px; overflow:hidden;}
.t5 .bw i {display:block; height:100%; background:#b2182b; border-radius:3px;}
[data-testid="stLayoutWrapper"]:has(> .st-key-colL) {flex:0 0 calc(60% - 8px) !important; width:calc(60% - 8px) !important; max-width:calc(60% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_intro) {flex:0 0 calc(48% - 8px) !important; width:calc(48% - 8px) !important; max-width:calc(48% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_scatter) {flex:0 0 calc(58% - 8px) !important; width:calc(58% - 8px) !important; max-width:calc(58% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_map) {flex:0 0 calc(72% - 8px) !important; width:calc(72% - 8px) !important; max-width:calc(72% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_lisa) {flex:0 0 calc(64% - 8px) !important; width:calc(64% - 8px) !important; max-width:calc(64% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_pca) {flex:0 0 calc(45% - 8px) !important; width:calc(45% - 8px) !important; max-width:calc(45% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_hm) {flex:0 0 calc(60% - 8px) !important; width:calc(60% - 8px) !important; max-width:calc(60% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_treemap) {flex:0 0 calc(58% - 8px) !important; width:calc(58% - 8px) !important; max-width:calc(58% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_net) {flex:0 0 calc(56% - 8px) !important; width:calc(56% - 8px) !important; max-width:calc(56% - 8px) !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-card_table) {flex:0 0 calc(54% - 8px) !important; width:calc(54% - 8px) !important; max-width:calc(54% - 8px) !important;}
@media (max-height: 670px) {
  .intro .stats span {font-size:1rem;}
  .intro .stats div {padding:5px 10px;}
  .intro h1 {font-size:1.25rem;}
  .intro .lead {margin-bottom:4px;}
}
@media (max-height: 620px) {
  .intro .howto {display:none;}
  .intro p, .intro li {font-size:.74rem; line-height:1.4;}
  .meth p, .meth li {font-size:.72rem; line-height:1.35;}
}
@media (max-width: 760px) {
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {overflow:auto !important; height:auto !important;}
  [data-testid="stMainBlockContainer"], .block-container {padding:58px 12px 24px 12px !important;}
  .ph {height:auto; flex-direction:column; align-items:flex-start; gap:4px; padding-left:0 !important;}
  .ph h2, .ph p, .ph .scope {white-space:normal; text-align:left;}
  .tiles {grid-template-columns:repeat(2, minmax(0,1fr));}
  div[data-testid="stHorizontalBlock"][class*="st-key-row_"] {flex-direction:column !important; flex-wrap:nowrap !important;
     height:auto !important; max-height:none !important; overflow:visible !important;}
  [class*="st-key-row_"] > [data-testid="stLayoutWrapper"], [data-testid="stLayoutWrapper"]:has(> [class*="st-key-"]) {
     flex:1 1 auto !important; width:100% !important; max-width:100% !important;}
  [class*="st-key-card_"], .st-key-colL {height:auto !important; max-height:none !important; width:100% !important;}
  .intro, .meth {height:auto !important; overflow:visible; padding-bottom:10px;}
  .intro .howto, .intro .menu {display:block !important;}
  .ctitle {white-space:normal;}
  .csub {height:auto; display:block;}
  .t5 .r {grid-template-columns:16px minmax(0,1fr) 54px 22% 38px 38px;}
  .hdr .meta {display:none;}
}
</style>
""", unsafe_allow_html=True)

# =====================================================================================
# 2. UKURAN LAYAR
# =====================================================================================
try:
    from streamlit_js_eval import streamlit_js_eval
    vp = streamlit_js_eval(js_expressions="[window.parent.innerHeight, window.parent.innerWidth]", key="vh")
except Exception:
    vp = None
vh, vw = (vp if isinstance(vp, (list, tuple)) and len(vp) == 2 and vp[0] else (None, None))
VH = max(600, min(int(vh) if vh else 760, 1400))
VW = int(vw) if vw else 1366
MOBILE = VW < 760                  # ponsel: kartu disusun ke bawah
MAIN_W = VW - 24 if MOBILE else VW - 292 - 44   # lebar area utama (sidebar 292 px, padding 2×22 px)

def cw(frac):
    """Lebar kartu dalam piksel (di ponsel kartu memakai lebar penuh)."""
    return int(MAIN_W if MOBILE else MAIN_W * frac)
PAGE_H = VH - 14 - 50 - 10 - 18   # tinggi area isi di bawah judul halaman
if MOBILE:
    PAGE_H = min(PAGE_H, 560)
CTRL = 72                          # tinggi baris kontrol (label + kotak pilihan + jarak)

# =====================================================================================
# 3. DATA
# =====================================================================================
@st.cache_data
def load():
    kab = pd.read_csv("data/processed/kabkota_2025.csv", dtype={"kode": str}, encoding="utf-8")
    prov = pd.read_csv("data/processed/provinsi_2025.csv", encoding="utf-8")
    edges = pd.read_csv("data/processed/network_edges.csv", encoding="utf-8")
    with open("data/processed/kabkota_simplified.geojson", encoding="utf-8") as f:
        geo = json.load(f)
    with open("data/processed/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    kab["label"] = kab.nama + ", " + kab.provinsi
    return kab, prov, edges, geo, meta

KAB, PROV, EDGES, GEO, META = load()
MED = {v: float(KAB[v].median()) for v in ["P0", "P1", "P2", "INTENSITAS", "IPM"]}

def fmt(x, d=2):
    return f"{x:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")

def color_range(var):
    return float(KAB[var].quantile(.02)), float(KAB[var].quantile(.98))

def scale_for(var):
    """Skala biru–merah dengan warna netral tepat di median kab/kota (IPM dibalik)."""
    lo, hi = color_range(var)
    m = min(max((MED[var] - lo) / (hi - lo), .05), .95)
    cols = [R2, R1, MID, B1, B2] if var == "IPM" else [B2, B1, MID, R1, R2]
    return [[0, cols[0]], [m / 2, cols[1]], [m, cols[2]], [m + (1 - m) / 2, cols[3]], [1, cols[4]]]

def layout(fig, l=8, r=8, t=6, b=6, legend=False):
    fig.update_layout(autosize=True, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=12, color=INK2), separators=",.",
                      hoverlabel=dict(bgcolor="#1f2937", font=dict(color="#fff", size=12.5, family=FONT), bordercolor="#1f2937"),
                      showlegend=legend, legend=dict(font=dict(size=11.5), bgcolor="rgba(255,255,255,.9)"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#d1d5db", linecolor="#d1d5db", tickfont=dict(size=11, color=MUTED),
                     title_font=dict(size=12, color=INK2))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#d1d5db", linecolor="#d1d5db", tickfont=dict(size=11, color=MUTED),
                     title_font=dict(size=12, color=INK2))
    return fig

CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
       "toImageButtonOptions": {"format": "png", "scale": 2}}

def page_header(title, desc):
    st.markdown(f"<div class='ph'><div><h2>{title}</h2><p>{desc}</p></div>"
                f"<div class='scope'>Cakupan: <b>{cakupan}</b><br>{SUMBER}</div></div>", unsafe_allow_html=True)

def head(t, s=None):
    st.markdown(f"<div class='ctitle'>{t}</div>" + (f"<div class='csub'>{s}</div>" if s else ""), unsafe_allow_html=True)

def foot(note=""):
    st.markdown(f"<div class='foot'><span>{SUMBER}</span><span>{note}</span></div>", unsafe_allow_html=True)

CH = 128  # tinggi "rangka" kartu: padding + judul (1 baris) + subjudul (2 baris) + footer + jarak

def chart(fig, h, key=None, **kw):
    fig.update_layout(height=max(int(h), 140))
    return st.plotly_chart(fig, config=kw.pop("config", CFG), width="stretch", key=key, **kw)

def page():
    return st.container(border=False, key="page", gap="small")

def row(key, h):
    return st.container(horizontal=True, height="content" if MOBILE else int(h), border=False, key=f"row_{key}", gap="small")

def card(key, h, w="stretch"):
    return st.container(border=True, height="content" if MOBILE else int(h), width=w, key=f"card_{key}")

# =====================================================================================
# 4. SIDEBAR: identitas, navigasi, filter
# =====================================================================================
PAGES = ["Dashboard", "Peta Sebaran", "Persentase & Intensitas", "Autokorelasi Spasial", "Multivariat", "Profil Klaster", "Hierarki", "Jaringan",
         "Data & Metode"]
st.logo("assets/logo_stis.png", size="large", icon_image="assets/logo_stis.png")
with st.sidebar:
    st.markdown("<div class='navlabel'>MENU</div>", unsafe_allow_html=True)
    hal = st.radio("Halaman", PAGES, key="hal", label_visibility="collapsed")
    with st.expander("Filter wilayah", expanded=False):
        f_pulau = st.multiselect("Pulau", PULAU_ORDER, placeholder="Semua pulau")
        prov_opts = sorted(KAB[KAB.pulau.isin(f_pulau)].provinsi.unique() if f_pulau else KAB.provinsi.unique())
        f_prov = st.multiselect("Provinsi", prov_opts, placeholder="Semua provinsi")
    st.markdown("<div class='sidemeta'>Data: Badan Pusat Statistik, 2025.</div>",
                unsafe_allow_html=True)

D = KAB.copy()
if f_pulau: D = D[D.pulau.isin(f_pulau)]
if f_prov: D = D[D.provinsi.isin(f_prov)]
if D.empty:
    st.warning("Tidak ada wilayah yang memenuhi filter. Kosongkan sebagian pilihan filter di sidebar.")
    st.stop()
cakupan = "514 kabupaten/kota" if len(D) == len(KAB) else f"{len(D)} dari 514 kabupaten/kota"

# =====================================================================================
# FUNGSI PETA
# =====================================================================================
def jenks(values, k):
    """Fisher–Jenks natural breaks."""
    x = np.sort(np.asarray(values, float)); n = len(x)
    mat1 = np.zeros((n + 1, k + 1)); mat2 = np.full((n + 1, k + 1), np.inf)
    mat1[1, 1:] = 1; mat2[1, 1:] = 0
    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1; v = x[i3 - 1]; s2 += v * v; s1 += v; w += 1
            var = s2 - s1 * s1 / w; i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    if mat2[l, j] >= var + mat2[i4, j - 1]:
                        mat1[l, j] = i3; mat2[l, j] = var + mat2[i4, j - 1]
        mat1[l, 1] = 1; mat2[l, 1] = var
    br = [x[-1]]; kk = n
    for j in range(k, 1, -1):
        br.append(x[int(mat1[kk, j]) - 2]); kk = int(mat1[kk, j] - 1)
    br.append(x[0])
    return sorted(br)

@st.cache_data
def breaks(var, method, k=5):
    v = KAB[var].values
    if method == "Natural breaks (Jenks)": return jenks(v, k)
    if method == "Kuantil": return list(np.quantile(v, np.linspace(0, 1, k + 1)))
    return list(np.linspace(v.min(), v.max(), k + 1))

def map_view(sub, map_w, map_h):
    sx, sy = max(sub.lon.max() - sub.lon.min(), .5) * 1.08 + 1, max(sub.lat.max() - sub.lat.min(), .5) * 1.12 + 1
    zoom = float(np.clip(min(np.log2(map_w * 360 / (512 * sx)), np.log2(map_h * 360 / (512 * sy))), 2.5, 7.5))
    return dict(lon=(sub.lon.min() + sub.lon.max()) / 2, lat=(sub.lat.min() + sub.lat.max()) / 2), zoom

HOVER = ("<b>%{customdata[0]}</b><br>P0: %{customdata[1]:.2f}%   P1: %{customdata[2]:.2f}   P2: %{customdata[3]:.2f}"
         "<br>Penduduk miskin: %{customdata[4]:,.2f} ribu   IPM: %{customdata[5]:.2f}<extra></extra>")

def hov(df):
    return np.stack([df.label, df.P0, df.P1, df.P2, df.MISKIN, df.IPM], -1)

def class_layer(fig, df, idx, colors, line="#ffffff", width=.4):
    """Satu lapisan choropleth untuk semua kelas (tooltip berfungsi di seluruh wilayah)."""
    n = len(colors); cs = []
    for i, c in enumerate(colors):
        cs += [[i / n, c], [(i + 1) / n, c]]
    fig.add_trace(go.Choroplethmap(geojson=GEO, locations=df.kode, z=np.asarray(idx, float) + .5, zmin=0, zmax=n,
                                   featureidkey="properties.kode", colorscale=cs, showscale=False, showlegend=False,
                                   marker=dict(line=dict(width=width, color=line), opacity=.95),
                                   customdata=hov(df), hovertemplate=HOVER))

def legend_items(fig, items):
    for name, colr in items:
        fig.add_trace(go.Scattermap(lon=[None], lat=[None], mode="markers", name=name, showlegend=True, hoverinfo="skip",
                                    marker=dict(size=12, color=colr)))

def outline(fig, df):
    fig.add_trace(go.Choroplethmap(geojson=GEO, locations=df.kode, z=[0] * len(df), featureidkey="properties.kode",
                                   colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False, showlegend=False,
                                   marker=dict(line=dict(width=3, color="#111827")), hoverinfo="skip"))

def classify(var, metode, df):
    b = breaks(var, metode)
    idx = np.clip(np.searchsorted(b[1:-1], df[var].values, side="right"), 0, 4)
    cols = ORD5[::-1] if var == "IPM" else ORD5
    items = [(f"{fmt(b[i])} – {fmt(b[i+1])}  ({(idx == i).sum()})", cols[i]) for i in range(5)]
    return idx, cols, items

def map_layout(fig, center, zoom, leg, horizontal=False):
    lg = dict(title=dict(text=leg, font=dict(size=12, color=INK)), x=.01, y=.02, yanchor="bottom",
              bgcolor="rgba(255,255,255,.94)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=11.5))
    if horizontal:
        lg.update(orientation="h", x=.5, xanchor="center", y=0, yanchor="bottom", title=dict(text=leg + " ", side="left"),
                  font=dict(size=10.5), itemwidth=30, tracegroupgap=0)
    if horizontal:  # geser peta sedikit ke atas agar tidak tertutup legenda di bawah
        center = dict(lon=center["lon"], lat=center["lat"] - 22 * 360 / (512 * 2 ** zoom))
    fig.update_layout(map=dict(style=MAP_STYLE, center=center, zoom=zoom), legend=lg)
    return layout(fig, l=0, r=0, t=0, b=0, legend=True)

MAPCFG = {"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

# =====================================================================================
# HALAMAN 1 — RINGKASAN
# =====================================================================================
RANK_C = ["#b2182b", "#d6604d", "#f4a582"]  # peringkat 1, 2, 3
RANK_REST = "#b8bcc6"                         # peringkat 4 dan seterusnya

def rank_colors(n):
    return [RANK_C[i] if i < 3 else RANK_REST for i in range(n)]

if hal == "Dashboard":
    page_header("Dashboard", "Ringkasan indikator kemiskinan kabupaten/kota di Indonesia, 2025.")
    with page():
        P_sel = PROV[PROV.provinsi.isin(D.provinsi.unique())]
        kpis = [  # (nilai, label, warna latar)
            (f"{fmt(D.MISKIN.sum() / 1000, 2)} juta", "Jumlah penduduk miskin", "#1f3a5f"),
            (f"{len(D)}", "Kabupaten/kota", "#2166ac"),
            (f"{fmt(D.P0.median())}%", "Median P0, persentase penduduk miskin", "#4a7fb5"),
            (f"{fmt(D.P1.median())}", "Median P1, indeks kedalaman", "#b2182b"),
            (f"{fmt(D.P2.median())}", "Median P2, indeks keparahan", "#c94f3d"),
            (f"{fmt(META['moran']['P1']['I'], 2)}", "Moran's I P1, autokorelasi spasial", "#5b6472"),
        ]
        st.markdown("<div class='tiles' style='margin-bottom:2px'>" + "".join(
            f"<div class='tile' style='background:{c}'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l, c in kpis)
            + "</div>", unsafe_allow_html=True)
        RH = PAGE_H - 100
        with row("r0", RH):
            with card("intro", RH):
                pap = D[D.pulau == "Papua"]
                n_hid = len(D[(D.P0 < MED["P0"]) & (D.INTENSITAS > KAB.INTENSITAS.quantile(.75))])
                n_hh = int((D.LISA_P1 == "Tinggi–Tinggi").sum())
                top_kab = D.loc[D.P1.idxmax()]
                temuan = []
                if len(pap):
                    temuan.append(f"Median P1 kab/kota di Papua <b>{fmt(pap.P1.median())}</b>, sekitar "
                                  f"{fmt(pap.P1.median() / max(D.P1.median(), 1e-9), 1)} kali median kab/kota terpilih.")
                temuan.append(f"<b>{n_hid}</b> kab/kota memiliki P0 di bawah median tetapi intensitas kemiskinan di atas kuartil ketiga.")
                temuan.append(f"<b>{n_hh}</b> kab/kota membentuk kantong P1 Tinggi–Tinggi (LISA, p &lt; 0,05).")
                temuan.append(f"P1 tertinggi di <b>{top_kab.nama}</b> ({fmt(top_kab.P1)}), dengan P0 {fmt(top_kab.P0)}%.")
                st.markdown(f"""<div class='intro spread' style='height:{"auto" if MOBILE else f"{RH - 62}px"}'>
<div><h1>Bukan Seberapa Banyak, Tapi Seberapa Dalam</h1>
<div class='lead'>Kedalaman dan keparahan kemiskinan kabupaten/kota di Indonesia</div>
<p>Dashboard ini membandingkan tiga ukuran kemiskinan BPS untuk 514 kabupaten/kota (2025):
<b>P0</b>, berapa banyak penduduk di bawah garis kemiskinan; <b>P1</b>, seberapa jauh pengeluaran mereka dari garis itu;
dan <b>P2</b>, seberapa timpang pengeluaran di antara penduduk miskin. Wilayah dengan P0 serupa bisa memiliki P1 dan P2 yang sangat berbeda.</p></div>
<div class='menu'><h3>Temuan utama</h3><ul>{''.join(f'<li>{t}</li>' for t in temuan)}</ul></div>
<div class='menu howto'><h3>Cara membaca</h3><p>Warna biru menandakan nilai rendah dan merah nilai tinggi. Gunakan menu di kiri
untuk peta, autokorelasi spasial, analisis multivariat, hierarki, jaringan, serta tabel dan unduhan data.</p></div>
</div>""", unsafe_allow_html=True)
                foot()
            with card("provbar", RH):
                head(f"Jumlah penduduk miskin menurut provinsi ({len(P_sel)} provinsi)",
                     "Ribu jiwa, 2025. Tiga provinsi teratas diberi warna berbeda.")
                t = P_sel.sort_values("MISKIN", ascending=False).reset_index(drop=True)
                t["nm"] = [f"{i + 1}. " + p.replace("Kepulauan ", "Kep. ").replace("Nusa Tenggara", "NT")
                           .replace("Daerah Istimewa ", "DI ") for i, p in enumerate(t.provinsi)]
                t["col"] = rank_colors(len(t))
                half = int(np.ceil(len(t) / 2)) if len(t) > 14 and not MOBILE else len(t)
                parts = [t.iloc[:half], t.iloc[half:]] if half < len(t) else [t]
                fig = make_subplots(rows=1, cols=len(parts), horizontal_spacing=.3 if len(parts) > 1 else 0)
                xmax = t.MISKIN.max() * 1.55
                for k, part in enumerate(parts, 1):
                    fig.add_trace(go.Bar(
                        x=part.MISKIN, y=part.nm, orientation="h", marker=dict(color=list(part.col), cornerradius=2),
                        text=[fmt(v, 1) for v in part.MISKIN], textposition="outside", textfont=dict(size=10.5 if RH < 600 else 12, color=INK),
                        cliponaxis=False, customdata=np.stack([part.provinsi, part.P0, part.P1, part.P2], -1),
                        hovertemplate="<b>%{customdata[0]}</b><br>Penduduk miskin: %{x:,.2f} ribu<br>P0: %{customdata[1]:.2f}%  "
                                      "P1: %{customdata[2]:.2f}  P2: %{customdata[3]:.2f}<extra></extra>"), row=1, col=k)
                    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10.5 if RH < 600 else 12, color=INK2), showgrid=False,
                                     tickmode="array", tickvals=list(part.nm), row=1, col=k)
                    fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, xmax], row=1, col=k)
                fig.update_layout(bargap=.25)
                chart(layout(fig, l=8, r=40, t=2, b=2), len(t) * 17 + 20 if MOBILE else RH - CH)
                foot("Angka resmi BPS tingkat provinsi")

# =====================================================================================
# HALAMAN — PERSENTASE & INTENSITAS
# =====================================================================================
if hal == "Persentase & Intensitas":
    page_header("Persentase & Intensitas", "Persentase penduduk miskin belum tentu mencerminkan seberapa dalam kemiskinan di suatu wilayah.")
    with page():
        RH = PAGE_H
        with row("r1b", RH):
            with card("scatter", RH):
                head("Persentase penduduk miskin (P0) dan intensitas kemiskinan",
                     "Ukuran lingkaran: jumlah penduduk miskin. Warna: P2. Garis putus-putus: median kab/kota.")
                lo, hi = color_range("P2")
                fig = go.Figure(go.Scatter(
                    x=D.P0, y=D.INTENSITAS, mode="markers",
                    marker=dict(size=np.sqrt(D.MISKIN) * 1.25 + 3, color=D.P2, colorscale=scale_for("P2"), cmin=lo, cmax=hi,
                                line=dict(color="#fff", width=.8), opacity=.9,
                                colorbar=dict(title=dict(text="P2", font=dict(size=12)), thickness=10, len=.7, tickfont=dict(size=11))),
                    customdata=np.stack([D.label, D.P1, D.P2, D.MISKIN, D.IPM], -1),
                    hovertemplate="<b>%{customdata[0]}</b><br>P0: %{x:.2f}%<br>P1: %{customdata[1]:.2f}  P2: %{customdata[2]:.2f}"
                                  "<br>Intensitas: %{y:.1f}%<br>Penduduk miskin: %{customdata[3]:,.2f} ribu<br>IPM: %{customdata[4]:.2f}<extra></extra>"))
                fig.add_hline(y=MED["INTENSITAS"], line=dict(color=MUTED, dash="dot", width=1))
                fig.add_vline(x=MED["P0"], line=dict(color=MUTED, dash="dot", width=1))
                for x, y, t, xa, ya in [(.01, .99, "P0 rendah, intensitas tinggi", "left", "top"),
                                        (.99, .02, "P0 tinggi, intensitas rendah", "right", "bottom"),
                                        (.01, .02, "P0 rendah, intensitas rendah", "left", "bottom")]:
                    fig.add_annotation(text=t, xref="paper", yref="paper", x=x, y=y, xanchor=xa, yanchor=ya, showarrow=False,
                                       font=dict(size=11.5, color=MUTED))
                for _, r in D.nlargest(4, "P1").iterrows():
                    fig.add_annotation(x=r.P0, y=r.INTENSITAS, text=r.nama, showarrow=True, arrowhead=0, arrowcolor=MUTED,
                                       ax=-62, ay=26, font=dict(size=11.5, color=INK), bgcolor="rgba(255,255,255,.85)")
                fig.update_xaxes(title="P0 (%)")
                fig.update_yaxes(title="Intensitas, P1/P0×100 (%)")
                chart(layout(fig, l=56, b=44), RH - CH)
                foot("Intensitas: rata-rata jarak pengeluaran penduduk miskin ke garis kemiskinan")
            with card("strip", RH):
                head("Sebaran P1 kabupaten/kota menurut pulau", "Satu titik mewakili satu kab/kota; garis hitam adalah median pulau.")
                fig = go.Figure()
                pul = [p for p in PULAU_ORDER if p in D.pulau.unique()][::-1]
                rng = np.random.default_rng(1)
                lo, hi = color_range("P1")
                for i, p in enumerate(pul):
                    s = D[D.pulau == p]
                    fig.add_trace(go.Scatter(x=s.P1, y=i + rng.uniform(-.28, .28, len(s)), mode="markers",
                                             marker=dict(size=7, color=s.P1, colorscale=scale_for("P1"), cmin=lo, cmax=hi,
                                                         line=dict(width=.5, color="#fff")),
                                             customdata=s.label, hovertemplate="<b>%{customdata}</b><br>P1: %{x:.2f}<extra></extra>"))
                    fig.add_shape(type="line", x0=s.P1.median(), x1=s.P1.median(), y0=i - .38, y1=i + .38, line=dict(color=INK, width=2.5))
                fig.update_yaxes(tickvals=list(range(len(pul))), ticktext=pul, showgrid=False, tickfont=dict(size=12, color=INK2),
                                 range=[-.6, len(pul) - .4])
                fig.update_xaxes(title="P1")
                chart(layout(fig, l=8, b=44), RH - CH)
                foot(f"Median P1 kab/kota nasional: {fmt(MED['P1'])}")

if hal == "Peta Sebaran":
    page_header("Peta Sebaran", "Sebaran indikator kemiskinan dan pembangunan manusia pada 514 kabupaten/kota.")
    with page():
        cc = st.columns([1.3, 1.25, 1.1, 1.35])
        jenis = cc[0].segmented_control("Jenis peta", ["Choropleth", "Simbol proporsional"], default="Choropleth", key="jenis") or "Choropleth"
        var = cc[1].selectbox("Indikator", ["P1", "P2", "P0", "INTENSITAS", "IPM"], key="mvar", format_func=lambda v: LABEL[v])
        metode = cc[2].selectbox("Klasifikasi", ["Natural breaks (Jenks)", "Kuantil", "Interval sama"], key="cls",
                                 disabled=jenis != "Choropleth")
        cari = cc[3].selectbox("Sorot kabupaten/kota", ["(tidak ada)"] + sorted(D.label), key="cari")
        mw = cw(.72); RH = PAGE_H - CTRL
        with row("r2", RH):
            with card("map", RH):
                sub = D[D.label == cari] if cari != "(tidak ada)" else D
                center, zoom = map_view(sub, mw - 30, RH - CH - 34)
                if cari != "(tidak ada)": zoom = 7.0
                fig = go.Figure()
                if jenis == "Choropleth":
                    idx, cols, items = classify(var, metode, D)
                    class_layer(fig, D, idx, cols)
                    legend_items(fig, items)
                    head(f"{LABEL[var].split(',')[0]} menurut kabupaten/kota, 2025",
                         f"Choropleth lima kelas, metode {metode.lower()}. Biru: rendah; merah: tinggi" +
                         (" (untuk IPM, biru berarti tinggi)." if var == "IPM" else "."))
                    leg = SHORT[var]
                else:
                    fig.add_trace(go.Choroplethmap(geojson=GEO, locations=D.kode, z=[1] * len(D), featureidkey="properties.kode",
                                                   colorscale=[[0, "#eceef1"], [1, "#eceef1"]], showscale=False, showlegend=False,
                                                   marker=dict(line=dict(width=.4, color="#ffffff")),
                                                   customdata=hov(D), hovertemplate=HOVER))
                    s = D.sort_values("MISKIN", ascending=False)
                    lo, hi = color_range(var)
                    fig.add_trace(go.Scattermap(lon=s.lon, lat=s.lat, mode="markers", showlegend=False,
                                                marker=dict(size=np.sqrt(s.MISKIN) * 1.9 + 2, color=s[var], colorscale=scale_for(var),
                                                            cmin=lo, cmax=hi, opacity=.85,
                                                            colorbar=dict(title=dict(text=SHORT[var], font=dict(size=12)), thickness=10,
                                                                          len=.45, x=.985, y=.3, tickfont=dict(size=11))),
                                                customdata=hov(s), hovertemplate=HOVER))
                    for v_ in [10, 100, 500]:
                        fig.add_trace(go.Scattermap(lon=[None], lat=[None], mode="markers", name=f"{v_} ribu jiwa",
                                                    marker=dict(size=np.sqrt(v_) * 1.9 + 2, color="#9ca3af")))
                    head("Jumlah penduduk miskin menurut kabupaten/kota, 2025",
                         f"Luas lingkaran sebanding dengan jumlah penduduk miskin; warna menunjukkan {SHORT[var]} " +
                         ("(merah rendah, biru tinggi)." if var == "IPM" else "(biru rendah, merah tinggi)."))
                    leg = "Jumlah penduduk miskin"
                if cari != "(tidak ada)":
                    outline(fig, D[D.label == cari])
                chart(map_layout(fig, center, zoom, leg, horizontal=jenis == "Choropleth"), RH - CH, config=MAPCFG)
                foot("Batas wilayah: GeoJSON kabupaten/kota (data pendukung)")
            with card("rank", RH):
                n = int(np.clip((RH - CH) / 27, 8, 22))
                asc = var == "IPM"
                t = (D.nsmallest(n, var) if asc else D.nlargest(n, var)).iloc[::-1]
                head(f"Peringkat {SHORT[var]} {'terendah' if asc else 'tertinggi'}", f"{n} kabupaten/kota dengan nilai {SHORT[var]} {'terendah' if asc else 'tertinggi'}.")
                lo, hi = color_range(var)
                fig = go.Figure(go.Bar(x=t[var], y=t.nama, orientation="h",
                                       marker=dict(color=rank_colors(len(t))[::-1], cornerradius=3),
                                       text=[fmt(v) for v in t[var]], textposition="outside", textfont=dict(size=11.5, color=INK),
                                       customdata=t.provinsi, hovertemplate="<b>%{y}</b>, %{customdata}<br>%{x:.2f}<extra></extra>"))
                fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, t[var].max() * 1.32])
                fig.update_yaxes(tickfont=dict(size=11.5, color=INK2))
                chart(layout(fig, l=8, b=4), RH - CH)
                foot()


# =====================================================================================
# HALAMAN 3 — AUTOKORELASI SPASIAL
# =====================================================================================
if hal == "Autokorelasi Spasial":
    page_header("Autokorelasi Spasial", "Apakah kab/kota dengan kemiskinan dalam cenderung berdekatan? Moran's I global dan LISA lokal.")
    with page():
        cc = st.columns([1.2, 1.4, 2.4])
        lv = cc[0].selectbox("Indikator", ["P1", "P2", "P0"], key="lvar", format_func=lambda v: LABEL[v])
        cari = cc[1].selectbox("Sorot kabupaten/kota", ["(tidak ada)"] + sorted(D.label), key="cari2")
        mi = META["moran"][lv]
        cc[2].markdown(f"<div class='note' style='padding-top:24px'>Moran's I global {lv} = <b>{fmt(mi['I'],3)}</b> "
                       f"(p = {fmt(mi['p'],3)}; 999 permutasi). Nilai positif yang tinggi berarti wilayah bertetangga memiliki nilai serupa.</div>",
                       unsafe_allow_html=True)
        mw = cw(.64); RH = PAGE_H - CTRL
        with row("r3", RH):
            with card("lisa", RH):
                head(f"Peta klaster LISA untuk {lv}, 2025",
                     "Ketetanggaan queen, bobot baku baris, 999 permutasi, p < 0,05. Merah: kantong nilai tinggi; biru: kantong nilai rendah.")
                sub = D[D.label == cari] if cari != "(tidak ada)" else D
                center, zoom = map_view(sub, mw - 30, RH - CH - 34)
                if cari != "(tidak ada)": zoom = 7.0
                fig = go.Figure()
                order = list(LISA_C)
                idx = D[f"LISA_{lv}"].map({c_: i for i, c_ in enumerate(order)}).values
                class_layer(fig, D, idx, [LISA_C[c_] for c_ in order])
                cnt = D[f"LISA_{lv}"].value_counts()
                legend_items(fig, [(f"{c_} ({cnt.get(c_, 0)})", LISA_C[c_]) for c_ in order if cnt.get(c_, 0)])
                if cari != "(tidak ada)":
                    outline(fig, D[D.label == cari])
                chart(map_layout(fig, center, zoom, f"LISA {lv}", horizontal=True), RH - CH, config=MAPCFG)
                foot("Batas wilayah: GeoJSON kabupaten/kota (data pendukung)")
            with card("moran", RH):
                head("Diagram pencar Moran", f"Sumbu x: {lv} baku; sumbu y: rata-rata {lv} baku wilayah tetangga (lag spasial).")
                zc, lc_ = f"Z_{lv}", f"LAG_{lv}"
                fig = go.Figure()
                for c_, colr in LISA_C.items():
                    s = D[D[f"LISA_{lv}"] == c_]
                    fig.add_trace(go.Scatter(x=s[zc], y=s[lc_], mode="markers", name=c_,
                                             marker=dict(size=6.5, color=colr, line=dict(width=.4, color="#fff")),
                                             customdata=s.label, hovertemplate="<b>%{customdata}</b><br>z: %{x:.2f}  lag: %{y:.2f}<extra></extra>"))
                xs = np.array([KAB[zc].min(), KAB[zc].max()])
                fig.add_trace(go.Scatter(x=xs, y=mi["I"] * xs, mode="lines", line=dict(color=INK, width=1.3), hoverinfo="skip"))
                fig.add_hline(y=0, line=dict(color="#d1d5db", width=1)); fig.add_vline(x=0, line=dict(color="#d1d5db", width=1))
                fig.update_xaxes(title=f"{lv} baku (z)", zeroline=False)
                fig.update_yaxes(title="Lag spasial", zeroline=False)
                chart(layout(fig, l=48, b=40), RH - CH - 168)
                cnt = D[f"LISA_{lv}"].value_counts()
                hh = D[D[f"LISA_{lv}"] == "Tinggi–Tinggi"].provinsi.value_counts().head(3)
                st.markdown("<div class='grid2'>" + "".join(
                    f"<div><span><span class='sw' style='background:{LISA_C[c_]}'></span>{c_}</span><b>{cnt.get(c_, 0)}</b></div>"
                    for c_ in LISA_C) + "</div>"
                    "<div class='csub free' style='margin-top:8px'>Provinsi dengan klaster Tinggi–Tinggi terbanyak</div><div class='tags'>" +
                    ("".join(f"<span class='tag'>{p} ({n})</span>" for p, n in hh.items()) or "<span class='csub'>Tidak ada pada filter ini.</span>")
                    + "</div>", unsafe_allow_html=True)
                foot("Garis: Moran's I")

# =====================================================================================
# HALAMAN 4 — MULTIVARIAT
# =====================================================================================
if hal == "Multivariat":
    page_header("Multivariat", "Sepuluh indikator kemiskinan dan IPM direduksi dengan PCA; pilih titik di biplot untuk menyorotnya di parallel coordinates.")
    with page():
        RH = PAGE_H
        with row("r4", RH):
            with card("pca", RH):
                ev = META["pca_var"]
                head(f"Biplot PCA: PC1 ({fmt(ev[0]*100,1)}%) dan PC2 ({fmt(ev[1]*100,1)}%)",
                     "Gunakan alat kotak atau laso untuk memilih wilayah. Warna dan bentuk penanda: klaster K-Means.")
                fig = go.Figure()
                for kl, col in KLASTER_C.items():
                    s = D[D.KLASTER == kl]
                    fig.add_trace(go.Scatter(x=s.PC1, y=s.PC2, mode="markers", name=f"{kl} ({len(s)})",
                                             marker=dict(color=col, symbol=KLASTER_S[kl], size=8,
                                                         line=dict(color="#6b9fc4" if kl == "K2 · Menengah" else "#fff", width=.7)),
                                             customdata=np.stack([s.label, s.P1, s.IPM, s.kode], -1),
                                             hovertemplate="<b>%{customdata[0]}</b><br>P1: %{customdata[1]:.2f}  IPM: %{customdata[2]:.2f}<extra></extra>"))
                out = D[D.PENCILAN]
                fig.add_trace(go.Scatter(x=out.PC1, y=out.PC2, mode="markers+text", name="Pencilan (jarak > P97,5)",
                                         text=out.nama, textposition="top center", textfont=dict(size=10.5, color=INK2),
                                         marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=INK, width=1.1)), hoverinfo="skip"))
                L = pd.DataFrame(META["loadings"]); sc = 8
                for v_, r in L.iterrows():
                    fig.add_annotation(x=r.PC1 * sc, y=r.PC2 * sc, ax=0, ay=0, axref="x", ayref="y", showarrow=True, arrowhead=2,
                                       arrowsize=.8, arrowwidth=1.2, arrowcolor="#4b5563", text="")
                    fig.add_annotation(x=r.PC1 * sc * 1.15, y=r.PC2 * sc * 1.15, text=SHORT[v_], showarrow=False,
                                       font=dict(size=11.5, color=INK), bgcolor="rgba(255,255,255,.8)")
                fig.update_xaxes(title="PC1 (nilai positif: kemiskinan lebih tinggi)")
                fig.update_yaxes(title="PC2")
                fig.update_layout(dragmode="lasso", legend=dict(orientation="h", x=0, y=1.01, yanchor="bottom", font=dict(size=11), bgcolor="rgba(0,0,0,0)"))
                ev_sel = chart(layout(fig, l=48, b=44, t=58, legend=True), RH - CH, key="pca_sel", on_select="rerun", selection_mode=("box", "lasso"),
                               config={"displaylogo": False, "modeBarButtonsToRemove": ["autoScale2d", "toggleSpikelines"]})
                foot("Panah: muatan (loading) variabel, diperbesar 8 kali")
            sel = set()
            try:
                for p in ev_sel.selection.points:
                    if p.get("customdata"): sel.add(str(p["customdata"][3]))
            except Exception:
                pass
            st.session_state["sel"] = sel
            with card("pc", RH):
                head("Parallel coordinates sepuluh indikator",
                     f"{len(sel)} kab/kota terpilih ditandai merah; wilayah lain abu-abu." if sel else
                     "Warna garis menunjukkan P1. Seret pada sumbu untuk menyaring rentang nilai.")
                dims = []
                for v_ in VARS:
                    vals = np.log10(D[v_] + 1) if v_ == "MISKIN" else D[v_]
                    dim = dict(label=PCLAB[v_], values=vals)
                    if v_ == "MISKIN":
                        tv = [1, 10, 100, 1000]; dim.update(tickvals=np.log10(np.array(tv) + 1), ticktext=[str(t) for t in tv])
                    dims.append(dim)
                if sel:
                    line = dict(color=D.kode.isin(sel).astype(int), colorscale=[[0, "rgba(209,213,219,.35)"], [1, R2]], cmin=0, cmax=1)
                else:
                    lo, hi = color_range("P1")
                    line = dict(color=D.P1, colorscale=scale_for("P1"), cmin=lo, cmax=hi)
                fig = go.Figure(go.Parcoords(line=line, dimensions=dims, labelfont=dict(size=11.5, color=INK), labelangle=-20,
                                             tickfont=dict(size=11, color=NAVY), rangefont=dict(size=1, color="rgba(0,0,0,0)")))
                chart(layout(fig, l=36, r=40, t=62, b=14), RH - CH)
                foot("Jumlah penduduk miskin ditampilkan dalam skala log")

# =====================================================================================
# HALAMAN 5 — PROFIL KLASTER
# =====================================================================================
if hal == "Profil Klaster":
    page_header("Profil Klaster", "Karakteristik empat klaster K-Means dan sebarannya menurut pulau.")
    with page():
        sel = st.session_state.get("sel", set()) & set(D.kode)
        RH = PAGE_H
        with row("r5", RH):
            with card("hm", RH):
                Zall = KAB[VARS].assign(MISKIN=np.log1p(KAB.MISKIN))
                Zs = (D[VARS].assign(MISKIN=np.log1p(D.MISKIN)) - Zall.mean()) / Zall.std()
                cent = Zs.groupby(D.KLASTER.values).mean().reindex([k for k in KLASTER_C if k in D.KLASTER.values])
                if sel:
                    cent.loc["Wilayah terpilih"] = Zs[D.kode.isin(sel).values].mean()
                if len(cent) >= 2:
                    cent = cent.iloc[:, leaves_list(linkage(cent.T.values, "average"))]
                head("Heatmap terklaster: profil rata-rata klaster (skor z)",
                     "Kolom diurutkan dengan klaster hierarki (average linkage). Merah: di atas rata-rata kab/kota; biru: di bawah."
                     + (" Baris terakhir: wilayah yang dipilih di halaman Multivariat." if sel else ""))
                fig = go.Figure(go.Heatmap(z=cent.values, x=[SHORT[c] for c in cent.columns], y=list(cent.index),
                                           zmid=0, zmin=-2.5, zmax=2.5, colorscale=RDBU,
                                           text=[[fmt(v, 1) for v in row_] for row_ in cent.values], texttemplate="%{text}",
                                           textfont=dict(size=13), xgap=3, ygap=3,
                                           colorbar=dict(thickness=10, len=.9, tickfont=dict(size=11), title=dict(text="z", font=dict(size=12))),
                                           hovertemplate="%{y}<br>%{x}: z = %{z:.2f}<extra></extra>"))
                fig.update_yaxes(autorange="reversed", tickfont=dict(size=12.5, color=INK2), showgrid=False)
                fig.update_xaxes(side="top", tickfont=dict(size=12, color=INK2), showgrid=False)
                chart(layout(fig, l=8, t=28, b=6), RH - CH)
                n4 = int((D.KLASTER == "K4 · Miskin dalam").sum())
                foot(f"K4 ({n4} kab/kota): P1 dan P2 tertinggi, RLS dan HLS terendah")
            with card("komp", RH):
                head("Komposisi klaster menurut pulau", "Persentase kab/kota di setiap pulau menurut klaster.")
                ct = pd.crosstab(D.pulau, D.KLASTER, normalize="index").reindex([p for p in PULAU_ORDER if p in D.pulau.unique()][::-1]) * 100
                fig = go.Figure()
                for kl, col in KLASTER_C.items():
                    if kl in ct.columns:
                        fig.add_trace(go.Bar(y=ct.index, x=ct[kl], name=kl, orientation="h", marker=dict(color=col, line=dict(color="#fff", width=1)),
                                             hovertemplate="%{y}<br>" + kl + ": %{x:.1f}%<extra></extra>"))
                fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.08, x=0, yanchor="top"))
                fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False)
                fig.update_yaxes(tickfont=dict(size=12, color=INK2))
                chart(layout(fig, l=8, b=60, legend=True), RH - CH)
                foot()

# =====================================================================================
# HALAMAN 6 — HIERARKI
# =====================================================================================
def hier_frame(Dv, size_v, color_v):
    """Warna kab/kota = nilai kab/kota; provinsi = angka resmi BPS tingkat provinsi;
    pulau & Indonesia = rata-rata provinsi tertimbang estimasi jumlah penduduk."""
    P = PROV.set_index("provinsi")
    def pval(p):
        return P.loc[p, "P1"] / P.loc[p, "P0"] * 100 if color_v == "INTENSITAS" else P.loc[p, color_v]
    sv = Dv.MISKIN if size_v == "MISKIN" else pd.Series(1, index=Dv.index)
    X = Dv.assign(_s=sv)
    rows = [(f"k{r.kode}", f"p{r.provinsi}", r.nama, r._s, r[color_v]) for _, r in X.iterrows()]
    for p, g in X.groupby("provinsi"):
        rows.append((f"p{p}", f"i{g.pulau.iloc[0]}", p, g._s.sum(), pval(p)))
    for i, g in X.groupby("pulau"):
        provs = g.provinsi.unique()
        rows.append((f"i{i}", "Indonesia", i, g._s.sum(), float(np.average([pval(p) for p in provs], weights=P.loc[provs, "PENDUDUK"]))))
    provs = Dv.provinsi.unique()
    rows.append(("Indonesia", "", "Indonesia", sv.sum(), float(np.average([pval(p) for p in provs], weights=P.loc[provs, "PENDUDUK"]))))
    return pd.DataFrame(rows, columns=["id", "parent", "label", "value", "color"])

if hal == "Hierarki":
    page_header("Hierarki", "Struktur pulau, provinsi, dan kabupaten/kota. Klik untuk masuk ke tingkat berikutnya; klik jalur di atas treemap untuk kembali.")
    with page():
        hc = st.columns([1.2, 1.2, 2.6])
        size_v = hc[0].selectbox("Ukuran", ["MISKIN", "n"], key="hsize",
                                 format_func=lambda x: "Jumlah penduduk miskin" if x == "MISKIN" else "Banyaknya kab/kota")
        color_v = hc[1].selectbox("Warna", ["P1", "P2", "P0", "INTENSITAS", "IPM"], key="hcol", format_func=lambda v: LABEL[v])
        hc[2].markdown("<div class='note' style='padding-top:24px'>Warna provinsi memakai angka resmi BPS tingkat provinsi; "
                       "warna pulau adalah rata-rata provinsi yang ditimbang dengan estimasi jumlah penduduk.</div>", unsafe_allow_html=True)
        HF = hier_frame(D, size_v, color_v)
        lo, hi = color_range(color_v)
        common = dict(ids=HF.id, parents=HF.parent, labels=HF.label, values=HF.value, branchvalues="total",
                      marker=dict(colors=HF.color, colorscale=scale_for(color_v), cmin=lo, cmax=hi, line=dict(width=1, color="#fff")),
                      hovertemplate="<b>%{label}</b><br>Ukuran: %{value:,.2f}<br>" + SHORT[color_v] + ": %{color:.2f}<extra></extra>")
        RH = PAGE_H - CTRL
        with row("r6", RH):
            with card("treemap", RH):
                head(f"Treemap: luas menurut {'jumlah penduduk miskin' if size_v=='MISKIN' else 'banyaknya kab/kota'}, warna menurut {SHORT[color_v]}",
                     "Biru: rendah; merah: tinggi" + (" (untuk IPM, biru berarti tinggi)" if color_v == "IPM" else "") + ". Titik tengah warna: median kab/kota.")
                fig = go.Figure(go.Treemap(**common, maxdepth=3, pathbar=dict(visible=True, thickness=24, textfont=dict(size=12.5)),
                                           texttemplate="<b>%{label}</b><br>%{value:,.0f}", textfont=dict(size=12.5)))
                fig.update_traces(marker_showscale=True, marker_colorbar=dict(title=dict(text=SHORT[color_v], font=dict(size=12)),
                                                                              thickness=10, len=.7, tickfont=dict(size=11)))
                chart(layout(fig, l=0, r=0, t=2, b=2), RH - CH)
                foot()
            with card("sunburst", RH):
                head("Sunburst: struktur yang sama dalam bentuk radial", "Cincin dalam: pulau; tengah: provinsi; luar: kabupaten/kota.")
                fig = go.Figure(go.Sunburst(**common, maxdepth=3, insidetextorientation="radial", textfont=dict(size=11.5)))
                chart(layout(fig, l=0, r=0, t=2, b=2), RH - CH)
                foot()

# =====================================================================================
# HALAMAN 7 — JARINGAN
# =====================================================================================
def eig(G):
    out = {n: 0.0 for n in G}
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        if H.number_of_edges() == 0: continue
        try:
            e = nx.eigenvector_centrality(H, weight="weight", max_iter=5000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            e = dict(H.degree(weight="weight"))
        out.update({n: v * len(comp) / len(G) for n, v in e.items()})
    return out

@st.cache_data
def build_net(thr, pul):
    nodes = PROV[PROV.pulau.isin(pul)] if pul else PROV
    E = EDGES[(EDGES.weight >= thr) & EDGES.source.isin(nodes.provinsi) & EDGES.target.isin(nodes.provinsi)]
    G = nx.Graph(); G.add_nodes_from(nodes.provinsi)
    for _, r in E.iterrows():
        G.add_edge(r.source, r.target, weight=r.weight, dist=1 / r.weight)
    comms = nx.community.louvain_communities(G, weight="weight", seed=1) if G.number_of_edges() else [{n} for n in G]
    singles = [c for c in comms if len(c) == 1]
    P1p = PROV.set_index("provinsi").P1
    comms = sorted([c for c in comms if len(c) > 1], key=lambda c: -np.mean([P1p[n] for n in c]))
    cid = {n: min(i, 6) for i, c in enumerate(comms) for n in c}
    cid.update({n: 6 for c in singles for n in c})
    cent = dict(Strength=dict(G.degree(weight="weight")), Betweenness=nx.betweenness_centrality(G, weight="dist"), Eigenvector=eig(G))
    mod = nx.community.modularity(G, comms + singles, weight="weight") if G.number_of_edges() else 0
    return G, cid, cent, mod, len(comms)

if hal == "Jaringan":
    page_header("Jaringan", "Provinsi dihubungkan bila profil sepuluh indikatornya mirip; komunitas menunjukkan kelompok provinsi yang serupa.")
    with page():
        nc = st.columns([1.3, 1.1, 1.4, 2.2])
        thr = nc[0].slider("Ambang kemiripan edge", 0.10, 0.95, 0.60, 0.05, key="thr")
        cmetric = nc[1].selectbox("Ukuran node", ["Strength", "Betweenness", "Eigenvector"], key="cm",
                                  format_func=lambda x: {"Strength": "Degree tertimbang", "Betweenness": "Betweenness",
                                                         "Eigenvector": "Eigenvector"}[x])
        focus = nc[2].selectbox("Sorot provinsi dan tetangganya", ["(tidak ada)"] + sorted(PROV.provinsi), key="focus")
        G, cid, cent, mod, ncom = build_net(thr, tuple(f_pulau))
        nc[3].markdown(f"<div class='note' style='padding-top:8px'>{G.number_of_nodes()} provinsi, <b>{G.number_of_edges()}</b> edge aktif "
                       f"(kernel Gauss, {META['knn']} tetangga terdekat). Komunitas Louvain: <b>{ncom}</b>; modularitas Q = <b>{fmt(mod,3)}</b>.</div>",
                       unsafe_allow_html=True)
        PV = PROV.set_index("provinsi"); AB = PV.singkatan.to_dict()
        nbr = set(G.neighbors(focus)) if focus in G else set()
        RH = PAGE_H - CTRL
        with row("r7", RH):
            with card("net", RH):
                head("Graf kemiripan antarprovinsi (tata letak force-directed)",
                     "Tebal garis: bobot kemiripan. Ukuran node: sentralitas terpilih. Warna: komunitas, diurutkan dari rata-rata P1 tertinggi.")
                fig = go.Figure()
                for u, v, w in G.edges(data=True):
                    on = focus == "(tidak ada)" or focus in (u, v)
                    fig.add_trace(go.Scatter(x=[PV.x[u], PV.x[v]], y=[PV.y[u], PV.y[v]], mode="lines", hoverinfo="skip", showlegend=False,
                                             line=dict(width=.5 + 3.5 * (w["weight"] - thr) / max(1 - thr, .05),
                                                       color=("rgba(31,58,95,.8)" if focus in (u, v) else "rgba(107,114,128,.28)") if on
                                                       else "rgba(209,213,219,.15)")))
                nodes = list(G)
                cv = np.array([cent[cmetric][n] for n in nodes]); cvn = (cv - cv.min()) / (np.ptp(cv) or 1)
                size = dict(zip(nodes, 11 + 15 * cvn))
                for c in range(7):
                    ns = [n for n in nodes if cid[n] == c]
                    if not ns: continue
                    fig.add_trace(go.Scatter(
                        x=[PV.x[n] for n in ns], y=[PV.y[n] for n in ns], mode="markers+text",
                        name=f"Komunitas {c+1}" if c < 6 else "Lainnya/terisolasi",
                        text=[AB[n] for n in ns], textposition="bottom center", textfont=dict(size=11, color=INK),
                        marker=dict(size=[size[n] for n in ns], color=COMM[c] if c < 6 else "#b8bcc6",
                                    opacity=[1 if focus == "(tidak ada)" or n == focus or n in nbr else .2 for n in ns],
                                    line=dict(color="#fff", width=1.2)),
                        customdata=np.stack([ns, [PV.P0[n] for n in ns], [PV.P1[n] for n in ns], [PV.IPM[n] for n in ns],
                                             [G.degree(n) for n in ns], [cent[cmetric][n] for n in ns]], -1),
                        hovertemplate="<b>%{customdata[0]}</b><br>P0 %{customdata[1]:.2f}%   P1 %{customdata[2]:.2f}   IPM %{customdata[3]:.2f}"
                                      "<br>Degree: %{customdata[4]}   " + cmetric + ": %{customdata[5]:.3f}<extra></extra>"))
                fig.update_xaxes(visible=False, range=[PROV.x.min() - .12, PROV.x.max() + .12])
                fig.update_yaxes(visible=False, range=[PROV.y.min() - .15, PROV.y.max() + .1])
                fig.update_layout(legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom"))
                chart(layout(fig, l=4, t=30, b=2, legend=True), RH - CH)
                foot("Label: singkatan nama provinsi")
            with card("adj", RH):
                head("Adjacency matrix diurutkan menurut komunitas", "Sel berisi bobot edge aktif; kotak berwarna menandai batas komunitas.")
                ordn = sorted(G, key=lambda n: (cid[n], -cent["Strength"][n]))
                M = np.full((len(ordn), len(ordn)), np.nan)
                for u, v, w in G.edges(data=True):
                    i, j = ordn.index(u), ordn.index(v); M[i, j] = M[j, i] = w["weight"]
                lab = [AB[n] for n in ordn]
                fig = go.Figure(go.Heatmap(z=M, x=lab, y=lab, colorscale=GREYS, zmin=thr, zmax=1, xgap=1, ygap=1,
                                           colorbar=dict(thickness=9, len=.8, tickfont=dict(size=10.5), title=dict(text="bobot", font=dict(size=11))),
                                           hovertemplate="%{y} – %{x}<br>kemiripan: %{z:.3f}<extra></extra>"))
                start = 0
                for c in range(7):
                    sz = sum(1 for n in ordn if cid[n] == c)
                    if sz and c < 6:
                        fig.add_shape(type="rect", x0=start - .5, x1=start + sz - .5, y0=start - .5, y1=start + sz - .5,
                                      line=dict(color=COMM[c], width=2))
                    start += sz
                if focus in ordn:
                    i = ordn.index(focus)
                    fig.add_shape(type="rect", x0=-.5, x1=len(ordn) - .5, y0=i - .5, y1=i + .5, line=dict(color=NAVY, width=1.5))
                fig.update_yaxes(autorange="reversed", tickfont=dict(size=9.5), showgrid=False)
                fig.update_xaxes(tickfont=dict(size=9.5), tickangle=-90, showgrid=False)
                chart(layout(fig, l=4, t=4, b=2), RH - CH)
                foot()

# =====================================================================================
# HALAMAN 8 — DATA & METODE
# =====================================================================================
if hal == "Data & Metode":
    page_header("Data & Metode", "Tabel data hasil olahan, sumber, metode analisis, dan keterbatasan.")
    with page():
        RH = PAGE_H
        with row("r8", RH):
            with card("table", RH):
                q1, q2, q3 = st.columns([1.6, 1, 1], vertical_alignment="center")
                q = q1.text_input("Cari", placeholder="Cari kabupaten/kota atau provinsi", label_visibility="collapsed")
                T = D if not q else D[D.label.str.contains(q, case=False)]
                cols = ["kode", "nama", "provinsi", "pulau", "P0", "P1", "P2", "MISKIN", "INTENSITAS", "IPM", "UHH", "HLS",
                        "RLS", "PENG", "KLASTER", "LISA_P1"]
                q2.download_button("CSV kab/kota", KAB.to_csv(index=False).encode("utf-8"), "kabkota_kemiskinan_2025.csv", "text/csv", width="stretch",
                                   on_click=lambda: st.toast("CSV kabupaten/kota sedang diunduh"))
                q3.download_button("CSV provinsi", PROV.to_csv(index=False).encode("utf-8"), "provinsi_kemiskinan_2025.csv", "text/csv", width="stretch",
                                   on_click=lambda: st.toast("CSV provinsi sedang diunduh"))
                st.dataframe(T[cols], hide_index=True, height=int(RH - 104), width="stretch", column_config={
                    "kode": "Kode", "nama": "Kabupaten/kota", "provinsi": "Provinsi", "pulau": "Pulau",
                    "P0": st.column_config.NumberColumn("P0 (%)", format="%.2f"), "P1": st.column_config.NumberColumn("P1", format="%.2f"),
                    "P2": st.column_config.NumberColumn("P2", format="%.2f"),
                    "MISKIN": st.column_config.NumberColumn("Miskin (ribu)", format="%.2f"),
                    "INTENSITAS": st.column_config.NumberColumn("Intensitas (%)", format="%.1f"),
                    "IPM": st.column_config.NumberColumn("IPM", format="%.2f"), "UHH": st.column_config.NumberColumn("UHH", format="%.2f"),
                    "HLS": st.column_config.NumberColumn("HLS", format="%.2f"), "RLS": st.column_config.NumberColumn("RLS", format="%.2f"),
                    "PENG": st.column_config.NumberColumn("Pengeluaran", format="%d"), "KLASTER": "Klaster", "LISA_P1": "LISA P1"})
                foot(f"{len(T)} baris ditampilkan")
            with card("meta", RH):
                src = "".join(f"<li><a href='{u}' target='_blank'>{t}</a></li>" for t, u in SOURCES)
                st.markdown(f"""<div class='meth' style='height:{"auto" if MOBILE else f"{RH - 62}px"}'>
<div><h4>Sumber data</h4><ul>{src}</ul>
<p style='color:#6b7280;margin-top:3px'>Tahun data 2025; diakses 3 Oktober 2026. Batas wilayah: GeoJSON kab/kota (data pendukung non-BPS).</p></div>
<div><h4>Pengolahan</h4><p>Lima tabel BPS (514 kab/kota) digabung per baris; tujuh nama wilayah diselaraskan dengan berkas batas wilayah; tidak ada nilai hilang.
Agregasi P0 dan jumlah penduduk miskin per provinsi sama dengan angka provinsi BPS.</p></div>
<div><h4>Metode analisis</h4><p>Moran's I dan LISA (ketetanggaan <i>queen</i>, 999 permutasi); PCA dan K-Means (k = 4) pada sepuluh indikator baku;
pencilan dari jarak Mahalanobis; jaringan kemiripan provinsi (kernel Gauss, 5 tetangga terdekat) dengan komunitas Louvain.</p></div>
<div><h4>Rancangan visual</h4><p>Skala biru–merah ColorBrewer RdBu yang aman bagi buta warna; median kab/kota sebagai titik tengah.
Choropleth untuk rasio dan indeks; jumlah absolut dengan simbol proporsional.</p></div>
<div><h4>Keterbatasan</h4><p>P1 dan P2 kab/kota berbasis Susenas sehingga galat baku relatif dapat tinggi di wilayah kecil; data satu periode;
jaringan menggambarkan kemiripan, bukan aliran.</p></div>
</div>""", unsafe_allow_html=True)
                foot()
