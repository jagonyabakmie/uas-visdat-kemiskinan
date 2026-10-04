"""
prep.py — Pra-pemrosesan data BPS untuk dashboard
"Bukan Seberapa Banyak, Tapi Seberapa Dalam: Peta Kedalaman & Keparahan Kemiskinan 514 Kabupaten/Kota, 2025"

Input  : data/raw/*.xlsx (tabel BPS), data/raw/kabkota.geojson (batas kab/kota, data pendukung non-BPS)
Output : data/processed/*.csv, *.json, kabkota_simplified.geojson

Jalankan:  python prep.py
"""
import json, re
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.strtree import STRtree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

RAW, OUT = "data/raw/", "data/processed/"
RNG = np.random.default_rng(2025)

# ------------------------------------------------------------------ 1. Baca tabel BPS
def load(fname, skip):
    return pd.read_excel(RAW + fname, header=None, dtype=str).iloc[skip:].reset_index(drop=True)

p0 = load("Persentase Penduduk Miskin (P0) Menurut Kabupaten_Kota, 2025.xlsx", 3)
p1 = load("Indeks Kedalaman Kemiskinan (P1) Menurut Kabupaten_Kota, 2025.xlsx", 3)
p2 = load("Indeks Keparahan Kemiskinan (P2) Menurut Kabupaten_Kota, 2025.xlsx", 3)
nm = load("Number of Poor Population (Thousand People) by Regency_City, 2025.xlsx", 3)
ipm = load("IPM 2025.xlsx", 1)  # baris terakhir = INDONESIA

# urutan baris kelima tabel identik (dicek: hanya beda ejaan 4 nama)
assert len(p0) == len(p1) == len(p2) == len(nm) == len(ipm) - 1

def num(s):
    s = str(s).strip()
    return None if s in ("-", "nan", "", "…") else float(s.replace(",", "."))

def ribu(s):
    """Pengeluaran per kapita ditulis format Indonesia '11.191' (ribu Rp). '6.3' -> 6300, '11' -> 11000."""
    s = str(s).strip()
    a, _, b = s.partition(".")
    v = float(a + (b + "000")[:3]) if b else float(a)
    return v * 1000 if v < 100 else v

rows, prov = [], None
for i in range(len(p0)):
    name = str(p0.iloc[i, 0]).strip()
    is_prov = name.upper() == name
    if is_prov:
        prov = name
    r = ipm.iloc[i]
    rows.append(dict(
        nama_bps=name, nama_ipm=str(r[0]).strip(), prov_bps=prov, level="prov" if is_prov else "kab",
        P0=num(p0.iloc[i, 1]), P1=num(p1.iloc[i, 1]), P2=num(p2.iloc[i, 1]), MISKIN=num(nm.iloc[i, 1]),
        UHH=num(r[1]), HLS=num(r[2]), RLS=num(r[3]), PENG=ribu(r[4]), IPM=num(r[5]), IPM_TUMBUH=num(r[6]),
    ))
df = pd.DataFrame(rows)
nasional = ipm.iloc[-1]
NAS_IPM = dict(UHH=num(nasional[1]), HLS=num(nasional[2]), RLS=num(nasional[3]), PENG=ribu(nasional[4]),
               IPM=num(nasional[5]), IPM_TUMBUH=num(nasional[6]))
assert df.isna().sum().sum() == 0

# ------------------------------------------------------------------ 2. Batas wilayah + kode BPS
g = json.load(open(RAW + "kabkota.geojson", encoding="utf-8"))
# geojson memuat 2 fitur kode 92.01 (Sorong; satu fragmen berlabel Papua Barat) -> digabung
feat = {}
for f in g["features"]:
    k = f["properties"]["kode"]
    geom = shape(f["geometry"]).buffer(0)
    if k in feat:
        feat[k]["geom"] = unary_union([feat[k]["geom"], geom])
        if f["properties"]["prov"] == "Papua Barat Daya":
            feat[k]["props"] = f["properties"]
    else:
        feat[k] = dict(geom=geom, props=f["properties"])

def norm(s):
    s = s.lower().replace("kepulauan", "kep").replace("adm.", "").replace("adm ", "")
    return re.sub(r"[^a-z]", "", s)

ALIAS = {  # nama di tabel BPS -> nama di geojson
    "toba samosir": "toba", "kota padangsidimpuan": "kota padangsidempuan",
    "kepulauan seribu": "kep seribu", "siau tagulandang biaro": "kep siau tagulandang biaro",
    "maluku tenggara barat": "kep tanimbar", "ndunga": "nduga", "mamuju utara": "pasangkayu",
}
gi = {}
for k, v in feat.items():
    p = v["props"]
    nmx = p["nama"] if (not p["kota"] or p["nama"].lower().startswith("kota")) else "Kota " + p["nama"]
    gi[(norm(nmx), norm(p["prov"]))] = k

PROV_FIX = {"KEP. BANGKA BELITUNG": "Kepulauan Bangka Belitung", "D I YOGYAKARTA": "Daerah Istimewa Yogyakarta",
            "DKI JAKARTA": "DKI Jakarta"}
def prov_title(s):
    return PROV_FIX.get(s, s.title())

kab = df[df.level == "kab"].copy()
kab["provinsi"] = kab.prov_bps.map(prov_title)
kab["nama"] = kab.nama_ipm.replace({"Ndunga": "Nduga"})
codes = []
for _, r in kab.iterrows():
    key = ALIAS.get(r.nama_bps.lower(), r.nama_bps)
    key = norm(key)
    if key.startswith("kotajakarta"):
        key = key.replace("kotajakarta", "kotajakarta")
    c = gi.get((key, norm(r.provinsi)))
    if c is None:  # Jakarta: 'Kota Adm. Jakarta X'
        c = next((v for (n, p), v in gi.items() if p == norm(r.provinsi) and n.endswith(key.replace("kota", ""))), None)
    codes.append(c)
kab["kode"] = codes
assert kab.kode.notna().all() and kab.kode.is_unique, kab[kab.kode.isna()]
kab["kode_prov"] = kab.kode.str[:2].astype(int)

def pulau(kp):
    if kp < 30: return "Sumatera"
    if kp < 50: return "Jawa"
    if kp < 60: return "Bali–Nusa Tenggara"
    if kp < 70: return "Kalimantan"
    if kp < 80: return "Sulawesi"
    if kp < 90: return "Maluku"
    return "Papua"
kab["pulau"] = kab.kode_prov.map(pulau)
kab["tipe"] = np.where(kab.nama.str.startswith("Kota"), "Kota", "Kabupaten")

# variabel turunan: intensitas = P1/P0*100 = rata-rata kesenjangan pengeluaran penduduk miskin thd GK (%)
kab["INTENSITAS"] = (kab.P1 / kab.P0 * 100).round(2)
# estimasi jumlah penduduk (ribu jiwa) = jumlah penduduk miskin / P0; dipakai sebagai bobot agregasi
kab["PENDUDUK"] = (kab.MISKIN * 100 / kab.P0).round(2)
kab["KETIMPANGAN"] = (kab.P2 / kab.P1).round(3)  # makin besar = makin timpang di antara penduduk miskin

# ------------------------------------------------------------------ 3. Geometri: sederhanakan + centroid + ketetanggaan
geoms = {k: feat[k]["geom"] for k in kab.kode}
def cw(geom):
    """Plotly/d3-geo butuh ring luar searah jarum jam (kebalikan RFC 7946)."""
    if isinstance(geom, Polygon): return orient(geom, sign=-1.0)
    if isinstance(geom, MultiPolygon): return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
    polys = [p for p in getattr(geom, "geoms", []) if isinstance(p, Polygon)]
    return MultiPolygon([orient(p, sign=-1.0) for p in polys])
simp = {k: cw(g_.simplify(0.012, preserve_topology=True)) for k, g_ in geoms.items()}
kab["lon"] = [geoms[k].representative_point().x for k in kab.kode]
kab["lat"] = [geoms[k].representative_point().y for k in kab.kode]

codes = list(kab.kode)
tree = STRtree([geoms[c].buffer(0.002) for c in codes])
nbrs = {c: set() for c in codes}
for i, c in enumerate(codes):
    for j in tree.query(geoms[c].buffer(0.002), predicate="intersects"):
        if j != i:
            nbrs[c].add(codes[j])
# wilayah kepulauan tanpa tetangga -> 2 tetangga terdekat (centroid)
xy = kab[["lon", "lat"]].to_numpy()
for i, c in enumerate(codes):
    if not nbrs[c]:
        d = np.hypot(*(xy - xy[i]).T); d[i] = np.inf
        for j in np.argsort(d)[:2]:
            nbrs[c].add(codes[j]); nbrs[codes[j]].add(c)

# ------------------------------------------------------------------ 4. Autokorelasi spasial (Moran's I & LISA) — P1
n = len(codes); idx = {c: i for i, c in enumerate(codes)}
W = np.zeros((n, n))
for c, s in nbrs.items():
    for t in s:
        W[idx[c], idx[t]] = 1
W = W / W.sum(1, keepdims=True)  # row-standardized

def moran(x, perms=999):
    z = (x - x.mean()) / x.std()
    I = (z @ W @ z) / n
    sims = np.array([(zp @ W @ zp) / n for zp in (RNG.permutation(z) for _ in range(perms))])
    p = (np.sum(sims >= I) + 1) / (perms + 1)
    lag = W @ z
    Ii = z * lag
    # permutasi bersyarat untuk LISA
    nb_count = (W > 0).sum(1)
    p_loc = np.ones(n)
    for i in range(n):
        k = nb_count[i]
        others = np.delete(z, i)
        samp = np.array([others[RNG.choice(n - 1, k, replace=False)].mean() for _ in range(perms)])
        Isim = z[i] * samp
        p_loc[i] = (np.sum(np.abs(Isim) >= abs(Ii[i])) + 1) / (perms + 1)
    quad = np.where(z > 0, np.where(lag > 0, "Tinggi–Tinggi", "Tinggi–Rendah"),
                    np.where(lag > 0, "Rendah–Tinggi", "Rendah–Rendah"))
    cls = np.where(p_loc < 0.05, quad, "Tidak signifikan")
    return float(I), float(p), cls, p_loc, z, lag

moran_meta = {}
for var in ["P0", "P1", "P2"]:
    I, p, cls, ploc, z, lag = moran(kab[var].to_numpy())
    kab[f"LISA_{var}"] = cls
    kab[f"LISA_{var}_p"] = ploc.round(4)
    kab[f"Z_{var}"] = z.round(4); kab[f"LAG_{var}"] = lag.round(4)
    moran_meta[var] = dict(I=round(I, 4), p=round(p, 4))
    print(f"Moran's I {var}: {I:.3f} (p={p:.3f})")

# ------------------------------------------------------------------ 5. Multivariat: PCA + K-Means
VARS = ["P0", "P1", "P2", "MISKIN", "UHH", "HLS", "RLS", "PENG", "IPM", "IPM_TUMBUH"]
X = kab[VARS].copy(); X["MISKIN"] = np.log1p(X["MISKIN"])  # jumlah miskin sangat menceng -> log
Z = StandardScaler().fit_transform(X)
pca = PCA(n_components=4, random_state=0).fit(Z)
pcs = pca.transform(Z)
for i in range(3):
    kab[f"PC{i+1}"] = pcs[:, i].round(4)
loadings = pd.DataFrame(pca.components_[:2].T, index=VARS, columns=["PC1", "PC2"])
if loadings.loc["P1", "PC1"] < 0:  # orientasi: PC1 positif = makin miskin
    kab["PC1"] *= -1; loadings["PC1"] *= -1; pcs[:, 0] *= -1
km = KMeans(n_clusters=4, n_init=50, random_state=0).fit(Z)
lab = km.labels_
# urutkan klaster menurut rata-rata P1 (0 = paling ringan)
order = pd.Series(kab.P1.values).groupby(lab).mean().sort_values().index.tolist()
remap = {old: new for new, old in enumerate(order)}
NAMA_KLASTER = ["K1 · Maju & ringan", "K2 · Menengah", "K3 · Rentan", "K4 · Miskin dalam"]
kab["KLASTER"] = [NAMA_KLASTER[remap[l]] for l in lab]
d = np.sqrt(((pcs[:, :4] / pcs[:, :4].std(0)) ** 2).sum(1))
kab["JARAK_PCA"] = d.round(3)
kab["PENCILAN"] = d > np.quantile(d, 0.975)
print("Varian dijelaskan:", pca.explained_variance_ratio_.round(3))

# ------------------------------------------------------------------ 6. Provinsi + jaringan kemiripan
pv = df[df.level == "prov"].copy()
pv["provinsi"] = pv.prov_bps.map(prov_title)
pv = pv.merge(kab.groupby("provinsi").agg(kode_prov=("kode_prov", "first"), pulau=("pulau", "first"),
                                          n_kab=("kode", "size")).reset_index(), on="provinsi")
pv["INTENSITAS"] = (pv.P1 / pv.P0 * 100).round(2)
pv["PENDUDUK"] = (pv.MISKIN * 100 / pv.P0).round(2)
SINGKATAN = {"Aceh": "ACEH", "Sumatera Utara": "SUMUT", "Sumatera Barat": "SUMBAR", "Riau": "RIAU", "Jambi": "JAMBI",
    "Sumatera Selatan": "SUMSEL", "Bengkulu": "BENGKULU", "Lampung": "LAMPUNG", "Kepulauan Bangka Belitung": "BABEL",
    "Kepulauan Riau": "KEPRI", "DKI Jakarta": "DKI", "Jawa Barat": "JABAR", "Jawa Tengah": "JATENG",
    "Daerah Istimewa Yogyakarta": "DIY", "Jawa Timur": "JATIM", "Banten": "BANTEN", "Bali": "BALI",
    "Nusa Tenggara Barat": "NTB", "Nusa Tenggara Timur": "NTT", "Kalimantan Barat": "KALBAR", "Kalimantan Tengah": "KALTENG",
    "Kalimantan Selatan": "KALSEL", "Kalimantan Timur": "KALTIM", "Kalimantan Utara": "KALTARA", "Sulawesi Utara": "SULUT",
    "Sulawesi Tengah": "SULTENG", "Sulawesi Selatan": "SULSEL", "Sulawesi Tenggara": "SULTRA", "Gorontalo": "GORONTALO",
    "Sulawesi Barat": "SULBAR", "Maluku": "MALUKU", "Maluku Utara": "MALUT", "Papua Barat": "PABAR",
    "Papua Barat Daya": "PBD", "Papua": "PAPUA", "Papua Selatan": "PAPSEL", "Papua Tengah": "PAPTENG",
    "Papua Pegunungan": "PAPEG"}
pv["singkatan"] = pv.provinsi.map(SINGKATAN)
assert pv.singkatan.notna().all()
XP = pv[VARS].copy(); XP["MISKIN"] = np.log1p(XP["MISKIN"])
ZP = StandardScaler().fit_transform(XP)
D = np.sqrt(((ZP[:, None, :] - ZP[None, :, :]) ** 2).sum(-1))
sigma = np.median(D[np.triu_indices_from(D, 1)])
S = np.exp(-(D ** 2) / (2 * sigma ** 2))
K = 5
G = nx.Graph()
names = pv.provinsi.tolist()
for i, a in enumerate(names):
    G.add_node(a)
    for j in np.argsort(-S[i])[1:K + 1]:
        G.add_edge(a, names[j], weight=round(float(S[i, j]), 4))
edges = pd.DataFrame([(u, v, w["weight"]) for u, v, w in G.edges(data=True)], columns=["source", "target", "weight"])
# tata letak graf: Kamada-Kawai pada jarak Euclid baku (graf lengkap), lalu pemisahan node yang terlalu rapat
Gc = nx.complete_graph(names)
pos = nx.kamada_kawai_layout(Gc, dist={a: {b: float(D[i, j]) ** 0.5 for j, b in enumerate(names)} for i, a in enumerate(names)})
P = np.array([pos[n] for n in names]); P = (P - P.mean(0)) / np.abs(P - P.mean(0)).max()
for _ in range(400):  # jarak minimum antarnode agar label tidak bertumpuk
    moved = False
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            dv = P[j] - P[i]; d = np.hypot(*dv)
            if d < 0.2:
                push = (0.2 - d) / 2 * (dv / (d + 1e-9)); P[i] -= push; P[j] += push; moved = True
    if not moved: break
pv["x"] = [P[names.index(n)][0] for n in pv.provinsi]; pv["y"] = [P[names.index(n)][1] for n in pv.provinsi]
pd.DataFrame(S, index=names, columns=names).round(4).to_csv(OUT + "provinsi_similarity.csv")
print("Jaringan:", G.number_of_nodes(), "node,", G.number_of_edges(), "edge")

# ------------------------------------------------------------------ 7. Simpan
keep = ["kode", "nama", "tipe", "provinsi", "kode_prov", "pulau"] + VARS + ["INTENSITAS", "KETIMPANGAN", "PENDUDUK", "lon", "lat",
        "PC1", "PC2", "PC3", "KLASTER", "JARAK_PCA", "PENCILAN"] + \
       [c for c in kab.columns if c.startswith(("LISA_", "Z_", "LAG_"))]
kab[keep].sort_values("kode").to_csv(OUT + "kabkota_2025.csv", index=False)
pv[["kode_prov", "provinsi", "singkatan", "pulau", "n_kab"] + VARS + ["INTENSITAS", "PENDUDUK", "x", "y"]].sort_values("kode_prov").to_csv(
    OUT + "provinsi_2025.csv", index=False)
edges.to_csv(OUT + "network_edges.csv", index=False)

fc = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "id": k, "properties": {"kode": k}, "geometry": mapping(simp[k])} for k in kab.kode]}
txt = json.dumps(fc, separators=(",", ":"))
txt = re.sub(r"(\d+\.\d{3})\d+", r"\1", txt)  # presisi 3 desimal (~100 m)
open(OUT + "kabkota_simplified.geojson", "w", encoding="utf-8").write(txt)

meta = dict(moran=moran_meta, pca_var=pca.explained_variance_ratio_.round(4).tolist(),
            loadings=loadings.round(4).to_dict(), nasional_ipm=NAS_IPM, sigma=round(float(sigma), 4), knn=K,
            klaster_centroid=pd.DataFrame(Z, columns=VARS).groupby(kab.KLASTER.values).mean().round(3).to_dict("index"))
json.dump(meta, open(OUT + "meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("Selesai. kab/kota:", len(kab), "| ukuran geojson:", round(len(txt) / 1e6, 2), "MB")
