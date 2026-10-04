# Bukan Seberapa Banyak, Tapi Seberapa Dalam

Dashboard interaktif tentang **kedalaman (P1) dan keparahan (P2) kemiskinan di 514 kabupaten/kota Indonesia**. Dibuat untuk UAS Visualisasi Data dan Informasi, Politeknik Statistika STIS, TA 2025/2026.

**Demo:** `https://<nama-app>.streamlit.app` *(isi setelah deploy)*

Di laptop, dashboard tampil dalam satu layar tanpa scroll; di ponsel kartu disusun ke bawah. Navigasi halaman ada di sidebar kiri yang bisa disembunyikan; filter wilayah (pulau, provinsi) berupa tombol di kanan atas setiap halaman. Setiap halaman memuat interpretasi yang berubah mengikuti filter. Tinggi kartu, grafik, dan ukuran huruf menyesuaikan ukuran layar secara otomatis.

## Halaman dan topik visualisasi (Lampiran A)

| Halaman | Topik | Teknik | Interaksi |
|---|---|---|---|
| Dashboard | Pengantar | Kotak angka sorotan, ringkasan dan interpretasi, jumlah penduduk miskin 38 provinsi | Tooltip |
| Peta Sebaran | **Geospasial** (514 kab/kota) | Choropleth (Jenks/kuantil/interval sama), simbol proporsional, peringkat | Tooltip, zoom/pan, legenda, pilihan layer, sorot wilayah |
| Autokorelasi Spasial | **Geospasial** | Peta klaster LISA, diagram pencar Moran, Moran's I | Pilihan indikator, sorot wilayah |
| Persentase & Intensitas | Cerita utama | Bubble scatter P0 vs intensitas, dot-strip P1 per pulau | Tooltip |
| Multivariat | **Berdimensi tinggi** (10 variabel × 514 unit) | Biplot PCA, parallel coordinates | Brushing (kotak/laso) di PCA yang langsung tersorot di parallel coordinates |
| Profil Klaster | **Berdimensi tinggi** | Heatmap terklaster (K-Means + klaster hierarki), komposisi klaster per pulau | Baris "Wilayah terpilih" mengikuti seleksi di halaman Multivariat |
| Hierarki | **Berhierarki** (Pulau → Provinsi → Kab/Kota) | Treemap, sunburst; ukuran dan warna memakai dua variabel berbeda | Drill-down, breadcrumb |
| Jaringan | **Berjaringan** (38 node, edge berbobot) | Graf force-directed, adjacency matrix, sentralitas, komunitas Louvain | Filter ambang bobot edge, sorot tetangga |
| Data & Metode | — | Tabel, sumber, metode | Pencarian, unduh CSV |

## Data

**Sumber utama: Badan Pusat Statistik (BPS).** Tahun data 2025. Tanggal akses: 3 Oktober 2026.

1. Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota, 2025: https://www.bps.go.id/id/statistics-table/2/NjIxIzI=/persentase-penduduk-miskin-menurut-kabupaten-kota.html
2. Indeks Kedalaman Kemiskinan (P1) Menurut Kabupaten/Kota, 2025: https://www.bps.go.id/id/statistics-table/2/NjIyIzI=/indeks-kedalaman-kemiskinan--p1--menurut-kabupaten-kota.html
3. Indeks Keparahan Kemiskinan (P2) Menurut Kabupaten/Kota, 2025: https://www.bps.go.id/id/statistics-table/2/NjIzIzI=/indeks-keparahan-kemiskinan-p2-menurut-kabupaten-kota.html
4. Jumlah Penduduk Miskin (Ribu Jiwa) Menurut Kabupaten/Kota, 2025: https://www.bps.go.id/en/statistics-table/2/NjE5IzI=/jumlah-penduduk-miskin--ribu-jiwa--menurut-kabupaten-kota.html
5. Indeks Pembangunan Manusia 2025 (UHH, HLS, RLS, pengeluaran per kapita, IPM, pertumbuhan IPM): https://www.bps.go.id/id/publication/2026/04/24/f96755ab0e48765d028c0462/indeks-pembangunan-manusia-2025.html

**Data pendukung (non-BPS):** batas wilayah kab/kota (`data/raw/kabkota.geojson`, 515 fitur; dicocokkan melalui nama kab/kota dan provinsi).

## Struktur

```
├── app.py                  # aplikasi Streamlit
├── prep.py                 # pra-pemrosesan (dapat direproduksi)
├── requirements.txt
├── assets/logo_stis.png    # logo kampus
├── .streamlit/config.toml  # tema
└── data/
    ├── raw/                # tabel BPS asli (.xlsx) + GeoJSON
    └── processed/          # hasil prep.py (dipakai app.py)
```

## Pra-pemrosesan (`prep.py`)

- Lima tabel BPS digabung per baris. Urutan barisnya identik; empat nama berbeda ejaan.
- Nama disamakan dengan GeoJSON, misalnya Toba Samosir → Toba, Ndunga → Nduga, Maluku Tenggara Barat → Kep. Tanimbar, dan kota administrasi Jakarta.
- Pengeluaran per kapita berformat Indonesia ("11.191") dikonversi ke angka (ribu Rp).
- GeoJSON: dua fitur Sorong (kode 92.01) digabung, geometri disederhanakan, presisi 3 desimal, ring diorientasikan ulang.
- Variabel turunan: **Intensitas = P1/P0 × 100**, yaitu rata-rata jarak pengeluaran penduduk miskin ke garis kemiskinan (%).
- Ketetanggaan *queen*. Wilayah kepulauan tanpa tetangga dihubungkan ke 2 tetangga terdekat. Moran's I dan LISA dihitung dengan 999 permutasi.
- PCA atas 10 variabel baku (jumlah miskin di-log), K-Means k=4 yang diurutkan menurut P1, dan pencilan berdasarkan jarak Mahalanobis.
- Jaringan provinsi: kemiripan exp(−d²/2σ²) pada 10 indikator baku, 5-NN.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
python prep.py          # opsional: membuat ulang data/processed dari data/raw
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push folder ini ke repositori GitHub **publik**.
2. Buka <https://share.streamlit.io>, pilih **Create app**, lalu pilih repo, branch `main`, dan file `app.py`.
3. Salin URL aplikasi ke README ini dan ke bagian akhir makalah.

## Pilihan desain

- **Warna:** seluruh besaran memakai satu skala divergen biru–merah (ColorBrewer RdBu) yang aman bagi pembaca buta warna. Biru berarti rendah dan merah berarti tinggi, dengan median kab/kota sebagai titik tengah. Untuk IPM arahnya dibalik (IPM tinggi = biru). Kelas LISA dan klaster K-Means memakai skema yang sama. Komunitas jaringan memakai warna kategorikal yang bukan biru/merah, dan adjacency matrix memakai skala abu-abu, agar tidak tertukar makna.
- **Choropleth** hanya untuk rasio/indeks. Angka absolut (jumlah penduduk miskin) ditampilkan dengan simbol proporsional.
- **Treemap/sunburst:** warna provinsi memakai angka resmi BPS tingkat provinsi. Warna pulau dan Indonesia adalah rata-rata provinsi tertimbang estimasi jumlah penduduk (jumlah penduduk miskin / P0).
- Setiap grafik mencantumkan "Sumber: BPS (2025), diolah" di bagian bawah kartu.

## Validasi pengolahan

- Agregasi P0 dan jumlah penduduk miskin dari 514 kab/kota sama dengan angka provinsi BPS (selisih maksimum 0,008 poin).
- P0 nasional hasil agregasi tertimbang = 8,47%, sama dengan angka BPS 2025. Total penduduk miskin = 23,85 juta jiwa.
- P1 dan P2 provinsi BPS tidak sama dengan rata-rata tertimbang kab/kota (estimasi langsung pada tingkat provinsi), sehingga dashboard memakai angka provinsi resmi untuk tingkat provinsi.

## Keterbatasan

P1 dan P2 tingkat kab/kota berbasis Susenas, sehingga RSE bisa tinggi untuk wilayah kecil. Data hanya satu titik waktu. Jaringan yang ditampilkan adalah jaringan *kemiripan*, bukan aliran fisik.

## Deklarasi alat bantu AI

Claude (Anthropic) dipakai sebagai alat bantu penulisan kode. Seluruh hasil telah diperiksa dan dipahami oleh penulis.
