import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import json
import requests


# Ambil GeoJSON
with open("indonesia.geojson", "r", encoding="utf-8") as f:
    geojson = json.load(f)


# Load data
df = pd.read_csv("Dataset Final Project.csv")

# Cleaning
for col in ["Rasio Peminat"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# Handle missing values
prospek_cols = [
    "Prospek Kerja 1",
    "Prospek Kerja 2",
    "Prospek Kerja 3",
    "Prospek Kerja 4",
]
df[prospek_cols] = df[prospek_cols].fillna("")

# —– SIDEBAR CUSTOM STYLING —–
st.sidebar.markdown(
    """
    <style>
    /* Sidebar background dan font */
    .css-1d391kg {  /* class sidebar container, bisa berubah tiap Streamlit versi */
        background-color: #f0f2f6;
    }
    /* Judul utama sidebar */
    .sidebar .sidebar-content .css-1avcm0n h3 {
        color: white;
        background-color: #4B8BBE;
        padding: 12px 8px;
        text-align: center;
        border-radius: 5px;
        margin-bottom: 16px;
    }
    /* Radio button label styling */
    .stRadio > div > label {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul
st.sidebar.markdown("### 📚 MENU UTAMA")
# Navigasi Analisis pakai radio (ganti selectbox)
analisis = st.sidebar.radio(
    "",
    (
        "1. Metadata & EDA",
        "2. Permintaan & Daya Saing",
        "3. Persebaran Lokasi Prodi",
        "4. Kinerja Universitas",
        "5. Jenjang Pendidikan",
        "6. Gaji & Prospek Kerja",
        "7. Prodi PTN Sepi Tapi Potensial",
        "8. Segmentasi Kategori & Kelompok",
    ),
)

st.sidebar.markdown("---")

# Filter Bidang Ilmu
st.sidebar.markdown("### 🔍 FILTER BIDANG ILMU")
bidang_choice = st.sidebar.radio(
    "",  # kosongkan label, judul sudah di atas
    ("Semua", "Saintek", "Soshum"),
    index=0,
    key="bidang_radio",
)

# Terapkan filter
df_filtered = df.copy()
if bidang_choice != "Semua":
    df = df_filtered[df_filtered["Bidang Ilmu"] == bidang_choice]


if analisis == "1. Metadata & EDA":
    st.title("📁 Metadata & Exploratory Data Analysis (EDA) Mendalam")
    st.markdown(
        """ <p style='font-size: 16px; text-align: justify;'>
    Selain metadata dan distribusi dasar, di sini kita akan menyelami: <ol> <li>Persebaran kategori & jenjang</li> <li>Boxplot untuk grup–grup penting</li> <li>Heatmap korelasi variabel numerik</li> <li>Identifikasi outlier dan skewness</li> <li>Top outlier prodi</li> </ol> </p>
    """,
        unsafe_allow_html=True,
    )

    # 📄 Metadata Dataset
    with st.expander("📋 Metadata Dataset"):
        st.subheader("Struktur & Informasi Dataset")
        st.markdown(
            """
            <p style='font-size: 15px; text-align: justify;'>
            Pada bagian ini ditampilkan jumlah baris & kolom, tipe data masing-masing kolom, serta jumlah nilai kosong per kolom. 
            Informasi ini penting untuk memahami bentuk dan kualitas data sebelum dilakukan analisis lebih lanjut.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"📊 Jumlah Data: {df.shape[0]} baris, {df.shape[1]} kolom")

        st.write("📌 Tipe Data Kolom:")
        st.dataframe(
            df.dtypes.reset_index().rename(columns={"index": "Kolom", 0: "Tipe Data"})
        )

        st.write("📌 Jumlah Nilai Kosong:")
        st.dataframe(
            df.isnull()
            .sum()
            .reset_index()
            .rename(columns={"index": "Kolom", 0: "Jumlah Null"})
        )
        st.markdown("**🔍 Deskripsi Fungsi Tiap Kolom**")
        col_funcs = {
            "NO": "Nomor urut baris data",
            "KODE": "Kode univ atau prodi (jika ada)",
            "Nama Prodi": "Nama program studi",
            "JENJANG": "Jenjang pendidikan (S1, D3, dst.)",
            "DAYA TAMPUNG 2025": "Kapasitas penerimaan mahasiswa tahun 2025",
            "PEMINAT 2024": "Jumlah pendaftar tahun 2024",
            "Rasio Peminat": "PEMINAT 2024 ÷ DAYA TAMPUNG 2025, mengukur kompetisi",
            "JENIS PORTOFOLIO": "Tipe seleksi portofolio (jika berlaku)",
            "KAB / KOTA": "Lokasi kabupaten/kota kampus",
            "PROVINSI-1": "Provinsi utama kampus",
            "PROVINSI-2": "Provinsi kedua (jika kampus punya cabang)",
            "Universitas": "Nama perguruan tinggi penyelenggara",
            "SITUS_WEB": "Alamat website resmi prodi/univ",
            "Kelompok": "Klasifikasi kelompok prodi (mis. Sains, Humaniora)",
            "Kategori": "Kategori ilmu (mis. Teknik, Ekonomi, dsb.)",
            "Gaji Minimal": "Gaji awal rata‑rata lulusan (estimasi)",
            "Gaji Maksimal": "Gaji maksimal rata‑rata lulusan (estimasi)",
            "Prospek Kerja 1": "Prospek kerja utama lulusan",
            "Prospek Kerja 2": "Prospek kerja alternatif",
            "Prospek Kerja 3": "Prospek kerja tambahan",
            "Prospek Kerja 4": "Prospek kerja lainnya",
            "Hasil": "Hasil clustering / segmentasi (jika ada)",
            "Bidang Ilmu": "Bidang ilmu utama prodi",
        }
        # tampilkan dua kolom agar lebih ringkas
        items = list(col_funcs.items())
        col1, col2 = st.columns(2)
        for i, (col, desc) in enumerate(items):
            target = col1 if i % 2 == 0 else col2
            with target:
                st.markdown(f"- **{col}**: {desc}")

    # 📉 Distribusi Rasio Peminat
    with st.expander("📉 Distribusi Rasio Peminat"):
        st.subheader("Distribusi Rasio Peminat (Daya Saing)")
        st.markdown(
            """
            <p style='font-size: 15px; text-align: justify;'>
            Rasio peminat dihitung dari jumlah peminat dibagi daya tampung. Nilai ini menunjukkan tingkat kompetitif tiap program studi. 
            Histogram ini memperlihatkan sebaran nilai rasio tersebut di seluruh prodi.
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig2 = px.histogram(
            df, x="Rasio Peminat", nbins=40, title="Histogram Rasio Peminat Prodi"
        )
        st.plotly_chart(fig2)

    # 🔗 Korelasi Peminat vs Daya Tampung
    with st.expander("🔗 Korelasi Peminat vs Daya Tampung"):
        st.subheader("Korelasi Peminat dan Daya Tampung")
        st.markdown(
            """
            <p style='font-size: 15px; text-align: justify;'>
            Scatter plot ini menunjukkan hubungan antara jumlah peminat dan daya tampung di masing-masing program studi. 
            Visualisasi ini berguna untuk melihat apakah program dengan daya tampung besar cenderung menarik lebih banyak peminat.
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig3 = px.scatter(
            df,
            x="DAYA TAMPUNG 2025",
            y="PEMINAT 2024",
            color="Kategori",
            hover_data=["Nama Prodi", "Universitas"],
            title="Scatter Plot: Daya Tampung vs Peminat",
        )
        fig3.update_layout(
            xaxis_title="Daya Tampung 2025", yaxis_title="Jumlah Peminat 2024"
        )
        st.plotly_chart(fig3)

    # 💡 Insight Awal
    with st.expander("💡 Insight Awal"):
        st.subheader("Insight Awal dari Data")
        st.markdown(
            """
            <ul style='font-size: 15px;'>
                <li>Beberapa program studi memiliki peminat jauh lebih tinggi dibanding daya tampung yang tersedia, menandakan kompetisi ketat.</li>
                <li>Distribusi jumlah peminat tidak merata — ada beberapa program studi sangat populer, tapi banyak juga prodi dengan peminat sedikit.</li>
                <li>Terdapat hubungan positif antara daya tampung dan jumlah peminat, walaupun tidak terlalu kuat — artinya prodi dengan daya tampung besar belum tentu paling diminati.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    # 1. Persebaran Kategori & Jenjang
    with st.expander("📊 Persebaran Kategori & Jenjang"):
        st.subheader("Countplot Kategori & Jenjang")
        st.markdown(
            """
            Menunjukkan jumlah program studi per kategori dan jenjang (S1, D3, dst.).
            Memudahkan melihat proporsi tiap kelompok relatif terhadap keseluruhan data.
            """,
            unsafe_allow_html=True,
        )

        # Kategori
        df_cat = (
            df["Kategori"]
            .value_counts()
            .rename_axis("Kategori")
            .reset_index(name="Count")
        )
        fig_cat = px.bar(
            df_cat,
            x="Kategori",
            y="Count",
            title="Jumlah Prodi per Kategori",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # Jenjang (note: kolom bernama 'JENJANG', bukan 'Jenjang')
        df_jenjang = (
            df["JENJANG"]
            .value_counts()
            .rename_axis("JENJANG")
            .reset_index(name="Count")
        )
        fig_jenjang = px.bar(
            df_jenjang,
            x="JENJANG",
            y="Count",
            title="Jumlah Prodi per Jenjang",
        )
        st.plotly_chart(fig_jenjang, use_container_width=True)

    # 2. Boxplot Peminat & Rasio per Jenjang
    with st.expander("📦 Boxplot Peminat & Rasio per Jenjang"):
        st.subheader("Boxplot Jumlah Peminat & Rasio by Jenjang")
        st.markdown(
            """
            Boxplot memperlihatkan median, kuartil, dan outlier distribusi:
            - Jumlah Peminat 2024 di tiap jenjang  
            - Rasio Peminat (daya saing) di tiap jenjang  
            """,
            unsafe_allow_html=True,
        )
        fig_box1 = px.box(
            df, x="JENJANG", y="PEMINAT 2024", title="Boxplot Peminat 2024 per Jenjang"
        )
        st.plotly_chart(fig_box1, use_container_width=True)

        fig_box2 = px.box(
            df,
            x="JENJANG",
            y="Rasio Peminat",
            title="Boxplot Rasio Peminat per Jenjang",
        )
        st.plotly_chart(fig_box2, use_container_width=True)

    # 3. Heatmap Korelasi Numerik
    with st.expander("🔗 Heatmap Korelasi Variabel Numerik"):
        st.subheader("Heatmap Korelasi")
        st.markdown(
            """
            Visualisasi korelasi Pearson antar variabel numerik:
            - PEMINAT 2024  
            - DAYA TAMPUNG 2025  
            - Rasio Peminat  
            
            Korelasi membantu mengetahui variabel mana yang cenderung naik-turun bersama.
            """,
            unsafe_allow_html=True,
        )
        corr = df[["PEMINAT 2024", "DAYA TAMPUNG 2025", "Rasio Peminat"]].corr()
        fig_heatmap, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig_heatmap)

    # 5. Ringkasan Insight Mendalam
    with st.expander("💡 Insight Mendalam"):
        st.subheader("Insight Tambahan")
        st.markdown(
            """
            <ul style='font-size:15px;'>
              <li>Distribusi kategori & jenjang membantu mengidentifikasi area fokus riset atau kebijakan.</li>
              <li>Boxplot menegaskan keberadaan banyak outlier—program studi flagship sangat mendominasi.</li>
              <li>Korelasi tinggi antara peminat & daya tampung (>{corr.loc['PEMINAT 2024','DAYA TAMPUNG 2025']:.2f}),  
                  menandakan institusi besar cenderung populer.</li>
              <li>Skewness >1 menunjukkan distribusi sangat memanjang ke kanan,  
                  artinya hanya segelintir prodi yang benar-benar memiliki peminat/rasio ekstrem.</li>
            </ul>
            """.replace(
                "{corr.loc['PEMINAT 2024','DAYA TAMPUNG 2025']:.2f}",
                f"{corr.loc['PEMINAT 2024','DAYA TAMPUNG 2025']:.2f}",
            ),
            unsafe_allow_html=True,
        )

    # 📈 Persentase Kategori untuk Kolom Kategorikal
    with st.expander("📊 Persentase Kategori"):
        st.subheader("Distribusi Persentase Kategori")
        st.markdown(
            """
            <p style='font-size:15px; text-align: justify;'>
            Untuk setiap kolom kategorikal, tampilkan persentase tiap nilai terhadap total baris.
            Gunakan bar chart horizontal untuk memudahkan perbandingan.
            </p>
            """,
            unsafe_allow_html=True,
        )
        # Identifikasi kolom kategorikal
        cat_cols = [
            c
            for c in df.select_dtypes(include=["object"]).columns
            if df[c].nunique() < 15
        ]
        for col in cat_cols:
            counts = (
                df[col].value_counts(normalize=True).mul(100).round(1).reset_index()
            )
            counts.columns = [col, "Persentase"]
            fig_pct = px.bar(
                counts,
                x="Persentase",
                y=col,
                orientation="h",
                title=f"Persentase Nilai pada `{col}`",
                text="Persentase",
            )
            fig_pct.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_pct, use_container_width=True)

# --- 2. Permintaan & Daya Saing ---
elif analisis == "2. Permintaan & Daya Saing":
    st.title("📊 Analisis Permintaan & Daya Saing Program Studi")
    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Analisis ini menyajikan wawasan mengenai permintaan dan daya saing program studi di Indonesia. Dengan menggunakan berbagai filter dan visualisasi, analisis ini mengidentifikasi kelompok program studi yang paling kompetitif dan paling sepi peminat, serta melakukan analisis klaster untuk memahami hubungan antara rasio peminat dan daya tampung. Pengguna dapat mengeksplorasi detail prodi dalam kelompok tertentu dan mendapatkan pemahaman lebih dalam mengenai daya saing tiap kelompok.
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Filter kategori dengan checkbox per kategori dalam 3 kolom
    st.subheader("Pilih Kategori:")
    kategori_terpilih = []

    kategori_list = df["Kategori"].dropna().unique()
    kolom = st.columns(3)  # Ganti angka ini sesuai kebutuhan kolom

    for i, kategori in enumerate(kategori_list):
        with kolom[i % 3]:  # Bagi merata ke 3 kolom
            if st.checkbox(kategori, value=True):
                kategori_terpilih.append(kategori)

    # Slider minimum peminat
    min_peminat = st.slider("Minimum Peminat:", 0, int(df["PEMINAT 2024"].max()), 0)

    # Filter data
    df_filtered = df[
        (df["Kategori"].isin(kategori_terpilih)) & (df["PEMINAT 2024"] >= min_peminat)
    ]

    # Hitung jumlah peminat per kelompok
    kelompok_data = (
        df_filtered.groupby("Kelompok")
        .agg(
            {"PEMINAT 2024": "sum", "DAYA TAMPUNG 2025": "sum", "Rasio Peminat": "mean"}
        )
        .reset_index()
    )
    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Distribusi Peminat:

    Grafik distribusi peminat menunjukkan bagaimana peminat tersebar di antara berbagai program studi. Ini bisa memberi gambaran mengenai program studi yang lebih populer dibandingkan yang lainnya.
   </p>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Distribusi Peminat")
    fig = px.histogram(df_filtered, x="PEMINAT 2024", nbins=50)
    st.plotly_chart(fig)

    # 📌 Pilihan Top-N kelompok berdasarkan rasio peminat
    st.subheader("🏆 Kelompok Paling Kompetitif & Paling Sepi Peminat")
    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Kelompok Paling Kompetitif & Paling Sepi Peminat:

    Analisis kelompok dengan rasio peminat tertinggi memberikan wawasan tentang program studi yang paling kompetitif, di mana peminat jauh lebih banyak dibandingkan daya tampung.

    Di sisi lain, kelompok dengan rasio peminat terendah menunjukkan program studi yang kurang diminati, mungkin karena faktor kualitas atau relevansi program tersebut.
    </p>
    """,
        unsafe_allow_html=True,
    )
    top_n = st.slider("Tentukan jumlah Top-N:", 5, 30, 10)

    top_kompetitif = kelompok_data.sort_values(
        by="Rasio Peminat", ascending=False
    ).head(top_n)
    st.markdown(f"**Top {top_n} Kelompok Paling Kompetitif (Rasio Tertinggi):**")
    st.dataframe(top_kompetitif)

    top_n_sepi = st.slider(
        "Tentukan jumlah Top-N Sepi Peminat:", 5, 30, 10, key="slider_top_sepi"
    )
    sepi_kompetitif = kelompok_data.sort_values(
        by="Rasio Peminat", ascending=True
    ).head(top_n_sepi)
    st.markdown(f"**Top {top_n_sepi} Kelompok Paling Sepi Peminat (Rasio Terendah):**")
    st.dataframe(sepi_kompetitif)

    # 📌 Cluster Analysis
    st.subheader("🧭 Cluster Analysis")
    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Analisis Klaster:

    Klasterisasi berdasarkan rasio peminat dan daya tampung memberikan gambaran yang lebih jelas tentang bagaimana program studi dikelompokkan. Misalnya, klaster dengan daya tampung besar dan rasio peminat tinggi menandakan program studi yang sangat kompetitif.
    </p>
    """,
        unsafe_allow_html=True,
    )
    cluster_data = kelompok_data[["Rasio Peminat", "DAYA TAMPUNG 2025"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(cluster_data)
    kelompok_data["Cluster"] = kmeans.labels_

    fig_cluster = px.scatter(
        kelompok_data,
        x="DAYA TAMPUNG 2025",
        y="Rasio Peminat",
        color="Cluster",
        hover_data=["Kelompok"],
        title="Cluster Berdasarkan Rasio Peminat & Daya Tampung",
    )
    fig_cluster.update_layout(
        xaxis_title="Daya Tampung 2025", yaxis_title="Rasio Peminat"
    )
    st.plotly_chart(fig_cluster)

    # 📌 Detail Prodi dari Kelompok
    st.subheader("🔍 Lihat Daftar Prodi dari Kelompok")
    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Eksplorasi Detail Prodi:

    Pengguna dapat menelusuri prodi dalam kelompok tertentu untuk melihat program studi mana yang memiliki daya saing tinggi, berdasarkan jumlah peminat dan daya tampung. Hal ini berguna bagi calon mahasiswa untuk memilih prodi yang tepat sesuai dengan tingkat persaingan dan daya tampung yang ada.
    </p>
    """,
        unsafe_allow_html=True,
    )
    selected_kelompok_detail = st.selectbox(
        "Pilih Kelompok untuk Lihat Detail Prodi:", options=kelompok_data["Kelompok"]
    )
    if st.button("Lihat Detail Prodi di Kelompok Ini"):
        detail_prodi = df_filtered[df_filtered["Kelompok"] == selected_kelompok_detail][
            [
                "Nama Prodi",
                "Universitas",
                "DAYA TAMPUNG 2025",
                "PEMINAT 2024",
                "Rasio Peminat",
            ]
        ].sort_values(by="Rasio Peminat", ascending=False)

        st.write(f"**Daftar Prodi di Kelompok {selected_kelompok_detail}:**")
        st.dataframe(detail_prodi)

# --- 3. Persebaran Lokasi Prodi ---
elif analisis == "3. Persebaran Lokasi Prodi":
    st.title("📍 Persebaran Lokasi Program Studi")
    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Analisis ini menggambarkan distribusi program studi berdasarkan kota dan jumlah peminat berdasarkan provinsi. Tabel pertama menunjukkan jumlah program studi di setiap kota, sedangkan heatmap selanjutnya menggambarkan jumlah peminat di setiap provinsi di Indonesia, dengan informasi tambahan mengenai daya tampung di masing-masing provinsi.
    </p>
    """,
        unsafe_allow_html=True,
    )

    # --- Jumlah Prodi per Kota ---
    st.subheader("Jumlah Program Studi per Kota")

    # Bersihkan kolom kota
    df["KAB / KOTA"] = df["KAB / KOTA"].astype(str).str.strip().str.title()

    # Agregasi jumlah prodi
    kota_count = (
        df.groupby("KAB / KOTA", dropna=True).size().reset_index(name="Jumlah Prodi")
    )

    # Tampilkan data
    st.dataframe(
        kota_count.sort_values("Jumlah Prodi", ascending=False),
        use_container_width=True,
    )

    # --- Jumlah Peminat per Provinsi ---
    st.subheader("Heatmap Jumlah Peminat per Provinsi")

    # Bersihkan nama provinsi
    df["PROVINSI-1"] = df["PROVINSI-1"].astype(str).str.strip().str.title()

    # Agregasi per provinsi
    provinsi = (
        df.groupby("PROVINSI-1", dropna=True)
        .agg({"PEMINAT 2024": "sum", "DAYA TAMPUNG 2025": "sum"})
        .reset_index()
    )

    # Tambahkan kolom custom hovertext agar tampil lebih informatif
    provinsi["hover"] = (
        "<b>"
        + provinsi["PROVINSI-1"]
        + "</b><br>"
        + "Peminat: "
        + provinsi["PEMINAT 2024"].astype(int).astype(str)
        + "<br>"
        + "Daya Tampung: "
        + provinsi["DAYA TAMPUNG 2025"].astype(int).astype(str)
    )

    # Gunakan color scale yang menarik
    fig = px.choropleth(
        provinsi,
        geojson=geojson,
        locations="PROVINSI-1",
        featureidkey="properties.state",
        color="PEMINAT 2024",
        hover_name="hover",  # Gunakan informasi hover custom
        color_continuous_scale="YlGnBu",  # Pilihan lain: "Viridis", "Cividis", "Turbo"
        title="🗺️ Heatmap Peminat Program Studi per Provinsi (2024)",
    )

    # Gaya peta & estetika tampilan
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showland=True,
        landcolor="whitesmoke",
        subunitcolor="gray",
    )

    # Tata letak & tampilan colorbar
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(size=24, family="Arial", color="darkblue"),
        coloraxis_colorbar=dict(
            title="Jumlah Peminat",
            tickformat=",",  # Format ribuan: 21.258 bukan 21.3k
            tickfont=dict(size=14),
        ),
    )

    # Hover custom
    fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")

    # Tampilkan dengan container penuh
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    Insight:

    Kota besar cenderung memiliki jumlah prodi terbanyak, sedangkan provinsi dengan jumlah peminat tinggi cenderung berada di wilayah dengan populasi besar.

    Indonesia timur memiliki peminat yang lebih rendah, menandakan peluang untuk memperluas akses pendidikan di daerah tersebut.

    Provinsi dengan jumlah peminat tinggi tapi daya tampung terbatas menghadapi persaingan ketat, membutuhkan solusi untuk memperbesar kapasitas.
    """
    )

# --- 4. Kinerja Universitas ---
elif analisis == "4. Kinerja Universitas":
    st.title("🏢 Analisis Kinerja Universitas di Indonesia")

    st.markdown(
        """
    <p style='font-size: 16px; text-align: justify;'>
    Halaman ini menyajikan analisis terhadap performa universitas berdasarkan jumlah program studi, jumlah peminat tahun 2024, dan rasio peminat terhadap daya tampung.
    Data ini memberikan gambaran mengenai daya tarik masing-masing universitas di mata calon mahasiswa.
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Agregasi data
    univ = (
        df.groupby("Universitas", dropna=True)
        .agg({"Nama Prodi": "count", "PEMINAT 2024": "sum", "Rasio Peminat": "mean"})
        .reset_index()
        .rename(columns={"Nama Prodi": "Jumlah Prodi"})
    )

    # Metrik ringkasan
    total_univ = univ["Universitas"].nunique()
    total_prodi = univ["Jumlah Prodi"].sum()
    univ_terbanyak = univ.sort_values("Jumlah Prodi", ascending=False).iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Universitas", total_univ)
    col2.metric("Total Program Studi", total_prodi)
    col3.markdown(
        f"""
        <div style='font-size: 16px; font-weight: bold;'>
            Universitas Prodi Terbanyak:
        </div>
        <div style='font-size: 15px;'>{univ_terbanyak['Universitas']}<br>
        ({univ_terbanyak['Jumlah Prodi']} Prodi)</div>
        """,
        unsafe_allow_html=True,
    )

    # Tabel universitas dengan peminat tertinggi
    st.subheader("📊 Tabel Peringkat Universitas berdasarkan Jumlah Peminat")
    st.dataframe(
        univ.sort_values("PEMINAT 2024", ascending=False), use_container_width=True
    )

    # Slider untuk mengatur jumlah top universitas
    st.subheader("🏆 Visualisasi Universitas Berdasarkan Peminat")
    top_n = st.slider("Tampilkan Top N Universitas Berdasarkan Peminat", 5, 30, 10)

    top_univ = univ.sort_values("PEMINAT 2024", ascending=False).head(top_n)

    fig = px.bar(
        top_univ,
        x="PEMINAT 2024",
        y="Universitas",
        orientation="h",
        color="PEMINAT 2024",
        color_continuous_scale="Blues",
        labels={"PEMINAT 2024": "Jumlah Peminat"},
        title=f"Top {top_n} Universitas dengan Jumlah Peminat Tertinggi (2024)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Insight
    st.subheader("🔍 Insight dan Analisis")
    st.markdown(
        """
    - Beberapa universitas memiliki **jumlah peminat sangat tinggi**, menandakan reputasi atau daya tarik kuat, baik dari sisi kualitas pendidikan maupun lokasi strategis.
    - Universitas dengan **jumlah program studi banyak tidak selalu memiliki peminat terbanyak** – menunjukkan bahwa keberagaman prodi bukan satu-satunya faktor daya tarik.
    - **Rata-rata rasio peminat terhadap daya tampung** bisa menjadi indikator tekanan persaingan masuk; universitas dengan rasio tinggi cenderung lebih selektif dan kompetitif.
    - Informasi ini bisa digunakan oleh calon mahasiswa untuk memahami **tren popularitas kampus**, dan oleh pembuat kebijakan untuk melihat distribusi minat yang tidak merata.
    """
    )

# --- 5. Jenjang Pendidikan ---
elif analisis == "5. Jenjang Pendidikan":
    st.title("🎓 Analisis Jenjang Pendidikan di Indonesia")

    st.markdown(
        """
        <p style='font-size:16px;'>
        Halaman ini menganalisis data berdasarkan jenjang pendidikan seperti D3, D4/Sarjana Terapan, dan S1.
        Analisis mencakup jumlah program studi, total peminat, daya tampung, serta rasio peminat terhadap daya tampung di tiap jenjang.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # --- Agregasi data ---
    jenjang = (
        df.groupby("JENJANG", dropna=True)
        .agg(
            {
                "Nama Prodi": "count",
                "PEMINAT 2024": "sum",
                "DAYA TAMPUNG 2025": "sum",
                "Rasio Peminat": "mean",
            }
        )
        .reset_index()
        .rename(columns={"Nama Prodi": "Jumlah Prodi"})
        .sort_values("PEMINAT 2024", ascending=False)
    )

    # --- Tampilkan dalam bentuk metrik ---
    st.subheader("📌 Ringkasan Per Jenjang Pendidikan")
    for i in range(len(jenjang)):
        with st.expander(f"📘 {jenjang.iloc[i]['JENJANG']}"):
            st.metric("Jumlah Prodi", jenjang.iloc[i]["Jumlah Prodi"])
            st.metric("Total Peminat", int(jenjang.iloc[i]["PEMINAT 2024"]))
            st.metric("Total Daya Tampung", int(jenjang.iloc[i]["DAYA TAMPUNG 2025"]))
            st.metric(
                "Rata-rata Rasio Peminat", round(jenjang.iloc[i]["Rasio Peminat"], 2)
            )

    # --- Visualisasi ---
    st.subheader("📊 Perbandingan Peminat dan Daya Tampung per Jenjang")

    fig = px.bar(
        jenjang,
        x="JENJANG",
        y=["PEMINAT 2024", "DAYA TAMPUNG 2025"],
        barmode="group",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        labels={
            "value": "Jumlah",
            "JENJANG": "Jenjang Pendidikan",
            "variable": "Kategori",
        },
        title="Peminat vs Daya Tampung Berdasarkan Jenjang",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Insight ---
    st.subheader("🔍 Insight dan Analisis")

    st.markdown(
        """
        - **Jenjang S1 (Sarjana)** masih mendominasi baik dari sisi jumlah prodi, peminat, maupun daya tampung.
        - **D4/Sarjana Terapan** menunjukkan tren peningkatan jumlah peminat meskipun jumlah prodinya belum sebanyak jenjang lain.
        - Rasio peminat yang tinggi di jenjang tertentu dapat menjadi indikator **tingkat kompetitif yang lebih besar**.
        - Jenjang pendidikan di luar S1 (D3/D4) tetap penting sebagai alternatif pendidikan vokasional yang aplikatif dan siap kerja.
        """
    )

# --- 6. Gaji & Prospek Kerja ---
elif analisis == "6. Gaji & Prospek Kerja":
    st.title("💼 Analisis Gaji & Prospek Kerja Lulusan")

    st.markdown(
        """
        <p style='font-size:16px;'>
        Halaman ini menampilkan informasi mengenai potensi gaji minimum dan maksimum lulusan dari berbagai kelompok program studi, 
        serta prospek kerja berdasarkan data prediksi dan klasifikasi hasil.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Hilangkan 'Rp' dan titik, lalu konversi ke float
    df["Gaji Minimal"] = (
        df["Gaji Minimal"].str.replace("Rp", "").str.replace(".", "").astype(float)
    )
    df["Gaji Maksimal"] = (
        df["Gaji Maksimal"].str.replace("Rp", "").str.replace(".", "").astype(float)
    )

    # --- Filter dan urutan ---
    df_sorted = df[
        ["Kelompok", "Gaji Minimal", "Gaji Maksimal", "PEMINAT 2024"]
        + prospek_cols
        + ["Hasil"]
    ].dropna()

    agg_dict = {
        "Gaji Minimal": "first",
        "Gaji Maksimal": "first",
        "PEMINAT 2024": "sum",
    }

    # tambahkan prospek_cols dan Hasil ke agg_dict
    for col in prospek_cols + ["Hasil"]:
        agg_dict[col] = "first"

    df_agg = df_sorted.groupby("Kelompok").agg(agg_dict).reset_index()

    top_n_gaji = st.slider(
        "Tentukan jumlah Top-N Gaji Maksimum Tertinggi:",
        5,
        30,
        10,
        key="slider_top_n_gaji",
    )

    # --- Tampilkan Top 10 Kelompok dengan Gaji Maksimum Tertinggi ---
    st.subheader(
        f"**🏆 Top {top_n_gaji} Kelompok Program Studi dengan Gaji Maksimum Tertinggi**"
    )
    top_gaji = df_agg.sort_values("Gaji Maksimal", ascending=False).head(top_n_gaji)
    st.dataframe(
        top_gaji.style.format(
            {
                "Gaji Maksimal": "Rp.{:.0f}",
                "Gaji Minimal": "Rp.{:.0f}",
                "PEMINAT 2024": "{:.0f}",
            }
        )
    )

    top_gaji["Gaji Maksimal"] = pd.to_numeric(
        top_gaji["Gaji Maksimal"], errors="coerce"
    )

    # --- Visualisasi Gaji Maksimal ---
    st.subheader("📈 Visualisasi Gaji Maksimal per Kelompok")
    fig_gaji = px.bar(
        top_gaji.sort_values("Gaji Maksimal", ascending=True),
        x="Gaji Maksimal",
        y="Kelompok",
        orientation="h",
        color="Gaji Maksimal",
        color_continuous_scale="blues",
        hover_data=["Gaji Minimal", "PEMINAT 2024"],
        labels={"Gaji Maksimal": "Gaji Maksimal (Rp)", "Kelompok": "Kelompok Prodi"},
        title="Kelompok dengan Gaji Maksimum Tertinggi",
    )
    fig_gaji.update_layout(height=500)
    st.plotly_chart(fig_gaji, use_container_width=True)

    # --- Klasifikasi prospek kerja per hasil ---
    st.subheader("🔍 Klasifikasi Prospek Kerja")

    prospek_summary = df_sorted["Hasil"].value_counts().reset_index()
    prospek_summary.columns = ["Kategori Prospek", "Jumlah Prodi"]
    fig_prospek = px.pie(
        prospek_summary,
        names="Kategori Prospek",
        values="Jumlah Prodi",
        title="Distribusi Kategori Prospek Kerja",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_prospek, use_container_width=True)

    # --- Insight ---
    st.subheader("🧠 Insight & Analisis")
    st.markdown(
        """
        - **Kelompok program studi dengan gaji maksimal tinggi** didominasi bidang teknologi, kesehatan, dan bisnis.
        - Beberapa kelompok dengan peminat rendah justru memiliki **potensi gaji yang tinggi**, menunjukkan peluang tersembunyi.
        - Klasifikasi hasil prospek kerja bisa menjadi referensi bagi calon mahasiswa untuk **menyesuaikan minat dengan peluang kerja di masa depan**.
        """
    )

elif analisis == "7. Prodi PTN Sepi Tapi Potensial":
    st.title("🎯 Prodi PTN Peminat Rendah/Tinggi Tapi Potensial")

    st.markdown(
        """
        <p style='font-size:16px;'>
        Halaman ini menyajikan <b>program studi di PTN</b> berdasarkan kombinasi <span style='color:red'><b>tingkat peminat</b></span> 
        dan <span style='color:green'><b>potensi masa depan</b></span>. Cocok untuk menemukan prodi dengan peluang karier baik 
        meskipun persaingan masuk rendah atau sedang.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Konversi gaji maksimal
    df["Gaji Maksimal"] = (
        df["Gaji Maksimal"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", None)
        .astype(float)
    )

    # -----------------------
    # 🔘 FILTER 1: Kategori Peminat
    # -----------------------
    st.subheader("📊 Filter 1: Kategori Peminat")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        pilih_sepi = st.checkbox("📉 Sepi Peminat")
    with col_p2:
        pilih_sedang = st.checkbox("📊 Sedang Peminat")
    with col_p3:
        pilih_ramai = st.checkbox("📈 Ramai Peminat")

    kategori_dipilih = []
    if pilih_sepi:
        kategori_dipilih.append("Sepi Peminat")
    if pilih_sedang:
        kategori_dipilih.append("Sedang Peminat")
    if pilih_ramai:
        kategori_dipilih.append("Ramai Peminat")

    # -----------------------
    # 🌟 FILTER 2: Potensi Hasil
    # -----------------------
    st.subheader("🌟 Filter 2: Potensi Hasil")

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        hasil_cukup = st.checkbox("🔹 Cukup Potensial")
    with col_h2:
        hasil_sangat = st.checkbox("🔸 Sangat Potensial")
    with col_h3:
        hasil_prospektif = st.checkbox("🏅 Sangat Prospektif")

    hasil_dipilih = []
    if hasil_cukup:
        hasil_dipilih.append("Cukup Potensial")
    if hasil_sangat:
        hasil_dipilih.append("Sangat Potensial")
    if hasil_prospektif:
        hasil_dipilih.append("Sangat Prospektif")

    # -----------------------
    # ⏳ FILTER VALIDATION
    # -----------------------
    if not kategori_dipilih or not hasil_dipilih:
        st.info("Silakan pilih minimal 1 kategori peminat dan 1 potensi hasil.")
    else:
        df_filtered = df[
            df["Kategori"].isin(kategori_dipilih) & df["Hasil"].isin(hasil_dipilih)
        ].copy()

        if df_filtered.empty:
            st.warning("Tidak ada data sesuai kombinasi filter yang dipilih.")
        else:
            # 🔢 Agregasi dan Visualisasi
            ranking_kelompok = (
                df_filtered.groupby("Kelompok")
                .agg(
                    Jumlah_Prodi=("Nama Prodi", "count"),
                    Rata2_Rasio_Peminat=("Rasio Peminat", "mean"),
                )
                .reset_index()
                .sort_values("Rata2_Rasio_Peminat", ascending=False)
            )

            st.subheader("🏅 Ranking Kelompok Terpilih")

            max_rank = len(ranking_kelompok)
            top_n = st.slider(
                "🔢 Pilih jumlah kelompok teratas:",
                1,
                max(1, max_rank),
                min(5, max_rank),
            )
            top_kelompok = ranking_kelompok.head(top_n)

            fig_bar = px.bar(
                top_kelompok,
                x="Rata2_Rasio_Peminat",
                y="Kelompok",
                orientation="h",
                color="Jumlah_Prodi",
                text="Jumlah_Prodi",
                color_continuous_scale="Agsunset",
                labels={
                    "Rata2_Rasio_Peminat": "Rata-rata Rasio Peminat per Daya Tampung",
                    "Kelompok": "Kelompok Prodi",
                },
            )
            fig_bar.update_layout(height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("🔍 Lihat Detail Prodi dalam Kelompok")
            kelompok_terpilih = st.selectbox(
                "📂 Pilih Kelompok:", top_kelompok["Kelompok"]
            )

            detail = df_filtered[df_filtered["Kelompok"] == kelompok_terpilih][
                [
                    "Nama Prodi",
                    "Universitas",
                    "Gaji Maksimal",
                    "DAYA TAMPUNG 2025",
                    "PEMINAT 2024",
                    "Prospek Kerja 1",
                    "Prospek Kerja 2",
                    "Prospek Kerja 3",
                    "Prospek Kerja 4",
                    "Hasil",
                ]
            ].sort_values("Gaji Maksimal", ascending=False)

            st.markdown(f"### 📄 Detail Prodi dalam **{kelompok_terpilih}**")
            st.dataframe(detail.style.format({"Gaji Maksimal": "Rp{:,.0f}"}))

            # Insight
            st.subheader("🧠 Insight & Analisis")
            st.markdown(
                f"""
                - ✅ Filter aktif: <b>{', '.join(kategori_dipilih)}</b> dan hasil <b>{', '.join(hasil_dipilih)}</b>.
                - 📊 Banyak prodi dengan potensi karier tinggi meski peminat tidak banyak.
                - 💡 Peluang besar untuk calon mahasiswa yang ingin jalur minim persaingan.
                """,
                unsafe_allow_html=True,
            )

# --- 8. Segmentasi Kategori & Kelompok ---
elif analisis == "8. Segmentasi Kategori & Kelompok":
    st.title("🏷️ Segmentasi Berdasarkan Kategori & Kelompok Ilmu")

    st.markdown(
        """
        <p style='font-size:16px;'>
        Halaman ini menampilkan <b>segmentasi program studi</b> berdasarkan <b>kategori popularitas</b> dan <b>kelompok bidang keilmuan</b>. 
        Analisis ini berguna untuk memahami hubungan antara <span style='color:blue'><b>minat calon mahasiswa</b></span> 
        dengan <span style='color:green'><b>potensi gaji lulusan</b></span> di berbagai kelompok studi.
        </p>
        """,
        unsafe_allow_html=True,
    )

    def klasifikasi_peminat(x):
        if x < 10:
            return "Sepi Peminat"
        elif x < 30:
            return "Sedang"
        else:
            return "Ramai Peminat"

    df["Kategori"] = df["Rasio Peminat"].apply(klasifikasi_peminat)

    # Bersihkan dan konversi Gaji Maksimal ke float
    df["Gaji Maksimal"] = (
        df["Gaji Maksimal"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", None)
        .astype(float)
    )

    # Filter data yang valid
    df_filtered = df[
        (df["DAYA TAMPUNG 2025"] > 0)
        & (df["Rasio Peminat"].notna())
        & (df["Gaji Maksimal"].notna())
    ].copy()

    # Agregasi segmentasi
    segmen = (
        df_filtered.groupby(["Kelompok", "Kategori"])
        .agg({"Rasio Peminat": "mean", "Gaji Maksimal": "first", "Nama Prodi": "count"})
        .rename(columns={"Nama Prodi": "Jumlah Prodi"})
        .reset_index()
    )

    # Ringkas label jika terlalu panjang
    segmen["Label Kelompok"] = segmen["Kelompok"].apply(
        lambda x: x if len(x) <= 30 else x[:30] + "…"
    )

    st.subheader("📊 Visualisasi Segmentasi")

    # Hitung garis bantu rata-rata
    x_avg = segmen["Rasio Peminat"].mean()
    y_avg = segmen["Gaji Maksimal"].mean()

    fig = px.scatter(
        segmen,
        size_max=40,
        opacity=0.8,
        x="Rasio Peminat",
        y="Gaji Maksimal",
        color="Kategori",
        size="Jumlah Prodi",
        hover_name="Kelompok",
        # Hapus text label untuk menghindari tumpukan
        labels={
            "Rasio Peminat": "Rata-rata Rasio Peminat",
            "Gaji Maksimal": "Rata-rata Gaji Maksimal",
            "Jumlah Prodi": "Jumlah Prodi",
        },
        title="Pemetaan Kelompok Ilmu Berdasarkan Popularitas dan Gaji",
        height=650,
    )

    # Tambahkan garis bantu rata-rata horizontal dan vertikal
    fig.add_vline(x=x_avg, line_width=1, line_dash="dot", line_color="gray")
    fig.add_hline(y=y_avg, line_width=1, line_dash="dot", line_color="gray")

    # Tambahan: beri keterangan kuadran di tooltip saja
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
        hoverlabel=dict(bgcolor="black", font_size=13),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabel detail
    st.subheader("📋 Data Segmentasi Lengkap")
    st.dataframe(
        segmen[
            ["Kelompok", "Kategori", "Rasio Peminat", "Gaji Maksimal", "Jumlah Prodi"]
        ]
        .sort_values(by="Rasio Peminat", ascending=False)
        .style.format({"Rasio Peminat": "{:.2f}", "Gaji Maksimal": "Rp{:,.0f}"})
    )

    # Insight
    st.subheader("🧠 Insight & Analisis")
    st.markdown(
        """
        <p style='font-size:16px;'>
        - 🔍 Kuadran kanan atas (rasio & gaji tinggi) menunjukkan kelompok prodi dengan daya saing tinggi namun menjanjikan dari sisi penghasilan.<br>
        - 🟡 Beberapa kelompok di kuadran kiri atas (gaji tinggi, peminat rendah) seperti <i>Aktuaria</i> atau <i>Data Science</i> memiliki potensi tersembunyi.<br>
        - 🔴 Kuadran kanan bawah (peminat tinggi, gaji rendah) perlu diwaspadai karena bisa jadi pasar kerjanya jenuh.<br>
        - 📉 Kelompok seperti <i>Bahasa Lokal</i> dan <i>Tradisi Lisan</i> masih minim peminat dan belum menunjukkan potensi gaji tinggi.<br>
        - 📊 Data ini cocok dijadikan dasar pengambilan keputusan strategis untuk mahasiswa, pendidik, dan perencana pendidikan.
        </p>
        """,
        unsafe_allow_html=True,
    )
