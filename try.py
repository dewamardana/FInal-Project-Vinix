import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset Final Project.csv", sep=",", thousands=".", decimal=",")
    # Data cleaning
    df["Gaji Minimal"] = (
        df["Gaji Minimal"]
        .str.replace("Rp", "")
        .str.replace(".", "", regex=False)
        .astype(float)
    )
    df["Gaji Maksimal"] = (
        df["Gaji Maksimal"]
        .str.replace("Rp", "")
        .str.replace(".", "", regex=False)
        .astype(float)
    )
    return df


df = load_data()

st.title("Analisis Prodi PTN dengan Prospek Kerja Baik 🎓")
st.write(
    """
**Objective**: Mengidentifikasi program studi di PTN yang memiliki peminat rendah namun prospek kerja baik
"""
)

# Sidebar filters
st.sidebar.header("Filter Data")
selected_provinsi = st.sidebar.multiselect("Pilih Provinsi", df["PROVINSI-1"].unique())
selected_kelompok = st.sidebar.multiselect(
    "Pilih Kelompok Studi", df["Kelompok"].unique()
)
min_rasio = st.sidebar.slider("Rasio Peminat Maksimum", 0.0, 20.0, 5.0)

# Filter data
filtered_df = df[
    (df["Kategori"] == "Sepi Peminat")
    & (df["Hasil"].str.contains("Prospek|Potensial", na=False))
    & (df["Rasio Peminat"] <= min_rasio)
]

if selected_provinsi:
    filtered_df = filtered_df[filtered_df["PROVINSI-1"].isin(selected_provinsi)]
if selected_kelompok:
    filtered_df = filtered_df[filtered_df["Kelompok"].isin(selected_kelompok)]

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Analisis Deskriptif",
        "Diagnostik & Korelasi",
        "Segmentasi & Clustering",
        "Analisis Geospasial",
        "Rekomendasi",
    ]
)

with tab1:
    st.header("Analisis Deskriptif 📊")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Prodi", filtered_df.shape[0])
    with col2:
        st.metric(
            "Rasio Peminat Rata-rata", f"{filtered_df['Rasio Peminat'].mean():.2f}"
        )
    with col3:
        st.metric("Universitas Terbanyak", filtered_df["Universitas"].mode()[0])

    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Rasio Peminat"], bins=10, kde=True, ax=ax)
    ax.set_title("Distribusi Rasio Peminat")
    st.pyplot(fig)

with tab2:
    st.header("Analisis Diagnostik 🔍")

    # Correlation analysis
    corr_matrix = filtered_df[
        ["Rasio Peminat", "DAYA TAMPUNG 2025", "PEMINAT 2024"]
    ].corr()
    st.write("**Matriks Korelasi**")
    st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))

    # Prospek kerja analysis
    prospek_counts = filtered_df["Hasil"].value_counts()
    fig = px.pie(
        prospek_counts, values=prospek_counts.values, names=prospek_counts.index
    )
    st.plotly_chart(fig)

with tab3:
    st.header("Segmentasi & Clustering 🧩")

    # Prepare data for clustering
    cluster_data = filtered_df[["Rasio Peminat", "DAYA TAMPUNG 2025"]].dropna()

    # K-means clustering
    kmeans = KMeans(n_clusters=3)
    cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)

    fig = px.scatter(
        cluster_data,
        x="Rasio Peminat",
        y="DAYA TAMPUNG 2025",
        color="Cluster",
        hover_name=filtered_df["Nama Prodi"],
    )
    st.plotly_chart(fig)

with tab4:
    st.header("Analisis Geospasial 🌍")

    # Geospatial plot
    fig = px.scatter_geo(
        filtered_df,
        lat="KAB / KOTA",  # Replace with actual coordinates if available
        lon="PROVINSI-1",
        hover_name="Nama Prodi",
        size="Rasio Peminat",
        projection="natural earth",
    )
    st.plotly_chart(fig)

with tab5:
    st.header("Rekomendasi Strategis 💡")

    st.write(
        """
    **Rekomendasi berdasarkan Analisis:**
    1. Fokus pada prodi dengan rasio peminat < 5 dan daya tampung tinggi
    2. Tingkatkan promosi prodi di kelompok teknologi dan kesehatan
    3. Kolaborasi dengan industri untuk program magang
    4. Sediakan beasiswa khusus untuk prodi prioritas
    """
    )

    top_prodi = filtered_df.sort_values(by=["Rasio Peminat", "DAYA TAMPUNG 2025"]).head(
        5
    )
    st.write("**Top 5 Prodi Rekomendasi**")
    st.dataframe(top_prodi[["Nama Prodi", "Universitas", "Rasio Peminat"]])

# Show raw data
if st.checkbox("Tampilkan Data Mentah"):
    st.dataframe(filtered_df)


# try 2
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans

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

# Sidebar - Navigasi analisis
analisis = st.sidebar.selectbox(
    "Pilih Analisis:",
    [
        "1. Permintaan & Daya Saing",
        "2. Persebaran Lokasi Prodi",
        "3. Kinerja Universitas",
        "4. Jenjang Pendidikan",
        "5. Gaji & Prospek Kerja",
        "6. Prodi PTN Sepi Tapi Potensial",
        "7. Segmentasi Kategori & Kelompok",
        "8. Kualitas & Enrichment Data",
    ],
)

# --- 1. Permintaan & Daya Saing ---
if analisis == "1. Permintaan & Daya Saing":
    st.title("📊 Analisis Permintaan & Daya Saing Program Studi")

    kategori_filter = st.multiselect(
        "Filter Kategori:",
        options=df["Kategori"].unique(),
        default=df["Kategori"].unique(),
    )
    min_peminat = st.slider("Minimum Peminat:", 0, int(df["PEMINAT 2024"].max()), 0)

    df_filtered = df[
        (df["Kategori"].isin(kategori_filter)) & (df["PEMINAT 2024"] >= min_peminat)
    ]

    st.subheader("Distribusi Peminat")
    fig = px.histogram(df_filtered, x="PEMINAT 2024", nbins=50)
    st.plotly_chart(fig)

    st.subheader("Top 10 Paling Kompetitif")
    top_kompetitif = df_filtered.sort_values(by="Rasio Peminat", ascending=False).head(
        10
    )
    st.dataframe(
        top_kompetitif[
            ["Nama Prodi", "DAYA TAMPUNG 2025", "PEMINAT 2024", "Rasio Peminat"]
        ]
    )

    st.subheader("Top 10 Sepi Peminat")
    sepi = df_filtered.sort_values(by="Rasio Peminat").head(10)
    st.dataframe(
        sepi[["Nama Prodi", "DAYA TAMPUNG 2025", "PEMINAT 2024", "Rasio Peminat"]]
    )

    st.subheader("Cluster Analysis")
    cluster_data = df_filtered[["Rasio Peminat", "DAYA TAMPUNG 2025"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(cluster_data)
    cluster_data["Cluster"] = kmeans.labels_
    fig = px.scatter(
        cluster_data, x="DAYA TAMPUNG 2025", y="Rasio Peminat", color="Cluster"
    )
    st.plotly_chart(fig)

    # ✅ Analisis Kelompok berdasarkan Rata-rata Rasio Peminat
    st.subheader("📈 Kelompok Berdasarkan Rata-rata Rasio Peminat")

    kelompok_summary = (
        df_filtered.groupby("Kelompok")["Rasio Peminat"]
        .mean()
        .reset_index()
        .sort_values(by="Rasio Peminat", ascending=False)
    )
    st.dataframe(kelompok_summary)

    selected_kelompok_detail = st.selectbox(
        "Pilih Kelompok untuk Lihat Detail Prodi:", options=kelompok_summary["Kelompok"]
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

# --- 2. Persebaran Lokasi Prodi ---
elif analisis == "2. Persebaran Lokasi Prodi":
    st.title("📍 Persebaran Lokasi Program Studi")
    st.subheader("Jumlah Program Studi per Kota")
    kota_count = df.groupby("KAB / KOTA").size().reset_index(name="Jumlah Prodi")
    st.dataframe(kota_count.sort_values("Jumlah Prodi", ascending=False))

    st.subheader("Peta Panas Jumlah Peminat per Provinsi")
    provinsi = (
        df.groupby("PROVINSI-1")
        .agg({"PEMINAT 2024": "sum", "DAYA TAMPUNG 2025": "sum"})
        .reset_index()
    )
    fig = px.choropleth(
        provinsi,
        locations="PROVINSI-1",
        locationmode="province names",
        color="PEMINAT 2024",
        scope="asia",
        title="Heatmap Peminat per Provinsi",
    )
    st.plotly_chart(fig)

# --- 3. Kinerja Universitas ---
elif analisis == "3. Kinerja Universitas":
    st.title("🏢 Analisis Kinerja Universitas")
    univ = (
        df.groupby("Universitas")
        .agg({"Nama Prodi": "count", "PEMINAT 2024": "sum", "Rasio Peminat": "mean"})
        .reset_index()
    )
    st.dataframe(
        univ.sort_values("PEMINAT 2024", ascending=False).rename(
            columns={"Nama Prodi": "Jumlah Prodi"}
        )
    )

# --- 4. Jenjang Pendidikan ---
elif analisis == "4. Jenjang Pendidikan":
    st.title("🎓 Analisis Jenjang Pendidikan")
    jenjang = (
        df.groupby("JENJANG")
        .agg(
            {
                "Nama Prodi": "count",
                "PEMINAT 2024": "sum",
                "DAYA TAMPUNG 2025": "sum",
                "Rasio Peminat": "mean",
            }
        )
        .reset_index()
    )
    st.dataframe(jenjang.rename(columns={"Nama Prodi": "Jumlah Prodi"}))

# --- 5. Gaji & Prospek Kerja ---
elif analisis == "5. Gaji & Prospek Kerja":
    st.title("💼 Gaji & Prospek Kerja")
    df_sorted = df[
        ["Nama Prodi", "Gaji Minimal", "Gaji Maksimal", "PEMINAT 2024"] + prospek_cols
    ].dropna()
    st.subheader("Top Gaji Maksimum")
    st.dataframe(df_sorted.sort_values("Gaji Maksimal", ascending=False).head(10))

# --- 6. Prodi PTN Sepi Tapi Potensial ---
elif analisis == "6. Prodi PTN Sepi Tapi Potensial":
    st.title("🎯 Prodi PTN Sepi Peminat Tapi Potensial")
    sepi_potensial = df[
        (df["Kategori"] == "Sepi")
        & (df["Gaji Maksimal"] > df["Gaji Maksimal"].quantile(0.75))
    ]
    st.dataframe(
        sepi_potensial[["Nama Prodi", "Universitas", "Gaji Maksimal"] + prospek_cols]
    )

# --- 7. Segmentasi Kategori & Kelompok ---
elif analisis == "7. Segmentasi Kategori & Kelompok":
    st.title("🏷️ Segmentasi Berdasarkan Kategori & Kelompok Ilmu")
    segmen = (
        df.groupby(["Kelompok", "Kategori"])
        .agg({"Rasio Peminat": "mean", "Gaji Maksimal": "mean"})
        .reset_index()
    )
    st.dataframe(segmen)

# --- 8. Kualitas & Enrichment ---
elif analisis == "8. Kualitas & Enrichment Data":
    st.title("🧹 Data Quality & Enrichment")
    st.subheader("Missing Value Summary")
    st.dataframe(df.isna().sum())

    st.subheader("Contoh Format Numerik Dibersihkan")
    st.write(df[["Rasio Peminat"]].head())
