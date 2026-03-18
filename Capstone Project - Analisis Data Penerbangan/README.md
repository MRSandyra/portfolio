# ✈️ Capstone Project — Analisis Data Penerbangan

Proyek ini merupakan analisis data tiket penerbangan menggunakan Python dan berbagai library data science. Tujuan utamanya adalah memahami pola penjualan tiket berdasarkan maskapai, rute, dan tanggal keberangkatan/kedatangan.

---

## 📁 Dataset

- **File:** `flight.csv`
- **Sumber:** Google Drive (diakses via Google Colab)
- **Deskripsi:** Dataset berisi informasi tiket penerbangan meliputi nama maskapai, bandara keberangkatan, bandara tujuan, stopover, harga tiket (dalam Dolar), dan tanggal kedatangan.

### Kolom Utama

| Kolom | Deskripsi |
|---|---|
| `Airline name` | Nama maskapai penerbangan |
| `Depreture Airport` | Bandara keberangkatan |
| `Destination Airport` | Bandara tujuan |
| `1st Stoppage` | Stopover pertama |
| `Ticket prize(Doller)` | Harga tiket dalam USD |
| `Arrival Date` | Tanggal kedatangan |
| `jumlah tiket terjual` | Jumlah tiket terjual per maskapai (kolom turunan) |

---

## 🛠️ Library yang Digunakan

```python
pandas       # Manipulasi dan analisis data
numpy        # Operasi matematika
matplotlib   # Visualisasi data
seaborn      # Visualisasi data statistik
os           # Operasi sistem (Google Colab)
```

---

## 🔄 Alur Analisis

### 1. Setup & Load Data
- Mount Google Drive ke Google Colab
- Import semua library yang dibutuhkan
- Baca dataset `flight.csv` ke dalam DataFrame

### 2. Understanding Data
- Cek dimensi data (`df.shape`)
- Lihat nama kolom (`df.columns`)
- Preview data awal (`df.head()`)
- Hapus kolom yang memiliki nilai kosong (NaN) lebih dari **70%**

### 3. Feature Engineering
- Hitung jumlah tiket terjual per maskapai menggunakan `groupby`
- Tambahkan kolom `jumlah tiket terjual` ke DataFrame utama via `merge`

### 4. Data Cleaning
- Deteksi dan hapus data duplikat (`drop_duplicates`)
- Cek dan tangani *missing values*:
  - Isi nilai kosong pada kolom `jumlah tiket terjual` menggunakan metode **backward fill (bfill)**
- Simpan salinan data bersih sebagai `df2`

### 5. Exploratory Data Analysis (EDA)

Beberapa analisis dan visualisasi yang dilakukan:

- **Statistik deskriptif** jumlah tiket terjual
- **Distribusi** jumlah tiket terjual (histogram + KDE)
- **Heatmap korelasi** antar variabel numerik
- **Boxplot** pengaruh maskapai terhadap jumlah tiket terjual
- **Bar chart** distribusi berdasarkan tanggal kedatangan
- **Violin plot** jumlah tiket terjual vs tanggal kedatangan
- **Pivot table** jumlah tiket terjual per maskapai (dengan gradient warna)
- **Top 10 maskapai** berdasarkan total tiket terjual

---

## 📊 Visualisasi Utama

| Visualisasi | Keterangan |
|---|---|
| Histogram Tiket Terjual | Distribusi frekuensi jumlah tiket terjual |
| Heatmap Korelasi | Hubungan antar variabel numerik |
| Boxplot per Maskapai | Perbandingan distribusi tiket antar maskapai |
| Violin Plot per Tanggal | Distribusi tiket berdasarkan tanggal kedatangan |
| Bar Chart Top 10 Maskapai | Maskapai dengan tiket terjual terbanyak |

---

## 💾 Output

Data yang telah dibersihkan dan diproses disimpan kembali ke file:

```
df2.csv
```

---

## 🚀 Cara Menjalankan

1. Buka notebook di **Google Colab**
2. Mount Google Drive dan arahkan ke folder dataset:
   ```
   /content/gdrive/MyDrive/Capstone Projek Dataset/
   ```
3. Pastikan file `flight.csv` tersedia di folder tersebut
4. Jalankan semua sel secara berurutan dari atas ke bawah

---

## 📌 Catatan

- Notebook ini dijalankan di lingkungan **Google Colab**
- Pastikan semua library sudah terinstall (tersedia secara default di Colab)
- Kolom dengan lebih dari 70% nilai kosong akan dihapus secara otomatis saat preprocessing
