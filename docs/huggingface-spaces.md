# Panduan Deploy Web Dashboard ke Hugging Face Spaces

AutoDataMining menyediakan **Web Dashboard No-Code Interaktif** berbasis **Streamlit** yang dapat diakses melalui browser tanpa perlu menginstal aplikasi apa pun. Anda dapat menerapkannya (*deploy*) secara gratis ke **Hugging Face Spaces**.

---

## 🌟 Mengapa Memakai Hugging Face Spaces?
- 🆓 **100% Gratis:** Diberikan kuota CPU & RAM gratis selamanya oleh Hugging Face.
- 🌐 **Dapat Diakses Siapa Saja:** Memiliki tautan web publik (misal: `https://huggingface.co/spaces/username/autodatamining`) yang bisa dicantumkan pada naskah skripsi, paper jurnal, atau dibagikan ke dosen pembimbing.
- 📱 **Ramah Pemula:** Penguji, dosen, atau mahasiswa dapat langsung mengunggah file CSV atau mencoba dataset edukasi bawaan tanpa menyentuh terminal/kode.

---

## Cara 1: Deploy Instan Melalui Web Hugging Face (Paling Mudah)

Langkah demi langkah (waktu pengerjaan ~3 menit):

1. **Buka Akun Hugging Face:**
   - Kunjungi [huggingface.co](https://huggingface.co/) dan login (atau daftar gratis jika belum memiliki akun).
2. **Buat Space Baru:**
   - Klik foto profil di pojok kanan atas $\to$ Pilih **New Space** (atau kunjungi [huggingface.co/new-space](https://huggingface.co/new-space)).
   - Isi form berikut:
     * **Space name:** `autodatamining-studio` (atau nama lain pilihan Anda).
     * **License:** `mit`.
     * **Select the Space SDK:** Pilih **Streamlit** 🔴.
     * **Space hardware:** Pilih **CPU basic (Free)**.
     * **Visibility:** **Public**.
   - Klik tombol **Create Space**.
3. **Unggah File Aplikasi:**
   - Di halaman Space yang baru dibuat, buka tab **Files and versions** $\to$ Klik **Add file** $\to$ **Upload files**.
   - Unggah berkas berikut dari repository AutoDataMining:
     1. `app.py` (file dashboard utama).
     2. Folder `src/` (berisi seluruh logika pustaka `autodatamining`).
     3. `spaces/huggingface/requirements.txt` (atau buat file `requirements.txt` di root Space).
     4. `spaces/huggingface/README.md` (metadata Streamlit Hugging Face).
4. **Selesai!**
   - Hugging Face akan otomatis membangun (*building*) aplikasi dalam 1–2 menit.
   - Status akan berubah menjadi **Running** 🟢 dan dashboard Anda langsung aktif secara online!

---

## Cara 2: Deploy Otomatis via Git Push

Jika Anda terbiasa menggunakan Git:

```bash
# 1. Clone repository Space dari Hugging Face
git clone https://huggingface.co/spaces/<username>/<space-name> hf-space
cd hf-space

# 2. Salin file aplikasi dari autodatamining
cp /path/to/autodatamining/app.py .
cp -r /path/to/autodatamining/src .
cp /path/to/autodatamining/spaces/huggingface/requirements.txt .
cp /path/to/autodatamining/spaces/huggingface/README.md .

# 3. Commit dan push ke Hugging Face
git add .
git commit -m "Deploy AutoDataMining No-Code Studio"
git push
```

---

## Cara 3: Menjalankan Dashboard Secara Lokal di Komputer Anda

Anda juga bisa menjalankan dashboard ini langsung di laptop/komputer Anda secara offline:

### Opsi A (Melalui Modul AutoDataMining):
```bash
# Pastikan streamlit sudah terpasang
pip install "autodatamining[ui]"

# Jalankan dashboard
python -m autodatamining.ui
```

### Opsi B (Melalui Perintah Streamlit):
```bash
streamlit run app.py
```

Browser akan terbuka secara otomatis di `http://127.0.0.1:8501`.
