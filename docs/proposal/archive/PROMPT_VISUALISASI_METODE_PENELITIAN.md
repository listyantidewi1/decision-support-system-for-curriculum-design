# Prompt untuk Visualisasi Metode Penelitian

Gunakan prompt berikut di ChatGPT, Mermaid, draw.io, Figma, PowerPoint, atau visual renderer lain untuk membuat **gambar akademik metode penelitian** pada proposal disertasi.

---

## Prompt

Buat satu diagram akademik berjudul:

**"Rancangan Metode Penelitian R&D untuk Pengembangan Sistem Rekomendasi Kompetensi"**

Diagram harus memvisualisasikan metode penelitian disertasi yang menggunakan **Research and Development (R&D) Borg and Gall** sebagai kerangka utama, dengan **Agile Development** sebagai proses teknis pengembangan sistem. Diagram harus menunjukkan bahwa pengembangan produk, validasi ahli, uji coba, revisi, dan evaluasi dilakukan bertahap.

Gunakan gaya akademik formal, bersih, tidak dekoratif, cocok untuk proposal disertasi. Orientasi disarankan **A4 landscape** atau **16:9 landscape**. Warna utama: biru tua, abu-abu, putih, dan satu warna aksen untuk tahap validasi/revisi. Jangan gunakan ilustrasi kartun, 3D, foto stok, gradient dekoratif, atau ikon berlebihan.

---

## Prinsip Utama Diagram

1. Alur utama Borg and Gall harus bergerak **kiri ke kanan**.
2. Agile Development harus digambar sebagai **proses teknis internal** di dalam tahap pengembangan dan revisi produk, bukan sebagai pengganti Borg and Gall.
3. Validasi ahli dan uji coba harus terlihat sebagai titik keputusan/revisi.
4. Instrumen penelitian harus muncul sebagai bagian dari metode, bukan hanya lampiran terpisah.
5. Output akhir harus berupa produk penelitian yang tervalidasi secara bertahap, bukan klaim final otomatis.
6. Jangan menggambar dashboard sebagai sumber validitas. Dashboard adalah produk/antarmuka yang divalidasi.
7. Jangan menggunakan istilah teknis pipeline terlalu dominan. Diagram metode harus dapat dipahami pembaca pendidikan vokasi.

---

## Swimlane/Domain Wajib

Gunakan empat swimlane horizontal dari atas ke bawah:

1. **Kerangka R&D Borg and Gall**
2. **Pengembangan Teknis Agile**
3. **Pengumpulan Data dan Instrumen**
4. **Validasi, Analisis, dan Revisi**

Jika swimlane terlalu padat, gunakan diagram dua lapis:

- lapis atas: alur Borg and Gall;
- lapis bawah: aktivitas pendukung, instrumen, validasi, dan analisis.

---

## Node Utama Borg and Gall

Gunakan node berikut sebagai alur utama. Jangan ubah urutan:

1. **Studi pendahuluan dan pengumpulan informasi**
2. **Perencanaan penelitian dan spesifikasi produk**
3. **Pengembangan produk awal**
4. **Uji coba awal**
5. **Revisi produk utama**
6. **Uji coba utama**
7. **Revisi produk operasional**
8. **Uji coba operasional**
9. **Revisi produk final**
10. **Diseminasi terbatas dan pelaporan**

Gunakan label Bahasa Indonesia. Jika ingin menambahkan istilah Inggris, letakkan sebagai sublabel kecil, misalnya `Borg and Gall: preliminary field testing`.

---

## Aktivitas Wajib pada Setiap Tahap

Tambahkan aktivitas ringkas di bawah atau di dalam setiap tahap.

### 1. Studi Pendahuluan dan Pengumpulan Informasi

Isi aktivitas:

- kajian literatur CBET, DACUM, KKNI, SKKNI, Spektrum Keahlian;
- identifikasi kebutuhan penyelarasan kurikulum RPL/TI;
- pengumpulan awal korpus lowongan kerja;
- wawancara awal atau studi kebutuhan dengan guru/dosen/praktisi jika tersedia.

Instrumen/data:

- pedoman studi dokumen;
- pedoman wawancara kebutuhan awal;
- dokumentasi korpus awal.

### 2. Perencanaan Penelitian dan Spesifikasi Produk

Isi aktivitas:

- merancang arsitektur sistem;
- merancang recommendation contract;
- merancang evidence package;
- menyusun kisi-kisi instrumen;
- menetapkan rencana validasi, gold labeling, uji coba, dan analisis.

Instrumen/data:

- matriks kebutuhan instrumen;
- blueprint instrumen validasi;
- rancangan skenario uji coba.

### 3. Pengembangan Produk Awal

Isi aktivitas:

- membangun pipeline evidence-first;
- membuat dashboard administratif dan publik;
- membuat evidence package;
- membuat validation package;
- membuat worksheet DACUM/CBET;
- membuat coverage analyzer.

Agile internal:

`build -> test -> review -> revise`

Instrumen/data:

- log pengembangan;
- log error;
- audit pipeline;
- dokumentasi versi produk.

### 4. Uji Coba Awal

Isi aktivitas:

- uji teknis pipeline;
- uji keterbacaan evidence package;
- validasi awal instrumen;
- engineering validation dengan korpus US/non-Indonesia jika diperlukan.

Instrumen/data:

- lembar validasi instrumen;
- checklist uji teknis;
- catatan reviewer awal.

Catatan visual wajib:

- tuliskan bahwa korpus US/non-Indonesia hanya untuk engineering validation, bukan klaim kurikulum Indonesia.

### 5. Revisi Produk Utama

Isi aktivitas:

- memperbaiki sentence filtering;
- memperbaiki extraction typing;
- memperbaiki cluster validity;
- memperbaiki evidence repair;
- memperbaiki dashboard dan evidence package.

Instrumen/data:

- log revisi;
- matriks masukan ahli;
- daftar masalah dan tindak lanjut.

### 6. Uji Coba Utama

Isi aktivitas:

- menjalankan pipeline pada korpus Indonesia yang telah diaudit;
- validasi rekomendasi kompetensi oleh ahli;
- validasi KKNI/SKKNI/Spektrum/okupasi;
- review evidence package;
- jika digunakan, Delphi putaran 1 dan 2.

Instrumen/data:

- lembar validasi ahli kurikulum CBET/DACUM;
- lembar validasi praktisi industri RPL/TI;
- lembar validasi ahli NLP/AI;
- lembar validasi ahli evaluasi;
- instrumen Delphi/expert panel;
- lembar review competency statement.

### 7. Revisi Produk Operasional

Isi aktivitas:

- merevisi rekomendasi, status rekomendasi, dan pemetaan;
- memperbaiki dashboard;
- memperbaiki template review;
- memperbaiki coverage analyzer.

Instrumen/data:

- rekap hasil validasi;
- log revisi operasional;
- keputusan retain/revise/drop.

### 8. Uji Coba Operasional

Isi aktivitas:

- uji penggunaan dashboard oleh calon pengguna;
- uji coverage analysis dokumen kurikulum;
- pengukuran usability;
- wawancara/FGD pengguna;
- stability/ablation test jika tersedia.

Instrumen/data:

- SUS;
- lembar observasi tugas;
- kuesioner usefulness, transparency, trust;
- pedoman wawancara/FGD;
- gold curriculum coverage;
- laporan stability/ablation.

### 9. Revisi Produk Final

Isi aktivitas:

- menyusun produk final;
- memperbaiki dokumentasi;
- menyusun pass/fail report;
- menyusun batasan klaim;
- menyusun rekomendasi penggunaan.

Instrumen/data:

- laporan validasi final;
- laporan revisi final;
- kriteria kesiapan produk.

### 10. Diseminasi Terbatas dan Pelaporan

Isi aktivitas:

- menyusun laporan disertasi;
- menyusun panduan penggunaan;
- menyusun policy brief jika layak;
- menyusun publikasi ilmiah;
- menyajikan batasan klaim dan rencana pengembangan.

Instrumen/data:

- dokumen produk final;
- laporan evaluasi;
- rekomendasi implementasi terbatas.

---

## Hubungan Agile Development

Tambahkan loop kecil di lane **Pengembangan Teknis Agile** yang berada di bawah tahap:

- Pengembangan produk awal;
- Revisi produk utama;
- Revisi produk operasional;
- Revisi produk final.

Loop harus berbentuk:

`Rancang -> Bangun -> Uji -> Tinjau -> Revisi`

Gunakan panah putus-putus dari hasil validasi/uji coba ke proses revisi Agile.

Jangan membuat Agile menjadi alur utama penelitian. Borg and Gall tetap alur utama.

---

## Gate Validasi dan Keputusan

Tambahkan decision box pada tiga titik:

1. Setelah **Uji coba awal**:
   - Instrumen dan prototipe cukup layak untuk uji utama?
   - Jika tidak: revisi produk utama.

2. Setelah **Uji coba utama**:
   - Rekomendasi kompetensi valid sebagai `review_candidate`?
   - Jika tidak: revisi operasional atau `not_recommended_from_this_corpus`.

3. Setelah **Uji coba operasional**:
   - Produk layak digunakan sebagai bahan telaah kurikulum terbatas?
   - Jika tidak: revisi final dan batasi klaim.

Jangan gunakan label "produk terbukti efektif secara nasional" karena penelitian ini hanya mengarah pada kelayakan dan manfaat awal sesuai konteks uji coba.

---

## Instrumen yang Harus Terlihat dalam Diagram

Di lane **Pengumpulan Data dan Instrumen**, tampilkan kelompok instrumen berikut:

1. pedoman studi dokumen dan analisis korpus;
2. pedoman wawancara kebutuhan awal;
3. instrumen gold labeling sentence relevance;
4. instrumen gold labeling hard skill, soft skill, dan knowledge;
5. lembar validasi ahli kurikulum CBET/DACUM;
6. lembar validasi praktisi industri RPL/TI;
7. lembar validasi ahli NLP/AI;
8. lembar validasi ahli evaluasi;
9. lembar validasi KKNI/SKKNI/Spektrum/okupasi;
10. worksheet DACUM/CBET;
11. instrumen Delphi/expert panel;
12. instrumen dashboard usability;
13. instrumen coverage analyzer;
14. pedoman wawancara/FGD;
15. log revisi dan pass/fail report.

Kelompokkan instrumen agar diagram tidak terlalu ramai. Jangan menulis semua instrumen sebagai node individual kecil jika membuat diagram sulit dibaca. Boleh gunakan kotak kelompok seperti:

- `Instrumen validasi ahli`
- `Instrumen gold labeling`
- `Instrumen uji pengguna`
- `Dokumentasi audit dan revisi`

---

## Analisis Data yang Harus Terlihat

Di lane **Validasi, Analisis, dan Revisi**, tampilkan analisis berikut:

- validitas isi: Aiken's V, I-CVI/S-CVI, atau CVR jika relevan;
- reliabilitas antar-penilai: Cohen's Kappa atau Fleiss' Kappa;
- evaluasi ekstraksi: precision, recall, F1-score, confusion matrix;
- evaluasi coverage analyzer: precision, recall, F1-score;
- usability: SUS dan observasi tugas;
- trust/usefulness/transparency: statistik deskriptif;
- masukan ahli: analisis kualitatif dan log revisi;
- stabilitas/ablation: Jaccard top-k, rank correlation, cost/runtime report.

Gunakan label ringkas:

`Validitas isi`, `Reliabilitas antar-penilai`, `Metrik NLP`, `Usability`, `Analisis kualitatif`, `Stability/ablation`.

---

## Catatan Visual Wajib

Tambahkan catatan kecil di bawah diagram:

> Borg and Gall digunakan sebagai kerangka penelitian R&D, sedangkan Agile digunakan sebagai proses teknis internal pengembangan sistem. Validasi ahli, audit bukti, dan uji coba bertahap menentukan batas klaim produk.

Tambahkan catatan kedua jika ruang cukup:

> Korpus US/non-Indonesia hanya digunakan untuk validasi teknis. Klaim kurikulum Indonesia membutuhkan korpus Indonesia valid, pemetaan resmi, validasi ahli, dan uji stabilitas.

---

## Larangan Koneksi

Jangan buat koneksi berikut:

1. `Agile Development -> menggantikan Borg and Gall`
2. `Dashboard -> validitas rekomendasi`
3. `LLM -> keputusan kurikulum final`
4. `Korpus US/non-Indonesia -> klaim kurikulum Indonesia`
5. `Uji usability -> validitas kompetensi`
6. `Validasi teknis pipeline -> official_ready tanpa ahli`

---

## Format Output yang Diminta

Berikan output dalam tiga bentuk:

1. **Mermaid flowchart** yang siap dirender.
2. **Deskripsi layout PowerPoint/draw.io/Figma**.
3. **Prompt visual renderer** untuk membuat versi final akademik.

Sebelum Mermaid, tampilkan dua tabel:

### Tabel Node

| Node | Swimlane | Peran | Tahap Borg and Gall | Panah masuk | Panah keluar |
|---|---|---|---|---|---|

### Tabel Panah

| Dari | Ke | Jenis panah | Makna |
|---|---|---|---|

Jenis panah:

- `solid` = alur utama;
- `dashed` = feedback/revisi;
- `decision` = jalur keputusan;
- `blocked` = jalur tidak memenuhi syarat.

Pastikan semua label diagram ringkas, akademik, dan mudah dipahami pembaca pendidikan vokasi. Jika diagram terlalu padat, buat versi utama yang sederhana dan versi detail sebagai gambar tambahan.
