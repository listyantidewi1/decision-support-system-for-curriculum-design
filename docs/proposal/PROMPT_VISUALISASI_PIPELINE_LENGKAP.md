# Prompt Tunggal untuk Visualisasi Keseluruhan Pipeline

Gunakan prompt berikut di ChatGPT, Mermaid, draw.io, Figma, atau visual renderer lain untuk membuat **satu gambar akademik** yang memvisualisasikan keseluruhan pipeline penelitian.

---

## Prompt

Buat satu diagram akademik berjudul:

**"Pipeline Evidence-First Sistem Rekomendasi Kompetensi Bidang Rekayasa Perangkat Lunak"**

Diagram harus memvisualisasikan keseluruhan pipeline penelitian dari data lowongan kerja sampai rekomendasi kompetensi untuk telaah kurikulum. Gunakan gaya akademik formal, bersih, tidak dekoratif, cocok untuk proposal disertasi. Gunakan orientasi **A4 landscape** atau rasio **16:9 landscape**. Warna utama: biru tua, abu-abu, putih, dan satu aksen hijau atau oranye untuk status/gate. Jangan gunakan ilustrasi kartun, foto, ikon berlebihan, 3D, atau gradient dekoratif.

Diagram harus memakai **swimlane/domain** berikut dari atas ke bawah:

1. **Sumber Data dan Validitas Korpus**
2. **Pengolahan Bukti Pasar Kerja**
3. **Perumusan Kompetensi Berbasis Bukti**
4. **Pemetaan Kurikulum dan Okupasi**
5. **Validasi Manusia dan Kelayakan Klaim**
6. **Output untuk Telaah Kurikulum**

Alur utama harus bergerak **kiri ke kanan**. Jangan membalik arah panah. Panah putus-putus hanya boleh dipakai untuk umpan balik/revisi. Dashboard tidak boleh digambar sebagai sumber validitas; dashboard hanya media telaah. LLM tidak boleh langsung menuju rekomendasi final; LLM hanya membantu drafting/ringkasan dan harus melewati pemeriksaan bukti serta validasi ahli.

Gunakan label akademik yang mudah dipahami pembaca pendidikan vokasi. Jangan jadikan istilah teknis seperti JSONL, CSV, HDBSCAN, SBERT, OpenRouter, atau provider LLM sebagai label utama. Jika perlu, istilah teknis boleh muncul sebagai catatan kecil.

### Node dan Panah Wajib

Gunakan node berikut dan jangan ubah urutan panah utama:

1. **Korpus lowongan kerja**
   -> 2. **Audit validitas korpus dan lineage**
   -> 3. **Persiapan dan seleksi kalimat lowongan**
   -> 4. **Ekstraksi keterampilan, pengetahuan, dan soft skill**
   -> 5. **Klasifikasi jenis bukti**
   -> 6. **Pengelompokan tema kompetensi**
   -> 7. **Audit validitas klaster dan tema demand**
   -> 8. **Perumusan pernyataan kompetensi berbasis bukti**
   -> 9. **Pemeriksaan grounding dan anti-halusinasi**
   -> 10. **Deduplication dan pemeringkatan demand-first**
   -> 11. **Pemetaan KKNI, SKKNI, Spektrum, dan okupasi**
   -> 12. **Recommendation contract**
   -> 13. **Paket bukti untuk review ahli**
   -> 14. **Validasi ahli dan/atau Delphi**
   -> 15. **Rekomendasi kompetensi untuk telaah kurikulum**
   -> 16. **Dashboard, worksheet DACUM/CBET, dan coverage analysis**

### Gate/Keputusan Wajib

Tambahkan gate visual berbentuk diamond atau decision box:

1. Setelah **Audit validitas korpus dan lineage**:
   - Jika lineage tidak jelas atau korpus tidak sesuai klaim: `exploratory` atau `invalid_or_incomplete`.
   - Jika korpus US/non-Indonesia: hanya `engineering validation`, tidak boleh menuju `official_ready`.
   - Jika korpus Indonesia valid: boleh lanjut sebagai kandidat klaim Indonesia.

2. Setelah **Audit validitas klaster dan tema demand**:
   - `valid cluster` lanjut ke perumusan kompetensi.
   - `split_required`, `mixed_domain`, `out_of_domain`, atau `weak evidence` tidak boleh langsung dirumuskan menjadi rekomendasi.

3. Setelah **Pemeriksaan grounding dan anti-halusinasi**:
   - Bukti cukup: lanjut.
   - Bukti lemah/tidak cocok: `not_recommended_from_this_corpus`.

4. Setelah **Recommendation contract**:
   - `exploratory`
   - `review_candidate`
   - `official_ready`
   - `not_recommended_from_this_corpus`

5. Sebelum **official_ready**:
   - harus ada korpus Indonesia valid;
   - pemetaan resmi terkonfirmasi;
   - validasi ahli lulus;
   - uji stabilitas/ablation lulus.

### Fondasi Teoretis Wajib

Tambahkan dua kotak fondasi di bawah area perumusan kompetensi:

- **CBET/Competency-Based Curriculum**: memberi kriteria bahwa kompetensi harus observable, assessable, dan performance-based.
- **DACUM**: memberi struktur occupation -> duty area -> task evidence -> skills/knowledge/soft skill -> competency statement.

Panah dari CBET dan DACUM harus menuju **Perumusan pernyataan kompetensi berbasis bukti** dan **Paket bukti untuk review ahli**, bukan menuju data mentah.

### Posisi LLM

Tambahkan kotak kecil di dekat **Perumusan pernyataan kompetensi berbasis bukti**:

**LLM sebagai alat bantu drafting dan ringkasan bukti**

Panah yang benar:

`Bukti terpilih -> LLM sebagai alat bantu -> Draft pernyataan kompetensi -> Pemeriksaan grounding dan anti-halusinasi`

Larangan panah:

- Jangan buat panah `LLM -> official_ready`.
- Jangan buat panah `LLM -> kurikulum final`.
- Jangan buat LLM tampak sebagai validator.

### Output Akhir

Output akhir diagram harus menampilkan bahwa sistem menghasilkan:

- **evidence package**;
- **status rekomendasi**;
- **daftar kompetensi prioritas**;
- **worksheet DACUM/CBET**;
- **coverage analysis kurikulum**;
- **bahan telaah, bukan keputusan kurikulum otomatis**.

Tambahkan catatan kecil di kanan bawah:

> Catatan: rekomendasi sistem adalah bahan telaah kurikulum. Klaim resmi membutuhkan korpus Indonesia valid, pemetaan resmi, validasi ahli, dan uji stabilitas.

### Format yang Diminta

Berikan output dalam tiga bentuk:

1. **Mermaid flowchart** yang siap dirender.
2. **Deskripsi layout gambar** untuk PowerPoint/draw.io/Figma.
3. **Prompt visual renderer** untuk membuat versi final akademik.

Sebelum Mermaid, tampilkan tabel berikut:

| Node | Domain/Lane | Peran | Panah masuk | Panah keluar |
|---|---|---|---|---|

Lalu tampilkan tabel panah:

| Dari | Ke | Jenis panah | Makna |
|---|---|---|---|

Jenis panah hanya boleh:

- `solid` untuk alur utama;
- `dashed` untuk feedback/revisi;
- `blocked` untuk jalur yang tidak memenuhi syarat.

Pastikan diagram tidak terlalu ramai. Jika Mermaid terlalu padat, gunakan subgraph/swimlane dan label pendek. Teks dalam node harus singkat, akademik, dan mudah dibaca.
