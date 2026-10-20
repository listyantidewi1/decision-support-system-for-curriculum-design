# Prompt Visualisasi Produk Final Sistem Rekomendasi Kompetensi

Gunakan prompt ini untuk meminta ChatGPT membuat gambar akademik yang berorientasi ke **produk akhir yang sudah matang dan siap digunakan**, bukan kondisi pipeline pengembangan saat ini.

Prompt ini cocok untuk gambar proposal pada bagian:

- deskripsi produk yang dikembangkan;
- model konseptual produk;
- arsitektur sistem target;
- alur penggunaan produk;
- manfaat produk bagi pengembang kurikulum;
- posisi sistem dalam ekosistem pengembangan kurikulum vokasi.

---

## Prompt Utama

Buat satu atau beberapa diagram akademik untuk menggambarkan **produk akhir** penelitian berikut:

**Sistem Rekomendasi Kompetensi Bidang Rekayasa Perangkat Lunak untuk Pengembangan Kurikulum Berbasis Bukti**

Fokus gambar bukan pada status pipeline saat ini, bukan log eksperimen, bukan masalah teknis implementasi, dan bukan urutan debugging. Fokus gambar adalah **bagaimana sistem akan bekerja sebagai produk operasional yang siap digunakan oleh institusi pendidikan, ahli kurikulum, praktisi industri, dan pemangku kepentingan**.

Gunakan gaya akademik formal, bersih, dan mudah dipahami pembaca pendidikan vokasi. Gambar harus cocok untuk proposal disertasi. Gunakan warna terbatas: biru tua, abu-abu, putih, dan satu warna aksen. Hindari gaya poster promosi, kartun, 3D, ikon berlebihan, foto stok, dan gradient dekoratif.

---

## Orientasi Future-State

Visualisasikan sistem sebagai **target product state** setelah penelitian selesai dan produk telah melalui:

1. audit validitas korpus;
2. pemrosesan bukti lowongan kerja;
3. perumusan kompetensi berbasis CBET dan DACUM;
4. pemeriksaan bukti dan anti-halusinasi;
5. pemetaan KKNI/SKKNI/Spektrum/okupasi;
6. validasi ahli;
7. uji coba pengguna;
8. revisi produk;
9. pelaporan batas klaim.

Jangan menggambarkan sistem sebagai eksperimen yang belum stabil. Namun, tetap tampilkan bahwa sistem memiliki tata kelola validasi dan status rekomendasi agar tidak terlihat sebagai mesin yang otomatis menentukan kurikulum.

Gunakan bahasa:

- "Sistem operasional"
- "Paket bukti"
- "Rekomendasi kompetensi tervalidasi"
- "Bahan telaah kurikulum"
- "Pengembang kurikulum"
- "Ahli dan praktisi"
- "Konteks Indonesia"
- "Keputusan berbasis bukti"

Hindari bahasa yang terlalu teknis sebagai label utama:

- JSONL;
- CSV;
- HDBSCAN;
- SBERT;
- OpenRouter;
- provider LLM;
- prompt;
- embedding;
- API.

Istilah teknis boleh muncul sebagai catatan kecil jika perlu, bukan sebagai elemen utama gambar.

---

## Prinsip Produk Akhir

Diagram harus menunjukkan prinsip berikut:

1. Sistem membantu pengembang kurikulum menemukan kompetensi yang paling banyak diminta pasar kerja.
2. Sistem tidak hanya menampilkan daftar skill, tetapi menyajikan **competency statement** yang dapat ditelaah untuk kurikulum.
3. Setiap rekomendasi disertai evidence package, job prevalence, source evidence, status validitas, dan alasan pemeringkatan.
4. Sistem mendukung kerangka **CBET**: kompetensi harus observable, assessable, dan performance-based.
5. Sistem mendukung kerangka **DACUM**: occupation -> duty area -> task evidence -> skills/knowledge/soft skill -> competency statement -> curriculum recommendation.
6. Sistem memisahkan rekomendasi teknis inti, proses/leadership, employability/support, dan out-of-domain.
7. Sistem membedakan status rekomendasi: exploratory, review candidate, official ready, atau not recommended from this corpus.
8. Keputusan kurikulum tetap berada pada manusia: ahli kurikulum, praktisi industri, sekolah/program studi, dan pemangku kebijakan.
9. Dashboard adalah media telaah dan eksplorasi, bukan sumber validitas.
10. Produk akhir harus siap mendukung penyusunan/revisi kurikulum, coverage analysis, review ahli, dan dokumentasi akademik.

---

## Gambar Utama yang Diminta

Buat **satu gambar utama** berjudul:

**"Model Produk Akhir Sistem Rekomendasi Kompetensi Berbasis Bukti"**

Gambar ini harus menunjukkan hubungan antara:

1. sumber bukti pasar kerja;
2. mesin rekomendasi kompetensi;
3. tata kelola validasi;
4. antarmuka pengguna;
5. pengguna/stakeholder;
6. output untuk pengembangan kurikulum.

Gunakan struktur visual lima zona dari kiri ke kanan:

### Zona 1. Sumber Bukti

Label utama:

- Korpus lowongan kerja Indonesia
- Dokumen resmi KKNI/SKKNI/Spektrum
- Masukan ahli dan praktisi
- Dokumen kurikulum institusi

Makna:

Sistem menggunakan bukti pasar kerja, rujukan resmi, dan dokumen kurikulum sebagai masukan.

### Zona 2. Mesin Rekomendasi Kompetensi

Label utama:

- Audit validitas korpus
- Ekstraksi keterampilan, pengetahuan, dan soft skill
- Pengelompokan tema kompetensi
- Perumusan competency statement
- Pemeriksaan bukti dan anti-halusinasi
- Pemeringkatan demand-first
- Pemetaan KKNI, SKKNI, Spektrum, dan okupasi

Makna:

Mesin mengubah bukti lowongan kerja menjadi rekomendasi kompetensi yang traceable dan dapat ditelaah.

### Zona 3. Tata Kelola Validasi

Label utama:

- Review ahli kurikulum
- Review praktisi industri
- Review ahli NLP/AI
- Review ahli evaluasi
- Delphi/expert panel
- Stability dan ablation report
- Recommendation contract

Makna:

Validasi manusia dan laporan ilmiah menentukan batas klaim dan status rekomendasi.

### Zona 4. Antarmuka Produk

Label utama:

- Dashboard publik
- Dashboard administratif
- Evidence package
- Worksheet DACUM/CBET
- Coverage analyzer
- Export laporan kurikulum

Makna:

Pengguna menelusuri rekomendasi, memeriksa bukti, membandingkan dengan kurikulum, dan menyiapkan dokumen telaah.

### Zona 5. Output dan Keputusan Kurikulum

Label utama:

- Daftar kompetensi prioritas
- Status rekomendasi
- Peta coverage kurikulum
- Rekomendasi gap kurikulum
- Bahan workshop kurikulum
- Bahan kebijakan terbatas
- Revisi kurikulum oleh institusi

Makna:

Output sistem menjadi bahan telaah dan keputusan kurikulum oleh manusia/institusi.

---

## Arah Panah Wajib

Gunakan alur utama kiri ke kanan:

`Sumber Bukti -> Mesin Rekomendasi Kompetensi -> Tata Kelola Validasi -> Antarmuka Produk -> Output dan Keputusan Kurikulum`

Tambahkan panah umpan balik putus-putus:

1. `Review ahli dan praktisi -.-> Mesin Rekomendasi Kompetensi`
2. `Umpan balik pengguna -.-> Antarmuka Produk`
3. `Keputusan kurikulum institusi -.-> Dokumen kurikulum institusi`

Larangan panah:

1. Jangan buat panah `LLM -> keputusan kurikulum`.
2. Jangan buat panah `Dashboard -> validitas rekomendasi`.
3. Jangan buat panah `Korpus US/non-Indonesia -> klaim kurikulum Indonesia`.
4. Jangan buat panah `Output sistem -> kurikulum final otomatis`.
5. Jangan buat panah `Ranking demand -> official ready` tanpa validasi ahli.

---

## Posisi LLM dalam Gambar Produk

Jika LLM perlu ditampilkan, posisikan sebagai subkomponen kecil di dalam **Mesin Rekomendasi Kompetensi**, dengan label:

**Asisten perumusan dan ringkasan bukti**

Jangan menggunakan label "AI decision maker", "automatic curriculum generator", atau "final recommendation engine".

Panah yang benar:

`Bukti terpilih -> Asisten perumusan dan ringkasan bukti -> Draft competency statement -> Pemeriksaan bukti -> Review ahli`

---

## Bahasa Label yang Disarankan

Gunakan label yang ramah untuk pembaca akademik pendidikan:

| Hindari label teknis | Gunakan label akademik |
|---|---|
| raw CSV | Korpus lowongan kerja |
| JSONL extraction | Ekstraksi bukti kompetensi |
| clustering model | Pengelompokan tema kompetensi |
| LLM generation | Perumusan kompetensi berbasis bukti |
| grounding gate | Pemeriksaan kesesuaian bukti |
| deduplication | Penggabungan rekomendasi sejenis |
| priority score | Pemeringkatan demand-first |
| validation package | Paket validasi ahli |
| dashboard app | Antarmuka telaah |
| official_ready | Siap dipertimbangkan sebagai rekomendasi resmi |

---

## Status Rekomendasi yang Harus Muncul

Tambahkan legenda kecil:

- **Exploratory**: sinyal awal yang berguna, belum siap menjadi rekomendasi formal.
- **Review candidate**: kandidat rekomendasi yang memiliki bukti dan perlu/telah ditelaah ahli.
- **Official ready**: hanya untuk rekomendasi yang lulus korpus Indonesia valid, pemetaan resmi, validasi ahli, dan stabilitas.
- **Not recommended from this corpus**: tidak cukup bukti dari korpus yang digunakan.

Legenda ini penting agar gambar tidak terlihat seperti sistem otomatis mengesahkan kompetensi.

---

## Caption Akademik

Gunakan caption:

> Gambar X. Model produk akhir sistem rekomendasi kompetensi berbasis bukti. Sistem mengintegrasikan bukti pasar kerja, rujukan kurikulum dan okupasi, validasi ahli, serta antarmuka telaah untuk menghasilkan rekomendasi kompetensi yang dapat digunakan sebagai bahan pengembangan kurikulum.

---

## Output yang Diminta dari ChatGPT

Berikan hasil dalam empat bagian:

### 1. Tabel Node Produk

| Node | Zona | Peran dalam produk akhir | Input | Output |
|---|---|---|---|---|

### 2. Tabel Panah

| Dari | Ke | Jenis panah | Makna |
|---|---|---|---|

Jenis panah:

- `solid`: alur utama produk;
- `dashed`: umpan balik/revisi;
- `gate`: keputusan validasi/status;
- `blocked`: jalur yang tidak boleh langsung menuju rekomendasi.

### 3. Mermaid Draft

Buat Mermaid flowchart yang siap dirender. Gunakan subgraph untuk lima zona:

1. Sumber Bukti
2. Mesin Rekomendasi Kompetensi
3. Tata Kelola Validasi
4. Antarmuka Produk
5. Output dan Keputusan Kurikulum

### 4. Prompt Visual Renderer

Buat prompt visual renderer seperti:

> Academic product model diagram, five horizontal zones from left to right, clean white background, navy and gray palette, one accent color for validation gates, Indonesian labels, dissertation proposal style, readable A4 landscape, vector diagram, no cartoons, no 3D, no decorative gradients, clear arrows, dashboard shown as review interface not decision maker, LLM shown only as assisted drafting component, final output shown as curriculum review material not automatic curriculum decision.

---

## Catatan Penting

Tambahkan catatan kecil di kanan bawah gambar:

> Produk ini mendukung keputusan kurikulum berbasis bukti. Keputusan final tetap memerlukan telaah ahli dan kebijakan institusi.

Jika gambar terlalu padat, buat dua versi:

1. **Versi ringkas** untuk Bab I/Bab II sebagai model produk akhir.
2. **Versi detail** untuk Bab III sebagai arsitektur operasional produk.

Jangan menggambar status pengembangan saat ini. Gambar harus menampilkan sistem sebagai produk akhir yang diharapkan setelah seluruh validasi penelitian selesai.
