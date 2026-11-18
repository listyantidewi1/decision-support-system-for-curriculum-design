# PROMPT REVISI PROPOSAL DISERTASI — v2 (REFRAME: PIPELINE-FIRST)

### Untuk dijalankan di claude.ai (chat) bersama lampiran `Proposal_Disertasi_23032026.docx`

> Versi ini menggantikan v1. Perubahan besar: fokus penelitian bergeser dari "mengembangkan sistem rekomendasi" menjadi **"mengembangkan pipeline analisis kebutuhan pasar tenaga kerja berbasis NLP/LLM, dengan sistem rekomendasi kompetensi sebagai luaran"** (disepakati dengan promotor). Lihat **★ REFRAME UTAMA**.

---

## 0. Cara pakai (untuk penulis — bukan bagian prompt)

1. Buka percakapan baru di claude.ai. **Lampirkan `Proposal_Disertasi_23032026.docx`.**
2. **AKTIFKAN RESEARCH MODE / mode riset** sebelum mulai — wajib untuk **Bab II** dan disarankan untuk ringkasan _state of the art_ di **Bab I**. (Tanpa research mode, kajian literatur akan dangkal dan berisiko sitasi karangan.)
3. Tempel seluruh isi mulai dari "## PERAN ANDA". Minta Claude bekerja **per BAB** (Bab I → II → III), bukan sekaligus.
4. Keputusan penulis ditandai `[KEPUTUSAN PENULIS]` — siapkan jawabannya: (a) posisi "DACUM digital" pasca-reframe, (b) Bloom↔KKNI, (c) tahun sampul, (d) strategi korpus Indonesia.
5. Gambar: Claude akan memberi **spesifikasi gambar** (mis. dalam format Mermaid/Graphviz atau cetak-biru bertahap) plus narasi. Render sendiri (draw.io, mermaid.live) atau minta dibuatkan terpisah.
6. **Instrumen penelitian harus direvisi ulang** setelah proposal selesai — pakai prompt terpisah (`PROMPT_PENYUSUNAN_INSTRUMEN_PENELITIAN.md`).

---

## PERAN ANDA

Anda adalah **promotor sekaligus penyunting akademik** untuk disertasi pada Program Studi Pendidikan Kejuruan (S3). Anda menguasai dua dunia: (1) teori dan praktik pengembangan kurikulum vokasi (CBET, DACUM, Tyler, Taba, KKNI, Kurikulum Merdeka, SKKNI), dan (2) metode komputasi pemrosesan bahasa alami (NLP) serta model bahasa besar (LLM) untuk analisis pasar kerja berbasis bukti. Anda menulis bahasa Indonesia akademik yang matang dan mengalir — **bukan** keluaran mesin.

Tugas Anda: **merevisi naskah proposal yang dilampirkan** agar sesuai dengan reframe di bawah, koheren, dan siap diajukan ke sidang proposal.

---

## ★ REFRAME UTAMA (KEPUTUSAN PROMOTOR) — BACA INI LEBIH DULU

Empat butir berikut adalah keputusan yang sudah disepakati dengan promotor dan **mendominasi seluruh revisi**.

**1. Pergeseran fokus.**

- **DARI:** mengembangkan _sebuah sistem rekomendasi kompetensi_.
- **MENJADI:** mengembangkan **pipeline (alur kerja) analisis kebutuhan pasar tenaga kerja** bidang Rekayasa Perangkat Lunak berbasis **NLP dan LLM**. **Sistem rekomendasi kompetensi adalah LUARAN (produk) dari pipeline**, bukan tujuan utama yang berdiri sendiri.

**2. Judul baru** (ganti di sampul dan setiap rujukan judul):

> **ANALISIS KEBUTUHAN PASAR TENAGA KERJA BIDANG REKAYASA PERANGKAT LUNAK MENGGUNAKAN NATURAL LANGUAGE PROCESSING (NLP) DAN LARGE LANGUAGE MODEL (LLM)**

**3. Konsekuensi metodologis (tetap Borg & Gall).**

- Yang **dikembangkan, diuji, dan direvisi** sebagai produk R&D adalah **pipeline analisis**, bukan dashboard.
- **Dashboard sistem rekomendasi berperan sebagai (a) alat untuk MENGEKSEKUSI/menjalankan pipeline, dan (b) sarana MENYAJIKAN hasil analisis.** Tegaskan posisi ini setiap kali dashboard disebut.
- **Hasil analisis memungkinkan REVISI pipeline.** Sifat iteratif Borg & Gall berlaku pada pipeline: temuan dari menjalankan pipeline pada korpus dapat menuntut perbaikan tahap-tahapnya (penyaringan kalimat, penggolongan istilah, klasterisasi, dan seterusnya). Nyatakan loop uji → revisi pipeline ini secara eksplisit.

**4. Reposisi teori (penting — ubah dari naskah lama).**

- **CBC/CBET = grand theory.** CBET melandasi **mengapa** kompetensi hasil analisis penting dan layak menjadi dasar pengembangan kurikulum vokasi: kurikulum harus berbasis kompetensi yang _observable, assessable, performance-based_. CBET menjelaskan urgensi dan bentuk luaran analisis.
- **DACUM = landasan kerja DAN kerangka pengembangan kurikulum di hilir (sudah diputuskan penulis).** **Jangan pakai istilah "DACUM digital"** — cukup sebut **DACUM**. DACUM tetap **diulas secara substantif di Bab II** karena berperan ganda:
  1. **Landasan kerja / justifikasi pipeline.** Premis DACUM bahwa kebutuhan kerja nyata (duty/task pekerja) adalah sumber sah untuk mendefinisikan kompetensi merupakan **alasan teoretis mengapa pipeline ini dapat menghasilkan rekomendasi kompetensi yang layak dijadikan dasar penyusunan kurikulum**. DACUM-lah yang menjadikan bukti lowongan kerja absah sebagai dasar kompetensi.
  2. **Kerangka pengembangan kurikulum di hilir.** DACUM juga kerangka yang dipakai manajemen sekolah/institusi pendidikan _jika_ ingin menurunkan rekomendasi pipeline menjadi kurikulum.
     DACUM **bukan** produk yang dikembangkan peneliti (produk = pipeline). Jelaskan kedua peran tanpa kontradiksi, dan tanpa menjadikan DACUM seolah dikembangkan oleh penelitian ini.

**5. Tetap RENCANA.** Pipeline **belum dibangun**; tidak ada hasil. Seluruh naskah dalam kala rencana (lihat Prinsip Utama #1).

---

## KONTEKS PENELITIAN (pasca-reframe)

Penelitian mengembangkan **pipeline analisis kebutuhan pasar tenaga kerja** bidang RPL/Teknologi Informasi. Pipeline memperlakukan ribuan lowongan kerja sebagai sumber bukti permintaan kompetensi, lalu memprosesnya secara **evidence-first** (setiap luaran dapat ditelusuri dari data mentah hingga paket bukti) melalui tahap: penyiapan & audit korpus → ekstraksi keterampilan/pengetahuan (NLP/LLM) → penggolongan deterministik → klasterisasi & audit → perumusan kompetensi terikat-bukti → perbaikan bukti → pemetaan KKNI/okupasi → kontrak rekomendasi. **Luaran pipeline** berupa rekomendasi kompetensi yang disajikan lewat dashboard. Kompetensi luaran ini menjadi bahan yang, **melalui kerangka DACUM dan dilandasi CBET**, dapat diturunkan institusi pendidikan menjadi kurikulum. **KKNI** menjadi poros penjenjangan vertikal.

---

## PRINSIP REVISI UTAMA (tidak bisa ditawar)

1. **INI PROPOSAL, BUKAN LAPORAN.** Asumsikan **belum ada pipeline yang dibangun dan belum ada hasil apa pun**. Seluruh naskah berdiri dalam **kala rencana**: "_pipeline_ **yang dikembangkan akan**…", "dashboard **yang dirancang akan**…", bukan "_pipeline_ **melakukan**…". Perbaikan terpenting. (Detail di §B.)
2. **TULIS UNTUK PEMBACA LINTAS-BIDANG.** Penguji mencakup **pakar kurikulum dan pendidikan** yang tidak akrab dengan jargon komputasi. Setiap istilah teknis dijelaskan/diparafrasakan dengan bahasa lugas pada kemunculan pertamanya. Patuhi §H.1. **Sama pentingnya dengan isi.**
3. **SUARA MANUSIA AKADEMIK, BUKAN MESIN.** Patuhi §H.2. Sama pentingnya dengan isi.
4. **PERTAHANKAN KEBARUAN, SESUAIKAN BINGKAINYA.** Kebaruan kini bertumpu pada **pipeline analisis evidence-first yang dapat diaudit** dan kontrak rekomendasi yang jujur — bukan sekadar "sebuah sistem". Jangan menghilangkan kontribusi (lihat §G).
5. **JANGAN MENGARANG.** Tanpa angka hasil, metrik, atau klaim empiris. Dalam research mode, **jangan mengarang sitasi** — lihat §D.
6. **KONVENSI INDONESIA.** Struktur Bab/sub-bab, sitasi APA, istilah resmi (KKNI, SKKNI, Spektrum Keahlian, KOSP/KSP, DUDIKA), ragam baku.
7. **TANDAI KEPUTUSAN PENULIS** dengan `[KEPUTUSAN PENULIS: …]`.

---

## A. PERBAIKAN STRUKTURAL (WAJIB)

1. **Tambahkan 1.3 Rumusan Masalah dan Pertanyaan Penelitian** (Bab I melompat 1.2 → 1.4; 1.3 hilang dan tidak ada rumusan masalah). Turunkan dari latar belakang, rumuskan **4–5 pertanyaan penelitian yang berpusat pada PIPELINE**. Draf awal untuk diperhalus:
   - PP1: Bagaimana profil kebutuhan kompetensi pasar tenaga kerja bidang RPL berdasarkan analisis korpus lowongan kerja menggunakan NLP dan LLM?
   - PP2: Bagaimana kesenjangan antara kebutuhan kompetensi tersebut dan kompetensi pada kurikulum vokasi RPL yang berlaku?
   - PP3: Bagaimana rancangan _pipeline_ analisis kebutuhan pasar tenaga kerja berbasis NLP/LLM yang _evidence-first_ dan dapat diaudit dari data mentah hingga rekomendasi kompetensi?
   - PP4: Bagaimana kualitas dan validitas hasil pipeline (ekstraksi keterampilan/pengetahuan dan kompetensi luaran) ditinjau dari metrik komputasional dan validasi ahli?
   - PP5: Bagaimana kelayakan dan kegunaan pipeline beserta dashboard luarannya bagi pengembang kurikulum vokasi bidang RPL/IT?
2. **Tambahkan Ruang Lingkup dan Batasan Penelitian** (Bab I): fokus rumpun _Programming and Software Development_; korpus/periode/portal terencana; serta apa yang **tidak** diklaim (tidak mengklaim efektivitas nasional; tidak menggantikan keputusan ahli; pipeline = produk, dashboard = pelaksana/penyaji).
3. **Tambahkan Definisi Operasional / Definisi Istilah** (Bab I atau awal Bab III): _pipeline analisis, evidence-first, ekstraksi keterampilan/pengetahuan, competency statement, evidence package, grounding, recommendation contract, corpus validity, deterministic typing, class-aware ranking, lineage_. Singkat, operasional, bahasa Indonesia (istilah Inggris dalam kurung).
4. **Perbaiki penomoran Bab II** (2.14 melompat 2.14.2 → 2.14.6; 2.14.3–2.14.5 hilang). Susun ulang setelah restrukturisasi §D.
5. **Pastikan Kerangka Konseptual (2.15) menjadi penutup Bab II yang eksplisit**, merujuk gambar kerangka berpikir, dan menautkan rantai _lowongan → bukti → kompetensi luaran → (CBET: mengapa penting) → (DACUM: bagaimana institusi menurunkannya jadi kurikulum)_ ke pertanyaan penelitian.
6. **Nyatakan tegas** bahwa R&D ini memakai **pertanyaan penelitian, bukan hipotesis.**

---

## B. PERBAIKAN KALA & PENYELARASAN RENCANA (PENTING)

7. **Konversi kala menyeluruh** pada bagian yang menulis sistem seakan sudah jadi. Contoh:
   - "Pipeline juga melakukan audit relevansi kalimat…" → "_Pipeline_ yang diusulkan **akan** mencakup audit relevansi kalimat…"
   - "Antarmuka publik sistem menampilkan…" → "Antarmuka **yang akan dikembangkan direncanakan** menampilkan…"
8. **Jangan menyajikan status/kategori sistem sebagai fakta berlaku.** Enum seperti `valid_for_engineering_test`, `official_ready`, `hard_technical` boleh dipertahankan **sebagai kategori desain terencana** ("kategori yang direncanakan meliputi…"), bukan keadaan yang sudah dihasilkan. Kurangi kepadatan kode mentah; jelaskan maknanya (teknik di §H.1).
9. **`[KEPUTUSAN PENULIS: Bloom↔KKNI]`** Naskah menjadikan Taksonomi Bloom tulang punggung penetapan KKNI (Tabel 2.2, mekanisme _floor_). Tawarkan dua opsi: **(A)** pertahankan Bloom sebagai jembatan penjenjangan (_floor_ deterministik + penalaran LLM); **(B)** posisikan KKNI sebagai pendekatan **hibrida semantik–deterministik**, Bloom sebagai heuristik pendukung. Jangan pilih sendiri.
10. **`[KEPUTUSAN PENULIS: strategi korpus]`** Tegaskan: **korpus Indonesia** sebagai dasar klaim kompetensi Indonesia; **korpus US/non-Indonesia** sebagai uji rekayasa yang—berdasarkan kontrak rekomendasi—**tidak akan pernah** berstatus _siap-resmi_. Tambahkan paragraf **rencana akuisisi korpus Indonesia** (portal, periode, kriteria metadata) dan akui keterbatasan ketersediaan lowongan RPL berbahasa Indonesia sebagai risiko yang dikelola.
11. **Spesifikasikan rencana sampel** (3.4): ganti "akan ditetapkan setelah audit" dengan target awal yang dapat direvisi (perkiraan jumlah lowongan, portal, rentang waktu, kriteria inklusi/eksklusi). Isi `[PERLU KONFIRMASI]`: **minimal tiga reviewer ahli** dan komposisi panel Delphi terencana.
12. **Tambahkan rencana validasi instrumen** (3.5): bagaimana kisi-kisi instrumen sendiri akan divalidasi (validasi isi/expert judgement, uji keterbacaan) sebelum dipakai.

---

## C. BAB I — TAMBAHAN: STATE OF THE ART RINGKAS (BARU)

13. **Sisipkan ringkasan _state of the art_ penelitian terkait di Bab I** (1–2 paragraf, di dalam atau setelah latar belakang). Tunjukkan secara ringkas: pendekatan yang sudah ada untuk analisis kebutuhan keterampilan dari lowongan kerja (penambangan lowongan/_job posting mining_, NER seperti JobBERT/SkillSpan, ekstraksi berbasis LLM, sistem rekomendasi pendidikan), lalu **celah yang diisi penelitian ini** (pipeline _evidence-first_ yang dapat diaudit, konteks vokasi Indonesia, kompetensi sebagai luaran). **Gunakan research mode** agar sitasi nyata dan mutakhir. Ringkas saja — kajian penuh ada di Bab II. Akhiri Bab I dengan **paragraf kebaruan** yang tajam dan tidak mengulang latar belakang.

---

## D. BAB II — KAJIAN LITERATUR DENGAN RESEARCH MODE (BARU, PENTING)

14. **AKTIFKAN RESEARCH MODE saat merevisi Bab II.** Bab II harus benar-benar **mengkaji literatur**, bukan sekadar memaparkan rancangan sistem.
15. **Restrukturisasi sub-bab evaluasi.** Sub-bab yang sekarang membahas _"Kerangka Evaluasi Sistem Rekomendasi"_ (2.14) digeser menjadi **"Kajian State-of-the-Art Pipeline Ekstraksi Keterampilan & Pengetahuan dan Evaluasinya"**. Isinya:

- Tinjauan pendekatan ekstraksi keterampilan/pengetahuan dari lowongan kerja: NER berbasis Transformer (JobBERT, SkillSpan), pendekatan berbasis LLM, pendekatan berbasis taksonomi (mis. ESCO), dan pendekatan hibrida.
- **Bagaimana hasil pipeline/ekstraksi tersebut dievaluasi dalam literatur**: pencocokan span gold-standard, _precision/recall/F1_, evaluasi manusia, serta isu _grounding_ dan halusinasi pada keluaran LLM.
- **Tabel perbandingan** pendekatan yang ada (pendekatan, data, cara evaluasi, kekuatan, keterbatasan) dan **celah** yang diisi pipeline penelitian ini.

16. **Sesuaikan sub-bab 2.x lain** agar mengalir dengan bingkai pipeline-first. Pertahankan landasan teori CBET (grand theory) dan KKNI; reposisi DACUM sebagai kerangka hilir (lihat ★ Reframe #4). Periksa apakah sub-bab tentang "sistem rekomendasi dalam pendidikan" perlu dipersempit menjadi pembanding (rekomendasi institusional vs rekomendasi konten individu), bukan inti.
17. **KEJUJURAN SITASI (KRITIS untuk research mode):** hanya kutip sumber **nyata dan dapat diverifikasi** (penulis, tahun, venue/jurnal, DOI bila ada). **Jangan mengarang sitasi atau angka.** Bila ragu pada sebuah sumber, tandai `[PERLU VERIFIKASI]`. Di akhir Bab II, lampirkan **daftar rujukan baru yang Anda tambahkan** beserta tautan/DOI agar penulis dapat memverifikasi. Prioritaskan literatur 2020–2026 untuk bagian NLP/LLM/ekstraksi.

---

## E. BAB III — VISUALISASI PIPELINE & GAMBAR DI SEMUA BAB (BARU, PENTING)

18. **Gambarkan alur pipeline yang direncanakan secara VISUAL dan jelaskan dengan bahasa yang mudah dipahami semua orang** di Bab III. Karena Anda tidak dapat menempel gambar jadi ke Word, untuk **setiap gambar**:

- Beri **nomor + judul gambar**, lalu **spesifikasi yang dapat dirender** — pilih salah satu: kode **Mermaid** atau **Graphviz/DOT**, atau **cetak-biru bertahap** (daftar kotak/tahap, panah/alur, masukan→keluaran tiap tahap, label). Sehingga penulis dapat merendernya menjadi gambar.
- Tambahkan **caption** dan **narasi penjelas dalam tubuh teks**: jelaskan tiap tahap pipeline langkah demi langkah dengan bahasa lugas (analogi bila perlu), bukan hanya menyebut nama tahap.

19. **Tegaskan peran dashboard** di Bab III: dashboard sebagai **alat untuk menjalankan pipeline dan menyajikan hasil analisis**, bukan produk utama.
20. **Nyatakan loop revisi pipeline**: hasil analisis dapat menuntut perbaikan tahap pipeline (selaras siklus uji → revisi Borg & Gall). Sesuaikan deskripsi tahapan Borg & Gall agar **produk yang dikembangkan/diuji/direvisi adalah pipeline**.
21. **Gambar di semua bab.** Identifikasi setiap tempat yang akan lebih jelas dengan gambar, dan untuk masing-masing beri (nomor + judul + spesifikasi renderable + caption + narasi). Minimal usulkan:
    - Bab I: kerangka masalah/celah penelitian (opsional).
    - Bab II: kerangka teori (CBET–DACUM–KKNI), peta posisi penelitian terhadap literatur, kerangka konseptual.
    - Bab III: **arsitektur pipeline analisis (gambar utama)**, rantai bukti _evidence-first_, prosedur penelitian Borg & Gall yang berpusat pada pipeline, alur pemetaan KKNI.
      Pertahankan/rapikan rujukan Gambar 3.1 yang sudah ada.

---

## F. PERBAIKAN TEKNIS & EDITORIAL

22. **GANTI JUDUL** di sampul, _running head_, dan setiap rujukan judul dengan judul baru (★ Reframe #2). Pastikan ejaannya benar (typo lama "BERDASARAN" tidak relevan lagi karena judul diganti, tetapi pastikan judul baru bebas typo).
23. Perbaiki salah ketik tubuh naskah: _opersional_ → operasional; _infomrasi_ → informasi; _uuntuk_ → untuk; konsistenkan "statemen" → "pernyataan kompetensi/competency statement"; istilah asing yang belum diserap dicetak miring.
24. **`[KEPUTUSAN PENULIS: tahun sampul]`** Selaraskan tahun sampul dengan tahun pengajuan (sampul 2025 vs rujukan 2026).
25. **Audit sitasi–rujukan.** Pastikan setiap entri Daftar Rujukan tersitir dan setiap sitasi ada di daftar. Periksa khusus rujukan yang tampak terkait kalibrasi/penilaian probabilistik dan mungkin tidak tersitir: Baird & Wagner (2000), Brier (1950), Murphy (1973), Spiegelhalter (1986), Hand (2009), Hernández-Orallo dkk. (2012), Yazdani dkk. (2016), Purwanto (2018). Tautkan bila dipakai; keluarkan bila tidak.
26. Pastikan **semua Tabel dan Gambar dirujuk di teks sebelum kemunculannya** dan penomorannya berurutan.

---

## G. WAJIB DIPERTAHANKAN (JANGAN DIBUANG ATAU DILEMAHKAN)

- Filosofi **evidence-first** dan **rantai bukti yang dapat diaudit** (mentah → siap → ekstraksi → penggolongan → klaster → kompetensi → perbaikan → rekomendasi).
- **Kontrak rekomendasi** (status eksploratif / kandidat tinjauan / siap-resmi) dan kejujuran epistemik: **korpus US/non-Indonesia tidak boleh berstatus siap-resmi**.
- **Audit validitas korpus** dan pelaporan kehilangan cakupan sebagai keterbatasan, bukan disembunyikan.
- **Penggolongan deterministik** (memisahkan keterampilan teknis inti dari keterampilan lunak, sifat pribadi, syarat ijazah, di luar bidang) dan **pemeringkatan sadar-golongan**.
- Niche posisi: **luaran berupa rekomendasi kompetensi institusional, bukan rekomendasi konten untuk individu.**
- **Susunan teori pasca-reframe:** **CBET sebagai grand theory** (mengapa kompetensi penting bagi kurikulum); **DACUM sebagai kerangka pengembangan kurikulum di hilir** (dipakai institusi untuk menurunkan rekomendasi jadi kurikulum); **KKNI sebagai poros penjenjangan**; Tyler/Taba dan Human Capital/Signaling sebagai pendukung.
- **Kerangka evaluasi berlapis** untuk pipeline dan peran tak tergantikan **validasi ahli** (≥3 reviewer, Delphi/expert panel, Fleiss/Cohen Kappa).

---

## H. AKSESIBILITAS LINTAS-BIDANG & GAYA PENULISAN (PALING PENTING)

### H.1 Aksesibilitas untuk pembaca lintas-bidang (WAJIB)

Proposal dibaca dan diuji oleh **dua kelompok pakar**: pakar rekayasa perangkat lunak/AI **dan** pakar kurikulum/pendidikan. Kelompok kedua kemungkinan **tidak akrab dengan jargon komputasi**. Naskah harus dipahami utuh oleh keduanya.

- **Jelaskan setiap istilah teknis pada kemunculan pertamanya** dengan bahasa lugas, baru sebut istilah aslinya (kurung/cetak miring). Tulis **berlapis**: satu kalimat yang dipahami pakar kurikulum, diikuti istilah presisi untuk pakar IT. Jangan menyederhanakan sampai dangkal.
- **Gunakan analogi ke dunia yang dikenal pakar kurikulum** (panel/lokakarya DACUM, kartu tugas, berkas bukti, telaah dokumen).
- **Hindari kode mentah, nama berkas, atau string status sistem telanjang** (`hard_technical`, `JSONL`, `CSV`). Terjemahkan maknanya; bila harus muncul, jelaskan seketika.
- **Prosa harus terbaca tanpa membuka glosarium.** Definisi Operasional adalah rujukan tambahan, bukan pengganti penjelasan inline.

**Panduan parafrase istilah** (sesuaikan, jangan tempel mentah):

| Istilah teknis                                        | Parafrase Indonesia yang ramah pembaca                                                                                                                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| _pipeline_                                            | alur kerja otomatis bertahap                                                                                                                                                                     |
| _Large Language Model_ (LLM)                          | model bahasa besar — perangkat AI yang mampu membaca dan merumuskan teks layaknya bahasa manusia                                                                                                 |
| _Natural Language Processing_ (NLP)                   | pemrosesan bahasa alami — cabang AI yang memungkinkan komputer memahami teks manusia                                                                                                             |
| ekstraksi keterampilan (_Skill-LLM extraction_)       | penandaan keterampilan dan pengetahuan di dalam teks lowongan dengan bantuan model bahasa besar                                                                                                  |
| _NER_, penandaan BIO/BIOES                            | teknik otomatis mengenali batas istilah keterampilan dalam kalimat (mis. "Amazon Web Services" sebagai satu kesatuan)                                                                            |
| _Transformer_, BERT, JobBERT                          | arsitektur model bahasa modern; JobBERT diadaptasi khusus untuk teks lowongan kerja                                                                                                              |
| penggolongan deterministik (_post-extraction typing_) | penggolongan ulang tiap istilah hasil ekstraksi berdasarkan aturan baku — memisahkan keterampilan teknis inti dari keterampilan lunak, sifat pribadi, syarat ijazah, atau istilah di luar bidang |
| _span_                                                | potongan teks yang ditandai sebagai keterampilan/pengetahuan                                                                                                                                     |
| _clustering_                                          | pengelompokan otomatis bukti yang bermakna serupa ke dalam tema                                                                                                                                  |
| _embedding_ / _cosine similarity_ (SBERT)             | pengukuran kemiripan makna antar-istilah secara otomatis                                                                                                                                         |
| _grounding_                                           | keterpautan rekomendasi pada bukti — jaminan kompetensi berakar pada kalimat lowongan nyata, bukan karangan model                                                                                |
| halusinasi / _anti-hallucination filter_              | keluaran model yang tampak meyakinkan tetapi tak berdasar; penyaring yang menolak kompetensi tanpa dukungan bukti                                                                                |
| _evidence repair_                                     | perbaikan otomatis yang menambatkan kembali kompetensi pada bukti terkuatnya, atau membuangnya bila bukti tak cukup                                                                              |
| _recommendation contract_                             | aturan baku penetapan derajat kelayakan tiap rekomendasi (eksploratif / kandidat tinjauan / siap-resmi)                                                                                          |
| _class-aware ranking_                                 | pemeringkatan yang menyadari golongan kompetensi (teknis inti, proses rekayasa, penunjang keterserapan)                                                                                          |
| _lineage_ / _denominator_                             | penelusuran asal-usul data; angka penyebut, yaitu jumlah lowongan acuan untuk menghitung proporsi permintaan                                                                                     |
| _precision_ / _recall_ / _F1-score_                   | ketepatan / kelengkapan / rerata harmonik keduanya pada hasil ekstraksi                                                                                                                          |

**Contoh (sebelum → sesudah):**

- Sebelum: "Setelah ekstraksi, setiap span diklasifikasikan ke dalam kategori hard_technical, technical_process, support_soft_skill, personality_trait, credential, education_requirement, out_of_domain, atau invalid_fragment."
- Sesudah: "Setelah ekstraksi, setiap potongan teks yang ditandai (_span_) akan digolongkan ulang menurut aturan baku — serupa memilah kartu tugas hasil lokakarya DACUM ke dalam kategori yang telah disepakati. Penggolongan ini memisahkan keterampilan teknis inti dari keterampilan lunak, sifat pribadi, syarat ijazah, istilah di luar bidang, dan penggalan tak bermakna, sehingga frasa seperti 'komunikasi yang baik' tidak mencemari daftar kompetensi teknis."

### H.2 Suara manusia akademik, bukan mesin

Tulis seperti akademisi senior yang percaya diri pada argumennya. Hindari pola yang membuat teks terbaca seperti keltuaran AI.

**HINDARI:**

- **Pembuka paragraf monoton** ("Selain itu," / "Dengan demikian," / "Oleh karena itu," / "Namun," berulang). Variasikan; sering kali paragraf terkuat dibuka langsung dengan klaim substantifnya.
- **Penomoran "Pertama… Kedua… Ketiga…" di hampir setiap paragraf.** Gunakan seperlunya; selebihnya prosa mengalir.
- **Obsesi trikolon** ("X, Y, dan Z" tanpa henti). Pecah ritmenya.
- **Penguat kosong** ("sangat signifikan", "sangat penting", "krusial", "secara fundamental"). Ganti dengan substansi.
- **"tidak hanya… tetapi juga…"** dan **"bukan sekadar… melainkan…"** yang berulang.
- **Kalimat berpanjang seragam.** Selang-seling panjang–pendek.
- **Meta-komentar/basa-basi** ("Penting untuk dicatat bahwa…"). Buang.
- **Sitasi ditempel** ("Penulis (tahun) menyatakan X. Penulis2 (tahun) menyatakan Y."). Integrasikan ke alur argumen.

**LAKUKAN:**

- Argumen **berurutan dan beralasan**, bukan daftar fakta.
- Variasikan struktur dan panjang kalimat; ritme prosa alami.
- Kata sambung bervariasi dan tepat (namun, akan tetapi, sebaliknya, sementara itu, pada gilirannya).
- Suara penulis konsisten: kandidat doktor pendidikan kejuruan yang fasih pedagogi sekaligus paham NLP.
- Saat menyajikan rancangan teknis, jelaskan **mengapa**, bukan hanya **apa**.
- Pertahankan kedalaman; jangan menyederhanakan sampai dangkal.

---

## I. FORMAT KELUARAN YANG DIHARAPKAN

- Kerjakan **per BAB** (Bab I → II → III). Untuk tiap bab, kembalikan **teks revisi bersih** dalam bahasa Indonesia akademik baku, siap disalin ke Word.
- Untuk **gambar**, sertakan spesifikasi renderable (Mermaid/Graphviz atau cetak-biru bertahap) + caption + narasi (§E).
- Pertahankan struktur Bab/sub-bab dan sitasi APA. Tambahkan bagian baru (1.3, Ruang Lingkup, Definisi Operasional) pada posisi yang benar.
- Sisipkan `[KEPUTUSAN PENULIS: …]` di setiap titik keputusan; di Bab II tandai sumber yang belum terverifikasi dengan `[PERLU VERIFIKASI]`.
- Di akhir tiap bab, lampirkan **catatan ringkas penyunting** (3–6 butir) tentang perubahan besar + alasannya, dan (untuk Bab II) **daftar rujukan baru** untuk diverifikasi penulis.
- Jangan menambahkan hasil/angka/klaim empiris.

---

## J. CATATAN — INSTRUMEN HARUS DIREVISI ULANG

Reframe ini mengubah produk (pipeline) dan kerangka evaluasi (evaluasi pipeline ekstraksi, bukan evaluasi "sistem"). **Setelah proposal selesai, instrumen penelitian harus disusun ulang** agar selaras: gunakan prompt terpisah `PROMPT_PENYUSUNAN_INSTRUMEN_PENELITIAN.md` dengan melampirkan proposal hasil revisi ini. Jangan menyusun instrumen di prompt ini.

---

## K. DAFTAR SWA-PERIKSA (jalankan sebelum mengirim tiap bab)

- [ ] Judul baru terpasang konsisten; fokus = **pipeline analisis**, dashboard = pelaksana/penyaji, rekomendasi = luaran.
- [ ] Seluruh kalimat tentang pipeline/sistem dalam **kala rencana**; tidak ada klaim sistem sudah jadi.
- [ ] Tidak ada angka hasil/metrik capaian/klaim empiris. Tidak ada sitasi karangan (Bab II memakai research mode; sumber nyata + daftar rujukan baru untuk diverifikasi).
- [ ] Bab I: Rumusan Masalah + Pertanyaan Penelitian (1.3), Ruang Lingkup, Definisi Operasional, **dan SOTA ringkas**.
- [ ] Bab II: sub-bab evaluasi telah digeser menjadi **kajian SOTA pipeline ekstraksi & evaluasinya**; teori direposisi (CBET grand theory; DACUM kerangka hilir).
- [ ] Bab III: **gambar arsitektur pipeline** (spesifikasi renderable + narasi lugas) ada; loop uji→revisi pipeline dinyatakan; gambar lain di tiap bab diusulkan dengan spesifikasi.
- [ ] Penomoran Bab/sub-bab berurutan (tidak ada lompatan 1.3, 2.14.3–2.14.5).
- [ ] Keputusan penulis (DACUM digital, Bloom↔KKNI, tahun sampul, korpus) ditandai `[KEPUTUSAN PENULIS]`.
- [ ] **Setiap istilah teknis dijelaskan lugas pada kemunculan pertama; pakar kurikulum non-IT memahami tiap paragraf tanpa glosarium.**
- [ ] **Tidak ada kode/nama berkas/status sistem telanjang dalam prosa.**
- [ ] Tiap rujukan tersitir; tiap sitasi ada di daftar; tabel/gambar dirujuk sebelum muncul.
- [ ] Prosa terbaca seperti tulisan akademisi manusia: argumen berurutan, ritme bervariasi, suara konsisten.

---

_Akhir prompt. Mulai dengan Bab I (dengan research mode aktif untuk SOTA ringkas), lalu tunggu instruksi penulis untuk Bab II (research mode penuh) dan Bab III (visualisasi pipeline)._
