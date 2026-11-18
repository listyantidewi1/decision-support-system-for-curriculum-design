# Prompt untuk Menyusun Instrumen Penelitian Disertasi (Lampiran) — Edisi Proposal 26 Juni 2026

Gunakan file ini sebagai **prompt utama di claude.ai (chat)**. Tujuannya: menghasilkan **paket lampiran instrumen penelitian** yang lengkap dan siap tempel untuk proposal disertasi.

**Lampirkan proposal berikut saat memakai prompt ini:**

- `Proposal_Disertasi_26062026.docx`  ← versi terbaru, **wajib** dipakai.

Catatan penggunaan:

1. Jika model tidak dapat membaca lampiran, minta model **berhenti** dan meminta proposal diunggah ulang. Jangan biarkan model menyusun instrumen tanpa membaca proposal.
2. Karena keluaran sangat panjang, minta model bekerja **bertahap per kelompok lampiran** (mis. Matriks Induk + Definisi Operasional dulu; lalu lembar validasi ahli; lalu gold-labeling lima-tag; lalu DACUM/Delphi; lalu rekap & berita acara), bukan sekaligus.
3. Prompt ini menggantikan `PROMPT_PENYUSUNAN_INSTRUMEN_PENELITIAN.md` yang lama. Perubahan terpenting: **skema anotasi kini LIMA-TAG** (skill-hard, skill-soft, knowledge, tool-technology, work-behavior), bukan empat-tag. Semua instrumen anotasi/ekstraksi mengikuti lima-tag ini.

---

## Prompt Utama (salin mulai dari sini ke claude.ai)

Anda adalah tim ahli multidisipliner yang terdiri atas:

1. ahli metodologi penelitian dan pengembangan (R&D) pendidikan vokasi;
2. ahli pengembangan kurikulum berbasis kompetensi (Competency-Based Curriculum/CBET);
3. ahli DACUM / pengembangan kurikulum vokasi berbasis okupasi;
4. ahli evaluasi pendidikan dan psikometri;
5. ahli NLP, LLM, data mining, dan sistem rekomendasi;
6. ahli bidang Rekayasa Perangkat Lunak/Teknologi Informasi;
7. reviewer disertasi yang teliti terhadap validitas instrumen, validitas klaim, dan keterlacakan bukti.

Saya melampirkan proposal disertasi berjudul:

**"Analisis Kebutuhan Pasar Tenaga Kerja Bidang Rekayasa Perangkat Lunak Menggunakan Natural Language Processing (NLP) dan Large Language Model (LLM)"** (Listyanti Dewi Astuti, Pascasarjana Pendidikan Kejuruan, Universitas Negeri Malang, 2026).

Tugas Anda: **baca proposal secara menyeluruh, lalu susun paket instrumen penelitian yang lengkap, siap pakai, dan siap diisi**, sebagai **lampiran proposal/disertasi**. Instrumen harus konsisten dengan proposal versi ini, yang berfokus pada pengembangan **pipeline (alur kerja) analisis kebutuhan pasar tenaga kerja berbasis NLP dan LLM** dengan pendekatan **Research and Development model Borg & Gall**, kerangka teori **CBET (grand theory)** dan **DACUM (landasan kerja sekaligus kerangka pengembangan kurikulum di hilir)**. Sistem rekomendasi kompetensi adalah **luaran pipeline**, dan **dashboard** adalah **pelaksana pipeline sekaligus penyaji hasil**.

Keluaran harus berbentuk **naskah lampiran**, formal dan siap ditempel ke bagian lampiran proposal — bukan sekadar daftar instrumen, konsep, atau rencana implementasi. Walaupun pelaksanaan validasi nanti dapat memakai dashboard/spreadsheet, **setiap instrumen harus dapat dipahami, diisi, dihitung, dan diinterpretasikan tanpa bergantung pada dashboard apa pun**.

Setiap instrumen wajib memuat: (1) identitas instrumen; (2) tujuan penggunaan; (3) responden/validator; (4) waktu penggunaan dalam tahapan penelitian (Borg & Gall); (5) petunjuk pengisian; (6) skala penilaian + rubrik tiap skala; (7) kisi-kisi; (8) lembar kosong siap isi; (9) kolom skor; (10) kolom catatan/rekomendasi; (11) cara menghitung skor; (12) rumus analisis; (13) contoh format rekapitulasi; (14) cara interpretasi; (15) kriteria keputusan/tindak lanjut.

Jangan menulis "instrumen ini dapat dikembangkan kemudian". Semua instrumen yang dibutuhkan proposal disusun **sekarang**, dalam bentuk siap lampir.

---

## Prinsip Wajib Tambahan (setara penting dengan kelengkapan instrumen)

### 1. Aksesibilitas lintas-bidang

Instrumen akan (a) diisi sebagian oleh **validator ahli kurikulum/pendidikan tanpa latar belakang IT**, dan (b) dilampirkan pada proposal yang diuji penguji lintas-bidang. Karena itu:

- Setiap butir, petunjuk, dan definisi harus dapat dipahami pembaca non-IT. Tulis **berlapis**: satu kalimat yang dipahami ahli kurikulum, diikuti istilah presisi (cetak miring/dalam kurung) agar validator teknis tetap paham maksudnya.
- Kode/label sistem (mis. `KEEP_REQUIREMENT`, `valid_cluster`, `tool-technology`, `work-behavior`, `siap ditelaah formal`) boleh dipakai pada **lembar anotasi teknis untuk annotator terlatih**, tetapi pada **lembar validasi ahli kurikulum dan seluruh prosa petunjuk**, setiap kode harus disertai padanan Indonesia yang jelas. Sediakan **glosarium ringkas** di awal tiap lembar yang memuat kode.
- Gunakan analogi ke dunia yang dikenal ahli kurikulum (panel/lokakarya DACUM, kartu tugas, berkas perkara/berkas bukti) bila membantu.
- Jangan menyederhanakan sampai dangkal — layani validator teknis dan validator kurikuler sekaligus.

### 2. Suara manusia akademik, bukan mesin

Petunjuk pengisian, pertanyaan wawancara/FGD/Delphi, lembar informasi penelitian, dan narasi instrumen harus terbaca seperti tulisan akademisi manusia: kalimat bervariasi; **tanpa** pola "Pertama… Kedua… Ketiga…" yang kaku di tiap paragraf; tanpa penguat kosong ("sangat penting", "krusial"); tanpa pembuka monoton ("Selain itu," / "Dengan demikian," berulang). Butir penilaian boleh ringkas dan terstruktur, tetapi narasi di sekitarnya harus alami.

### 3. Kala rencana (sistem belum dibangun)

Proposal ini **rencana penelitian**; belum ada sistem maupun hasil. Instrumen dirancang terhadap **luaran sistem yang direncanakan**. Tandai setiap contoh data/kompetensi/kalimat/cluster sebagai `[contoh ilustratif]`. **Jangan** menulis angka hasil, metrik capaian, atau klaim keberhasilan apa pun.

### 4. Penyelarasan dengan kisi-kisi yang sudah ada di Bab III proposal

Instrumen wajib mengoperasionalkan kisi-kisi yang sudah tercantum di proposal. Rujuk **berdasarkan judul tabel dan subbab**, karena pada naskah terdapat penomoran tabel yang berulang (ada "Tabel 3.1"/"Tabel 3.2" di Subbab 3.3.2 *dan* di Subbab 3.5). Gunakan acuan berikut:

- **Kisi-Kisi Instrumen Wawancara Ahli Kurikulum dan Praktisi Industri (Subbab 3.5)** — aspek: relevansi okupasi; kesesuaian DACUM; kesesuaian CBET; pemetaan KKNI/Spektrum/SKKNI; kegunaan berkas bukti; risiko rekomendasi.
- **Kisi-Kisi Kuesioner dan Lembar Validasi Produk (Subbab 3.5)** — aspek: keabsahan korpus; kualitas ekstraksi; koherensi pengelompokan; kualitas pernyataan kompetensi; kontrak rekomendasi; analisis cakupan; kegunaan papan informasi.
- **Pemetaan Metrik Evaluasi terhadap Tahapan Borg & Gall (Subbab 3.5)** — gunakan sebagai sumber kolom "Tahap Borg & Gall" di Matriks Induk.

Pastikan **setiap indikator pada ketiga tabel memiliki butir instrumen yang bersesuaian**, dan kaitkan tiap instrumen ke **tujuan penelitian (Subbab 1.2)** dan **pertanyaan penelitian (PP1–PP4)**. Bila PP tidak tertulis eksplisit, **turunkan dari empat tujuan penelitian** dan nyatakan asumsi penurunannya. Bila kisi-kisi pada naskah berbeda dari ringkasan di atas, **ikuti naskah proposal**. Tambahkan satu butir di "Daftar Konfirmasi" yang menanyakan penomoran tabel ganda agar diperbaiki peneliti.

### 5. Selaras dengan bingkai proposal (PIPELINE-FIRST + LIMA-TAG) — WAJIB

- **Produk yang divalidasi adalah *pipeline* analisis**, bukan "sebuah sistem". **Dashboard/papan informasi** hanyalah pelaksana pipeline dan penyaji hasil. Ganti penyebutan "sistem" menjadi **"pipeline/alur kerja"** kecuali ketika memang merujuk dashboard sebagai pelaksana/penyaji.
- **CBET = grand theory**; **DACUM diulas dwiperan** — (a) **landasan kerja** yang menjelaskan mengapa rekomendasi pipeline layak menjadi dasar kurikulum, dan (b) **kerangka pengembangan kurikulum di hilir** bagi institusi. **Tanpa istilah "DACUM digital".**
- **Tanpa Taksonomi Bloom sebagai jembatan penetapan KKNI.** Jenjang KKNI ditetapkan **langsung** melalui pencocokan makna (*semantic matching* dengan *sentence embedding* seperti SBERT) antara pernyataan kompetensi dan deskriptor resmi KKNI — mempertimbangkan kompleksitas, otonomi, dan tanggung jawab — lalu **divalidasi ahli**. Bloom hanya boleh muncul sebagai catatan pedagogik opsional, **bukan** dasar jenjang. Instrumen validasi KKNI **tidak** menilai "Bloom level".
- **Future-aware di level keterampilan.** Pembobotan tren (*future weight*) berbasis **Laporan Future of Jobs WEF (2025)** dihitung pada **keterampilan**, lalu **diagregasi ke kompetensi** (mis. nilai maksimum atau rerata berbobot prevalensi). **Aturan kejujuran:** bobot bersumber dari pencocokan ke Laporan WEF sebagai bukti, **bukan tebakan model**; bobot yang hanya berasal dari saran model ditandai dan **tidak dipakai** dalam pemeringkatan publik. Analisis tren temporal dari korpus (mis. koreksi Benjamini-Hochberg; uji Durbin-Watson) **hanya** dijalankan bila data lowongan lintas-waktu memadai; jika tidak, diposisikan **eksploratif**.
- **Anotasi gold LIMA-TAG.** Penelitian melatih & menguji model JobBERT/encoder dengan **data acuan beranotasi sendiri** memakai **lima tag dalam notasi BIO/BIOES**:
  1. **skill-hard** — keterampilan teknis. `[contoh ilustratif: "mengembangkan REST API", "menulis kueri SQL"]`
  2. **skill-soft** — keterampilan lunak yang relevan kerja. `[contoh ilustratif: "berkomunikasi dengan pemangku kepentingan teknis"]`
  3. **knowledge** — pengetahuan/konsep. `[contoh ilustratif: "prinsip basis data relasional"]`
  4. **tool-technology** — alat/teknologi/bahasa/framework sebagai entitas tersendiri. `[contoh ilustratif: "Python", "Docker", "React", "PostgreSQL", "AWS"]`
  5. **work-behavior** — sikap/perilaku kerja. `[contoh ilustratif: "ketelitian terhadap detail keamanan"]`
  Anotasi dilakukan pada **dua korpus (Indonesia & Global/multibahasa)** dengan **eksperimen** menentukan korpus pelatihan terbaik (Indonesia / Global / gabungan-transfer). Instrumen anotasi & evaluasi ekstraksi mengikuti skema lima-tag ini (lihat Bagian H).
  **Penting:** `tool-technology` ditandai terpisah agar istilah teknologi **tidak** langsung diperlakukan sebagai kompetensi; ia adalah bukti yang masih harus dikaitkan dengan tindakan kerja, konteks, dan kompleksitas.

---

## Konteks Penelitian yang Harus Dipertahankan

Penelitian mengembangkan **pipeline analisis kebutuhan pasar tenaga kerja** bidang Rekayasa Perangkat Lunak/Pengembangan Gim/TI; **rekomendasi kompetensi adalah luaran pipeline**, disajikan melalui dashboard. Prinsip **evidence-first**: setiap rekomendasi kompetensi harus dapat ditelusuri mundur sepanjang rantai berikut (sesuai Subbab 3.3.1):

`korpus lowongan mentah → audit penelusuran/lineage → penyiapan & audit relevansi kalimat → ekstraksi lima-tag (skill-hard/skill-soft/knowledge/tool-technology/work-behavior) → penggolongan deterministik pascaekstraksi → konversi tabel bukti → pengelompokan bukti + audit kesahihan cluster → audit tema permintaan → perumusan competency statement terikat bukti → penyaringan antihalusinasi → perbaikan bukti → penggabungan duplikasi → pemetaan KKNI/jenjang pendidikan/okupasi/SKKNI/Spektrum → kontrak rekomendasi → dashboard + evidence package + paket validasi`

**Grand theory & posisi teori:**

1. **CBET** — output harus berupa **competency statement** yang *observable, assessable, performance-based*, diawali kata kerja; bukan sekadar daftar topik/tools.
2. **DACUM (dwiperan, tanpa "DACUM digital")** — (a) landasan kerja: kebutuhan kerja nyata mendefinisikan kompetensi → alasan rekomendasi pipeline layak jadi dasar kurikulum; (b) kerangka hilir institusi: `okupasi/spektrum → duty area → task evidence → tools/technology → hard skill/knowledge/soft skill → competency statement → curriculum recommendation → assessment idea`.
3. **Human Capital Theory & Signaling Theory** — hanya teori pendukung, bukan dasar utama penyusunan kurikulum.

**Prinsip ilmiah yang wajib dijaga:**

1. Korpus US/non-Indonesia hanya untuk **uji rekayasa (engineering validation)**, bukan klaim kebutuhan kurikulum Indonesia.
2. Klaim kurikulum Indonesia hanya boleh dari **korpus Indonesia yang valid, lengkap lineage-nya, dan lolos validasi ahli**.
3. **LLM bukan otoritas kurikulum**; ia hanya membantu ekstraksi, perumusan, ringkasan, dan audit yang tetap divalidasi manusia.
4. Rekomendasi memiliki **status kelayakan** (kontrak rekomendasi, Subbab 3.3.5), tiga kelas:
   - **Eksploratif** — sinyal awal; bukti terbatas/lineage belum lengkap/grounding belum memadai; hanya untuk pemantauan & telaah awal.
   - **Kandidat tinjauan** — korpus valid, evidence package lengkap, cluster koheren, grounding memadai, tanpa klaim tanpa bukti; bahan diskusi ahli & analisis kurikulum.
   - **Siap ditelaah formal** — lolos audit korpus/ekstraksi/grounding/pemetaan, validasi ahli/Delphi, dan uji stabilitas; bahan telaah formal, **bukan keputusan otomatis**.
5. Status **"siap ditelaah formal" tidak akan pernah** diberikan untuk korpus non-Indonesia pada konteks kurikulum Indonesia.
6. Bila bukti lemah/campuran/tidak cukup, pipeline menyatakan kompetensi **tidak direkomendasikan dari korpus tersebut**, bukan memaksakan rekomendasi. Status dapat **naik/turun** seiring bukti baru atau temuan audit.
7. Instrumen harus mengukur **validitas komputasional dan validitas kurikuler sekaligus**, dipisah tetapi saling terhubung.

**Protokol benchmark model (Subbab 3.3.2 — bukan model tunggal).** Instrumen evaluasi model harus mendukung perbandingan beberapa keluarga pendekatan: (a) rule-based/dictionary-based; (b) encoder sequence labeling pada data acuan lima-tag (BIO/BIOES); (c) domain-adapted encoder (JobBERT); (d) open-type information extraction; (e) prompted LLM tanpa retrieval; (f) **retrieval-verified schema-constrained LLM** (kandidat pipeline utama). Evaluasi mencakup strict/partial span-F1, macro-F1 per tag, confusion matrix, schema conformance, groundedness, evidence precision, unsupported-claim/output rate, dan abstention rate.

**Eksperimen korpus pelatihan (Subbab 3.3.2).** Tiga konfigurasi: **Korpus Indonesia** (dasar utama klaim), **Korpus Global/multibahasa** (uji transfer & pembanding, **tidak** menjadi dasar rekomendasi formal Indonesia), dan **Gabungan/transfer** (dipakai hanya bila meningkatkan performa tanpa melemahkan keabsahan konteks lokal). Pemilihan konfigurasi terbaik **tidak hanya** dari F1, tetapi juga grounding, kepatuhan skema, kelengkapan dokumen, stabilitas, biaya/waktu, dan penerimaan ahli.

**Populasi, sampel, dan subjek (Subbab 3.4).**

- Korpus Indonesia direncanakan dari Jobstreet Indonesia, Glints, Kalibrr, Karier.mu, LinkedIn; rentang ±12–24 bulan terakhir; **target 10.000–20.000** lowongan berbahasa Indonesia yang lolos kriteria kelengkapan. Untuk tren temporal, perluas hingga ≥36 bulan bila metadata tanggal memadai.
- Subjek manusia: ahli kurikulum vokasi, praktisi industri perangkat lunak, guru SMK RPL/Pengembangan Gim, dosen vokasi TI, dan pengguna sistem informasi.
- **Reliabilitas antar-penilai:** sekurang-kurangnya **tiga reviewer ahli** untuk penilaian kualitas ekstraksi dan kompetensi.
- **Panel Delphi (uji coba utama): 7–9 orang**, menyeimbangkan tiga sudut pandang — ahli kurikulum vokasi, praktisi industri perangkat lunak, dan asesor kompetensi.
- **Empat unit analisis** (perjelas denominator, jangan dicampur): unit **dokumen** (satu lowongan), unit **kalimat**, unit **span** (entitas lima-tag), unit **rekomendasi** (pernyataan kompetensi).

---

## Tujuan Output: konstruk yang harus diukur/divalidasi

1. validitas kebutuhan dan **desain pipeline** (sebagai produk);
2. validitas korpus, data lineage, dan denominator (korpus Indonesia & Global);
3. kualitas penyaringan relevansi kalimat (*sentence relevance filtering*);
4. **kualitas anotasi gold lima-tag (skill-hard, skill-soft, knowledge, tool-technology, work-behavior)** dan kualitas ekstraksi;
5. kualitas penggolongan deterministik (typing) hasil ekstraksi;
6. **ketepatan protokol benchmark model** (6 pendekatan) dan **eksperimen korpus pelatihan** (Indonesia/Global/gabungan-transfer): precision/recall/F1 per tag, strict/partial span-F1, confusion matrix;
7. koherensi cluster dan keputusan split/merge/drop;
8. validitas audit tema permintaan (*demand theme audit*);
9. kualitas competency statement berdasarkan CBET;
10. kesesuaian evidence bundle dengan competency statement (grounding);
11. **ketepatan penetapan jenjang KKNI** (metode langsung *semantic matching*, tanpa Bloom) + pemetaan SKKNI, Spektrum Keahlian, dan/atau skema okupasi (pisahkan pemetaan resmi vs komunitas);
12. **ketepatan & kejujuran pembobotan future weight** berbasis Laporan Future of Jobs WEF (dihitung di level keterampilan, diagregasi ke kompetensi);
13. kesesuaian DACUM (landasan kerja + kerangka hilir): okupasi → duty → task → skills/knowledge → competency → curriculum recommendation;
14. kualitas ranking sadar-golongan, deduplication, dan recommendation contract;
15. usefulness, trust, transparency, dan usability dashboard sebagai pelaksana pipeline + penyaji hasil;
16. validitas coverage analyzer kurikulum;
17. reliabilitas antar-penilai;
18. **stabilitas pipeline** lintas seed/model/penyedia/korpus (ablation + Jaccard top-k + perubahan Top-K + NDCG@K bila acuan tersedia);
19. cost/runtime feasibility dan reproduksibilitas (config/versi/parameter/biaya/waktu);
20. kelayakan pipeline (beserta luaran & dashboard) dalam uji coba terbatas.

---

## Keluaran Wajib (paket lampiran bernomor)

Sajikan sebagai **paket lampiran proposal** dengan penomoran jelas (Lampiran 1, 2, 3, …). Untuk **setiap** lampiran instrumen, berikan tiga bagian: (1) **Panduan Pengisian**; (2) **Formulir Siap Isi** (tabel kosong); (3) **Panduan Pengolahan & Interpretasi** (rumus, contoh rekap, kategori interpretasi, kriteria keputusan).

### A. Matriks Induk Instrumen Penelitian
Tabel yang memetakan: tujuan penelitian → pertanyaan penelitian (PP1–PP4) → variabel/konstruk → indikator → sumber data → responden/penilai → **tahap Borg & Gall** (ambil dari Pemetaan Metrik §3.5) → instrumen → bentuk data → teknik analisis → kriteria keputusan. Format tabel siap pakai.

### B. Definisi Operasional Konstruk
Definisi operasional untuk semua konstruk penting; selaraskan dengan **Tabel 1.1 Definisi Operasional** pada proposal dan tambahkan yang belum ada. Minimal: validitas korpus; data lineage; denominator; sentence relevance; extraction quality; **skill-hard; skill-soft; knowledge; tool-technology; work-behavior**; skema anotasi **lima-tag BIO/BIOES** dan reliabilitas antar-annotator; korpus pelatihan & eksperimen korpus (Indonesia/Global/gabungan-transfer); post-extraction typing; cluster coherence; mixed cluster; demand theme/demand signal; job prevalence; evidence grounding; future weight (berbasis WEF, di level keterampilan); hallucination/fabrication risk; competency statement quality; CBET alignment; DACUM alignment; KKNI alignment; SKKNI/Spektrum alignment; recommendation status (eksploratif/kandidat tinjauan/siap ditelaah formal); dashboard usefulness; dashboard transparency; curriculum coverage; trust in recommendation; official readiness/"siap ditelaah formal".
Tiap definisi memuat: definisi konseptual; definisi operasional; indikator; contoh bukti/data `[contoh ilustratif]`; cara pengukuran; kriteria interpretasi.

### C. Instrumen Validasi Ahli Kurikulum CBET/DACUM
Lembar validasi siap isi: identitas validator; petunjuk; **skala 1–4 tanpa pilihan netral**; rubrik skala; butir; kolom skor; kolom komentar; kolom rekomendasi revisi. Aspek: (1) kesesuaian competency statement dengan CBET; (2) keterukuran & keteramatan; (3) ketepatan action verb, objek kerja, konteks, standar performa; (4) kesesuaian DACUM duty-task-skill-knowledge; (5) kelayakan sebagai bahan pengembangan kurikulum; (6) pemisahan kompetensi inti teknis vs proses/leadership vs employability/support vs out-of-domain; (7) risiko overclaim dari evidence pasar kerja; (8) kejelasan status rekomendasi. Sediakan keputusan akhir: layak tanpa revisi / revisi kecil / revisi besar / tidak layak.

### D. Instrumen Validasi Ahli RPL/TI dan Praktisi Industri
Untuk praktisi industri/software engineer senior/tech lead/HR teknis/asesor kompetensi. Aspek: (1) relevansi kompetensi dengan kerja RPL/TI; (2) ketepatan task evidence dari lowongan; (3) kecukupan keterampilan teknis; (4) kecukupan knowledge; (5) kecukupan soft skill yang relevan; (6) ketepatan pemetaan okupasi/fungsi kerja; (7) ketepatan prioritas demand; (8) kompetensi terlalu luas/sempit/redundant/missing; (9) kelayakan untuk jenjang SMK/Diploma/Sarjana vokasi; (10) status yang sesuai (kandidat tinjauan / tidak direkomendasikan dari korpus / dapat dipertimbangkan untuk "siap ditelaah formal" setelah syarat formal terpenuhi).

### E. Instrumen Validasi Ahli NLP/AI/Data Mining
Aspek: (1) validitas pipeline evidence-first; (2) kecukupan data lineage; (3) sentence relevance filtering; (4) extraction span quality; (5) **typing lima-tag** (skill-hard/skill-soft/knowledge/tool-technology/work-behavior); (6) cluster validity; (7) demand theme audit; (8) antihalusinasi/grounding; (9) deduplication; (10) ranking demand-first sadar-golongan; (11) desain ablation/stability; (12) keterbatasan LLM & risiko bias; (13) ketepatan **skema anotasi lima-tag** dan rancangan **eksperimen korpus pelatihan** + **protokol benchmark 6 pendekatan**; (14) ketepatan & **kejujuran future weight** berbasis WEF (di level keterampilan, diagregasi ke kompetensi). Instrumen menilai apakah **pipeline** layak dipercaya secara teknis, bukan sekadar tampilan dashboard.

### F. Instrumen Validasi Ahli Pengukuran/Evaluasi (Psikometri)
Aspek: (1) kesesuaian instrumen dengan tujuan penelitian; (2) kejelasan konstruk; (3) validitas isi; (4) kejelasan butir; (5) kejelasan skala; (6) kelayakan teknik analisis; (7) reliabilitas antar-penilai; (8) kecukupan sampel/validator; (9) kriteria keputusan pass/fail; (10) risiko bias penilaian.

### G. Instrumen Gold Labeling: Sentence Relevance
Panduan + lembar anotasi: definisi label; contoh positif/negatif; petunjuk kasus ambigu; **aturan konservatif** (kalimat requirement yang tidak pasti lebih baik diberi keep/uncertain daripada dibuang diam-diam). Label minimal: `KEEP_REQUIREMENT`, `KEEP_TASK`, `KEEP_TOOL_OR_TECH`, `KEEP_CREDENTIAL_OR_EDUCATION`, `KEEP_SOFT_SKILL`, `DROP_COMPANY_DESCRIPTION`, `DROP_BENEFIT`, `DROP_LOGISTICS`, `DROP_DUPLICATE`, `UNCERTAIN_KEEP`. Kolom: `sentence_id`, `job_id`, `sentence_text`, `label_gold`, `confidence`, `rationale`, `annotator_id`, `notes`.

### H. Instrumen Gold Labeling: Anotasi Span LIMA-TAG (untuk melatih & menguji JobBERT)
Pedoman anotasi + lembar siap isi untuk membangun data gold yang **melatih & menguji** model JobBERT/encoder. Skema **memperluas SkillSpan** (skill/knowledge) menjadi **lima tag** dalam notasi **BIO** (juga jelaskan opsi BIOES bila dipakai):

- **skill-hard**, **skill-soft**, **knowledge**, **tool-technology**, **work-behavior** (lihat definisi & `[contoh ilustratif]` di Prinsip §5).

Pedoman wajib memuat: definisi tiap tag + **batas antar-tag** (kapan sebuah frasa masuk skill-hard vs knowledge vs tool-technology vs skill-soft vs work-behavior; khususnya **aturan memisahkan tool-technology dari knowledge dan dari skill-hard**); **aturan batas span** + **format BIO** (`B-skill-hard`, `I-skill-hard`, `B-tool-technology`, …, `O`); aturan untuk tool/framework/bahasa pemrograman, frasa kerja, soft skill, sikap kerja, dan domain pengetahuan; aturan kasus ambigu + **prosedur adjudication** antar-annotator; contoh benar & salah (`[contoh ilustratif]`); **lembar anotasi siap isi** (kolom: `sentence_id`, `job_id`, `token`, `tag_BIO`, `annotator_id`, `catatan`).

Dua hal khusus yang harus ditegaskan:
1. **Dua korpus terpisah** — pedoman/lembar untuk **korpus Indonesia** dan **korpus Global/multibahasa**, masing-masing dibagi **train/dev/test secara terstratifikasi** (berdasarkan sumber platform, bahasa, jenis pekerjaan, kategori tag). Pedoman boleh sama; contoh disesuaikan bahasa korpus.
2. **Reliabilitas antar-annotator (IAA)** — prosedur menghitung kesepakatan pada span/tag: **Cohen's Kappa** (2 annotator), **Fleiss' Kappa** (>2 annotator), atau **Krippendorff's Alpha** (bila struktur data/missing value memerlukan) dan/atau F1 antar-annotator; sertakan **ambang** dan **aturan adjudikasi** + pembekuan versi data acuan.

Tambahkan **lembar evaluasi model ekstraksi** untuk **protokol benchmark (6 pendekatan)** dan **eksperimen korpus pelatihan (Indonesia/Global/gabungan-transfer)**: bandingkan konfigurasi terhadap data uji memakai **strict/partial span-F1, precision/recall/F1 per tag, dan confusion matrix** (lembar rekap siap isi; **tanpa angka hasil — hanya format**), plus kolom metrik LLM (JSON validity, schema conformance, missing-field rate, unsupported output rate, abstention rate, groundedness/evidence precision).

> Catatan keterkaitan: skema lima-tag ini adalah versi **gold/terlatih** dari penggolongan (typing) pipeline. Bila pipeline di hilir memakai kategori lebih rinci (mis. memisahkan *credential/education requirement*, *out-of-domain*, *invalid fragment*, atau memecah *work-behavior* menjadi sifat pribadi vs sikap kerja), jelaskan **pemetaan dari lima-tag gold ke kategori hilir** dalam pedoman agar tidak rancu.

### I. Instrumen Validasi Cluster
Lembar validasi cluster dengan status: `valid_cluster`, `split_required`, `mixed_domain`, `support_process_cluster`, `out_of_domain_cluster`, `weak_low_frequency_theme`, `drop_invalid`. Aspek: (1) koherensi semantik; (2) kesamaan fungsi kerja; (3) kesamaan domain teknis; (4) kecukupan source jobs; (5) ada/tidaknya campuran soft skill/credential/out-of-domain; (6) kebutuhan split/merge; (7) kelayakan masuk competency generation. Sertakan rubrik skor, tabel siap isi, aturan keputusan.

### J. Instrumen Validasi Competency Statement dan Evidence Package
Instrumen inti, satu kompetensi per baris/form, elemen: `competency_id`, `competency_statement`, `competency_class`, `recommendation_status`, `job_prevalence`, `source_job_count`, evidence sentences, related skills, related knowledge, related soft skills, **related tools/technology**, KKNI label, occupation/SKKNI/Spektrum mapping, future_weight (+ sumber: WEF/saran model), blockers, alasan "tidak direkomendasikan" bila ada. Aspek: (1) relevansi RPL/TI; (2) grounding terhadap evidence; (3) tidak hallucinated; (4) tidak terlalu luas; (5) tidak terlalu sempit; (6) tidak redundant; (7) assessable; (8) observable; (9) sesuai CBET; (10) sesuai DACUM; (11) tepat jenjang KKNI; (12) tepat pemetaan okupasi/SKKNI/Spektrum; (13) layak sebagai bahan telaah kurikulum; (14) status rekomendasi sudah jujur; (15) perlu direvisi/digabung/dipisah/dipindah kelas/dihapus. Format keputusan: `accept_as_review_candidate`, `revise_statement`, `merge_with_other`, `split_required`, `demote_to_support`, `mark_out_of_domain`, `not_recommended_from_this_corpus`, `candidate_for_official_ready_after_validation`.

### K. Instrumen Validasi KKNI, SKKNI, Spektrum, dan Okupasi
Menilai: **ketepatan jenjang KKNI** (kompleksitas, otonomi, tanggung jawab) — **tanpa menilai "Bloom level"**; **ketepatan metode penetapan KKNI** itu sendiri (apakah pencocokan makna pernyataan kompetensi ke deskriptor resmi KKNI masuk akal menurut ahli); kesesuaian dengan deskriptor KKNI resmi; okupasi/fungsi kerja; SKKNI/unit kompetensi bila tersedia; Spektrum Keahlian (konteks SMK); status **pemetaan resmi vs pemetaan komunitas**. Sertakan peringatan: pemetaan komunitas **tidak** boleh dianggap resmi sebelum divalidasi dan dirujuk ke dokumen resmi versi tertentu.

### L. Worksheet DACUM/CBET
Worksheet siap isi untuk expert review berbasis DACUM:
`okupasi/skema → duty area → task evidence → tools/technology → hard skill → knowledge → soft skill → work behavior → competency statement → curriculum implication → assessment idea`.
Dapat dipakai dalam workshop ahli atau Delphi.

### M. Instrumen Validasi Dashboard dan Pengalaman Pengguna
Untuk pengguna dashboard publik & administratif: (1) skenario tugas; (2) lembar observasi tugas; (3) task success checklist; (4) **System Usability Scale (SUS) versi Bahasa Indonesia**; (5) perceived usefulness; (6) perceived transparency; (7) trust in recommendation; (8) intention to use; (9) open-ended feedback. Tegaskan: instrumen dashboard **tidak** dipakai mengklaim validitas rekomendasi kurikulum — usability hanya mengukur kegunaan antarmuka.

### N. Instrumen Validasi Coverage Analyzer Kurikulum
Menilai ketepatan pembandingan dokumen kurikulum vs rekomendasi kompetensi: gold curriculum coverage; covered/partially-covered/uncovered competency; false positive; false negative; precision; recall; F1; komentar ahli.

### O. Panduan Wawancara dan FGD
Untuk: ahli kurikulum; praktisi industri; guru SMK RPL/Pengembangan Gim; dosen vokasi/TI; pengguna dashboard; panel Delphi. Gali: kebutuhan pipeline; kepercayaan terhadap rekomendasi; kecukupan evidence package; risiko salah interpretasi; relevansi kurikulum; kesiapan implementasi; revisi yang diperlukan. (Gunakan aspek **Kisi-Kisi Wawancara §3.5**: relevansi okupasi, kesesuaian DACUM, kesesuaian CBET, pemetaan KKNI/Spektrum/SKKNI, kegunaan berkas bukti, risiko rekomendasi.)

### P. Instrumen Delphi / Expert Panel
Panel **7–9 orang** (ahli kurikulum vokasi, praktisi industri, asesor kompetensi). Paket beberapa putaran: putaran 1 (penilaian awal + komentar bebas); putaran 2 (revisi + rating konsensus); putaran 3 (konfirmasi final, bila perlu). Sediakan: format rating; format komentar; **kriteria konsensus** (mis. persentase kesepakatan, IQR, Kendall's W) ditetapkan **sebelum** analisis; aturan keputusan retain/revise/drop; cara melaporkan perubahan antar-putaran. Rekomendasi yang tak mencapai konsensus **diturunkan** menjadi kandidat tinjauan/eksploratif, bukan dipaksakan.

### Q. Paket Analisis Validitas dan Reliabilitas
Rumus + tabel perhitungan + interpretasi untuk: **Aiken's V**; **I-CVI & S-CVI/Ave**; **Lawshe CVR** (bila relevan); **Cronbach's alpha / McDonald's omega** (skala persepsi); **Cohen's Kappa**; **Fleiss' Kappa**; **Krippendorff's Alpha**; percent agreement; **precision/recall/F1**; **strict & partial span-F1**; confusion matrix; **Wilson confidence interval**; **Jaccard top-k stability**; **perubahan Top-K**; **NDCG@K** (bila acuan peringkat tersedia); **Spearman rank correlation** (stabilitas ranking); **Kendall's W** (konsensus Delphi); pass/fail report. Untuk tiap teknik: (1) kapan dipakai; (2) input data; (3) rumus; (4) struktur tabel rekap; (5) contoh membaca hasil; (6) kategori interpretasi; (7) keputusan tindak lanjut. Sertakan **ambang interpretasi awal yang wajar**, tetapi tandai "dapat disesuaikan dengan keputusan promotor/ahli metodologi" bila proposal belum menetapkannya. **Jangan hanya menyebut nama teknik** — beri cara hitung & cara baca.

### R. Template Rekap Data (Sheet Excel)
Struktur sheet minimal — untuk tiap sheet beri daftar kolom, tipe data, contoh nilai `[contoh ilustratif]`, dan catatan validasi:
1. `validator_identity`; 2. `instrument_blueprint`; 3. `expert_scores`; 4. `aiken_v`; 5. `cvi_cvr`; 6. `sentence_gold_labels`; 7. `span_gold_labels` (**lima-tag BIO**; kolom korpus: Indonesia/Global; split train/dev/test); 8. `cluster_review`; 9. `competency_review`; 10. `kkni_mapping_review`; 11. `dacum_worksheet`; 12. `dashboard_sus`; 13. `dashboard_task_observation`; 14. `coverage_gold`; 15. `reliability_iaa`; 16. `stability_ablation`; 17. `revision_log`; 18. `extraction_model_eval` (strict/partial span-F1 & precision/recall/F1 per tag untuk tiap pendekatan benchmark **dan** tiap konfigurasi korpus: Indonesia/Global/gabungan); 19. `future_weight_review` (validasi pembobotan tren berbasis WEF di level keterampilan); 20. `llm_schema_conformance` (JSON validity, schema conformance, missing-field, unsupported output, abstention).

### S. Template Berita Acara dan Lembar Persetujuan
Template: lembar informasi penelitian untuk validator/responden; informed consent; berita acara validasi ahli; berita acara FGD/Delphi; log revisi berdasarkan masukan ahli. Bahasa Indonesia formal, praktis, siap pakai. (Selaraskan dengan **Etika Penelitian §3.7**: kerahasiaan identitas, hak menarik diri, penggunaan data hanya untuk penelitian.)

---

## Format Penulisan

Bahasa Indonesia akademik; dokumen lengkap dengan heading jelas. Setiap sitasi/rujukan dalam instrumen (mis. sumber SUS, rumus statistik, deskriptor KKNI, Laporan Future of Jobs WEF) ditulis **gaya APA edisi ke-7** (sitasi dalam teks: Penulis, tahun; *dkk.* untuk ≥3 penulis; daftar rujukan alfabetis, indentasi gantung, DOI/URL bila ada). Selaraskan dengan Pedoman Pascasarjana UM bila menetapkan gaya lain.

Untuk tiap instrumen, gunakan urutan: (1) Nama; (2) Tujuan; (3) Responden/validator; (4) Waktu penggunaan (tahap Borg & Gall); (5) Konstruk yang diukur; (6) Petunjuk pengisian; (7) Skala & rubrik; (8) Kisi-kisi; (9) Lembar penilaian siap isi; (10) Pertanyaan terbuka/komentar; (11) Cara penskoran; (12) Kriteria interpretasi; (13) Keputusan tindak lanjut; (14) Format rekapitulasi; (15) Contoh interpretasi hasil.

Format tabel butir minimal:

| No | Aspek | Indikator | Butir Pernyataan/Penilaian | Skala | Skor | Catatan/Rekomendasi |
| -- | ----- | --------- | -------------------------- | ----- | ---- | ------------------- |

**Skala validasi ahli 1–4 (tanpa netral):** 1 = tidak valid/sesuai/jelas; 2 = kurang; 3 = valid/sesuai/jelas; 4 = sangat valid/sesuai/jelas. **Usability/acceptance** boleh skala 1–5 (mis. SUS) dengan alasan dinyatakan.

Setelah tiap tabel instrumen, tambahkan: **Cara Pengolahan Skor** (skor per aspek, total, rerata, persentase kelayakan/indeks validitas); **Cara Interpretasi** (kategori + ambang eksplisit); **Keputusan Revisi** (diterima tanpa revisi / revisi minor / revisi mayor / validasi ulang / tidak digunakan); **Format Rekapitulasi** (tabel siap pindah ke Excel).

---

## Batasan yang Harus Ditaati

1. Jangan mengklaim instrumen sudah valid sebelum diuji.
2. Jangan menganggap output LLM kebenaran tanpa validasi ahli.
3. Jangan menyatukan validasi teknis pipeline dengan validasi kurikulum; pisah tetapi saling terhubung.
4. Jangan membuat status "siap ditelaah formal" untuk data US/non-Indonesia.
5. Jangan menghapus kemungkinan hasil negatif; instrumen harus memungkinkan ahli menyatakan suatu kompetensi **tidak direkomendasikan dari korpus tertentu**.
6. Jangan membuat butir terlalu umum ("sistem sudah baik"); tiap butir mengukur aspek spesifik.
7. Jika proposal mengandung bagian `[PERLU KONFIRMASI]`, jadikan daftar kebutuhan konfirmasi dan hubungkan ke instrumen/keputusan metodologis.
8. Jangan menyerahkan detail pengisian/penghitungan/interpretasi kepada dashboard; instrumen proposal harus berdiri sendiri sebagai dokumen akademik.
9. Jangan menunda penyusunan instrumen ke tahap pelaksanaan; semua instrumen utama (termasuk petunjuk, scoring, rumus, interpretasi) ditulis **sekarang** sebagai lampiran.
10. Tiap butir & petunjuk dapat dipahami validator non-IT; kode sistem selalu disertai padanan Indonesia + glosarium ringkas; jangan tampilkan jargon teknis telanjang pada lembar ahli kurikulum (lihat Prinsip §1).
11. Tulis petunjuk & narasi dengan suara akademik manusia, bukan pola mesin; jangan mencantumkan angka hasil/klaim capaian; contoh ditandai `[contoh ilustratif]` (Prinsip §2–§3).
12. **Gunakan skema LIMA-TAG** (skill-hard, skill-soft, knowledge, tool-technology, work-behavior) secara konsisten di semua instrumen anotasi/ekstraksi/typing/rekap. Jangan memakai skema empat-tag lama (yang memakai "attitude" tanpa "tool-technology").

---

## Output Akhir yang Saya Inginkan (urutan)

1. daftar lampiran instrumen yang akan masuk ke proposal;
2. ringkasan fungsi setiap instrumen;
3. matriks penelitian–instrumen–metrik (Matriks Induk);
4. definisi operasional konstruk;
5. semua instrumen siap isi;
6. semua panduan anotasi/gold labeling (sentence relevance + span lima-tag, dua korpus, train/dev/test);
7. semua worksheet DACUM/CBET;
8. semua panduan wawancara/FGD/Delphi;
9. rumus, cara perhitungan, dan cara interpretasi (Paket Validitas & Reliabilitas);
10. template rekap skor & sheet Excel;
11. template berita acara & informed consent;
12. checklist kesiapan validasi;
13. daftar hal yang masih perlu dikonfirmasi kepada peneliti/promotor (termasuk: penomoran tabel ganda di Bab III; ambang konsensus Delphi; rumus agregasi future weight; ambang Kappa/Aiken's V; komposisi & jumlah final validator).

Jika jawaban terlalu panjang untuk satu respons, bagi menjadi beberapa bagian dan lanjutkan sistematis. **Jangan meringkas** bagian instrumen. Prioritaskan kelengkapan dan kesiapan pakai.
