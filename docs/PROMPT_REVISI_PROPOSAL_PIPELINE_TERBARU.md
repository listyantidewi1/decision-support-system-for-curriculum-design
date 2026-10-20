# Prompt untuk ChatGPT: Revisi Proposal Disertasi Sesuai Pipeline Terbaru

Gunakan prompt ini bersama file proposal lama:

`Proposal_Disertasi_Revisi.docx`

Tujuan prompt ini adalah meminta ChatGPT memperbaiki proposal seperlunya, bukan menulis ulang dari nol.

---

## Prompt Utama

Anda adalah asisten akademik tingkat doktoral yang berpengalaman dalam penelitian pendidikan vokasi, pengembangan kurikulum berbasis kompetensi, DACUM, NLP/LLM untuk analisis pasar tenaga kerja, dan metodologi R&D Borg & Gall.

Saya melampirkan proposal disertasi lama berjudul:

**"Pengembangan Sistem Rekomendasi Kompetensi Bidang Rekayasa Perangkat Lunak Berdasarkan Analisis Kebutuhan Pasar Tenaga Kerja"**

Tugas Anda adalah **merevisi proposal tersebut seperlunya** agar selaras dengan pipeline penelitian terbaru yang sudah dikembangkan. Jangan menulis proposal baru dari nol jika struktur lama masih bisa dipertahankan. Pertahankan gaya akademik Bahasa Indonesia, penomoran bab/subbab, dan format proposal sejauh memungkinkan.

Output yang saya butuhkan:

1. Dokumen proposal revisi lengkap, sebaiknya dalam format `.docx`.
2. Daftar ringkas bagian yang Anda ubah.
3. Catatan bagian yang masih memerlukan data, rujukan, atau keputusan peneliti, ditandai dengan `[PERLU KONFIRMASI]` atau `[PERLU RUJUKAN]`.

Jangan mengarang data, hasil eksperimen, angka validasi, nama dataset, atau rujukan. Jika Anda tidak yakin, beri penanda `[PERLU KONFIRMASI]`. Jika Anda menambahkan rujukan baru, pastikan rujukan tersebut nyata dan relevan; jika tidak dapat memverifikasi, gunakan `[PERLU RUJUKAN]`.

---

## Prinsip Revisi

Proposal lama sudah memiliki struktur Bab I, Bab II, dan Bab III. Revisi harus bersifat **terarah**:

- Pertahankan fokus penelitian pada pengembangan sistem rekomendasi kompetensi untuk pendidikan vokasi bidang Rekayasa Perangkat Lunak/IT.
- Jadikan **Competency-Based Curriculum / Competency-Based Education and Training** sebagai teori utama kurikulum.
- Jadikan **DACUM (Developing a Curriculum)** sebagai kerangka utama pengembangan kurikulum.
- Tempatkan Human Capital Theory dan Signaling Theory sebagai teori pendukung, bukan grand theory utama.
- Tegaskan bahwa sistem ini bukan menggantikan ahli kurikulum atau panel DACUM, melainkan menyediakan **evidence package** awal yang dapat divalidasi oleh ahli.
- Hindari klaim bahwa rekomendasi langsung "resmi" atau "siap diterapkan" sebelum korpus Indonesia, pemetaan resmi, validasi ahli, dan uji stabilitas selesai.

---

## Konteks Pipeline Terbaru yang Harus Dicerminkan

Pipeline terbaru adalah pipeline **evidence-first** untuk menghasilkan rekomendasi kompetensi yang defensibel secara ilmiah. Rekomendasi hanya dapat dipercaya jika seluruh rantai bukti berikut dapat ditelusuri:

`raw job CSV -> prepared sentences -> Skill-LLM extraction JSONL -> Phase-1 evidence CSV -> clustering -> competency generation -> evidence repair -> recommendation contract -> dashboard/evidence packages`

### 1. Audit Data Lineage dan Validitas Korpus

Proposal harus menjelaskan bahwa sistem kini memiliki audit lineage dan validitas korpus sebelum rekomendasi digunakan.

Audit menjawab:

- CSV mentah mana yang menghasilkan dataset prepared.
- Kalimat prepared mana yang dikirim ke Skill-LLM.
- Output Skill-LLM JSONL mana yang dikonversi menjadi Phase-1 CSV.
- Apakah denominator `jobs_metadata.csv` konsisten dengan `source_job_ids` kompetensi.
- Apakah korpus bersifat Indonesia, US/non-Indonesia, campuran, atau unknown.
- Apakah korpus valid untuk klaim yang dibuat.

Status validitas korpus:

- `valid_for_engineering_test`
- `valid_for_indonesian_claim`
- `invalid_or_incomplete`
- `needs_manual_review`

Catatan penting untuk proposal:

- Korpus US hanya boleh diposisikan sebagai **uji teknis/engineering validation**, bukan dasar klaim kebutuhan kurikulum Indonesia.
- Klaim kurikulum Indonesia hanya dapat dibuat dari korpus Indonesia yang lolos audit lineage, validitas bahasa/negara, cakupan metadata, dan review ahli.
- Korpus Indonesia saat ini harus diperlakukan sebagai konteks target penelitian, tetapi status kesiapan final harus dijelaskan hati-hati jika Phase-1 evidence atau validasi belum lengkap.

### 2. Integritas Denominator dan Cakupan Tahap

Pipeline terbaru melaporkan denominator secara terpisah:

- raw jobs,
- prepared jobs,
- sentence-filtered jobs,
- Skill-LLM input jobs,
- Skill-LLM output jobs,
- Phase-1 jobs,
- final source-job jobs.

Gunakan `prepared valid-description jobs` atau denominator yang diaudit sebagai denominator pasar tenaga kerja. Jika terjadi kehilangan cakupan ekstraksi atau sentence filtering, hal itu harus dilaporkan sebagai keterbatasan atau blocker, bukan disembunyikan.

### 3. Perbaikan Kualitas Ekstraksi

Jangan lagi menulis bahwa semua span `SKILL` dari Skill-LLM otomatis menjadi hard skill.

Pipeline terbaru melakukan deterministic post-extraction typing:

- `hard_technical`
- `technical_process`
- `support_soft_skill`
- `personality_trait`
- `credential`
- `education_requirement`
- `out_of_domain`
- `invalid_fragment`

Proposal harus menjelaskan bahwa soft skill, credential, personality traits, dan fragmen generik dipisahkan sebelum clustering agar tidak mencemari daftar kompetensi teknis inti.

Contoh prinsip:

- "communication skills", "work ethic", "attention to detail" tidak boleh menjadi core technical competency.
- "Python", "SQL", "REST API", "Terraform", "data pipelines", "automated testing" dapat menjadi kandidat teknis jika benar-benar didukung oleh bukti lowongan kerja.
- Jika Python tidak muncul atau hanya muncul sangat lemah dalam korpus tertentu, maka Python tidak boleh dipaksakan untuk bertahan sebagai rekomendasi.

### 4. Audit Sentence Relevance dan Domain Filtering

Proposal perlu menjelaskan bahwa filtering kalimat tidak boleh terlalu agresif. Kalimat yang mungkin mengandung persyaratan pekerjaan harus dipertahankan bila ragu.

Pipeline juga membedakan:

- `domain_excluded`: bukti non-IT yang dikeluarkan dari core recommendation,
- `out_of_domain_visible_signal`: bukti non-IT yang tetap dilaporkan sebagai sinyal visible tetapi tidak masuk rekomendasi inti.

### 5. Audit Validitas Cluster

Sebelum kompetensi dibuat, cluster diaudit dengan status:

- `valid_cluster`
- `split_required`
- `mixed_domain`
- `support_process_cluster`
- `out_of_domain_cluster`
- `weak_low_frequency_theme`

Proposal harus menekankan bahwa cluster campuran atau junk-drawer tidak boleh langsung menghasilkan rekomendasi inti. Jika cluster terlalu campur, sistem melaporkannya sebagai `split_required` atau `mixed_domain`.

Contoh cluster yang harus diwaspadai:

- testing bercampur Tableau/scripting/automation,
- Power Apps/Salesforce/IoT/user research/AWS,
- architecture/CAD/ASIC/design review,
- frontend/mobile/accessibility/rendering.

### 6. Demand Theme Audit dan Prinsip "Not Recommended From This Corpus"

Pipeline terbaru memiliki audit tema permintaan untuk tema sentinel seperti:

- Python,
- REST/API,
- Kubernetes,
- data pipelines,
- automated testing,
- debugging/troubleshooting,
- cloud/IaC,
- NLP/GenAI,
- architecture/design review.

Tema sentinel bukan daftar kompetensi yang harus muncul. Tema tersebut adalah probe audit. Status tema:

- `strong_direct_evidence`
- `moderate_direct_evidence`
- `weak_or_scattered_evidence`
- `tool_mention_only`
- `not_detected`
- `mixed_or_ambiguous`
- `out_of_domain`

Jika bukti tidak cukup, keluaran yang benar adalah:

`not_recommended_from_this_corpus`

Proposal harus menyatakan bahwa "tidak direkomendasikan dari korpus ini" adalah hasil ilmiah yang valid.

### 7. Competency Generation dan Anti-Hallucination

Proposal harus disesuaikan dengan format output terbaru:

- Tidak lagi menggunakan `title + description` sebagai bentuk utama.
- Output utama adalah **satu competency statement** yang verb-led, assessable, dan dapat ditelusuri ke bukti.
- LLM tidak boleh membuat demand truth. LLM hanya membantu merumuskan statement dari evidence bundle yang eksplisit.
- Sistem menjalankan anti-hallucination filter, evidence repair, grounding, dan deduplication.
- Kompetensi terlalu luas, menggabungkan domain tidak terkait, atau hanya berbasis tool mention harus ditolak atau diberi status `not_recommended_from_this_corpus`.

### 8. KKNI, Pendidikan, SKKNI, Spektrum, dan Okupasi

Proposal harus menjelaskan bahwa:

- Pelabelan KKNI dilakukan sebagai penilaian kompleksitas/otonomi tugas, bukan sekadar hasil LLM.
- Distribusi pendidikan dilaporkan berdasarkan cakupan metadata pada source jobs.
- Pemetaan okupasi menggunakan statement kompetensi terbaru, bukan title/description lama.
- Pemetaan resmi SKKNI harus dipisahkan dari community occupation mapping.
- Community mapping tidak boleh dianggap resmi sebelum validasi ahli.
- Official references seperti KKNI, SKKNI, Spektrum Keahlian, dan referensi komunitas harus diberi versi.

### 9. Recommendation Contract

Proposal harus menjelaskan status rekomendasi:

- `exploratory`: sinyal berguna, tetapi belum cukup untuk rekomendasi kurikulum formal.
- `review_candidate`: traceable, relevan, dan dapat masuk paket review ahli.
- `official_ready`: hanya boleh muncul jika semua syarat terpenuhi.

Aturan keras:

- US/non-Indonesia corpus tidak boleh menghasilkan `official_ready`.
- Unknown lineage tidak boleh melebihi `exploratory`.
- Indonesian corpus hanya dapat menjadi `official_ready` setelah validitas korpus, pemetaan resmi, expert review, dan stability validation lulus.

Ranking publik harus class-aware:

1. `core_technical`
2. `engineering_process`
3. `employability_support`
4. `out_of_domain_or_needs_review`

Untuk core technical, ranking harus demand-first:

`job_prevalence -> source_job_count -> grounding_score -> public_future_weight`

`future_weight` dari LLM harus disebut sebagai `llm_suggested_not_evidence` kecuali ada pemetaan deterministik atau bukti eksternal. Jangan menulis seolah-olah bobot masa depan LLM adalah bukti empiris.

### 10. DACUM/CBC Evidence Layer

Proposal harus menjelaskan kontribusi utama penelitian sebagai **digital DACUM berbasis bukti pasar tenaga kerja**, bukan sekadar sistem rekomendasi AI.

Gunakan rantai konseptual:

`occupation/skema -> duty area -> task evidence -> skills/knowledge -> competency statement -> curriculum recommendation`

Dalam narasi proposal:

- DACUM konvensional tetap menjadi fondasi filosofis.
- Pipeline digital memperluas DACUM dengan sinyal lowongan kerja skala besar.
- Expert review tetap diperlukan untuk menilai validitas, konteks lokal, dan penerjemahan ke kurikulum.
- Competency-Based Curriculum/CBET menjadi format hasil akhir: kompetensi harus observable, assessable, dan berbasis performa kerja.

### 11. Evidence Package, Validation Package, dan Dashboard

Proposal perlu memperbarui deskripsi produk:

- Dashboard tidak hanya menampilkan rekomendasi, tetapi juga status validitas korpus, blockers, audit cluster, demand theme audit, not-recommended themes, dan evidence package.
- Evidence package memuat provenance, source sentences, audit lineage, ranking class-aware, dan paket review ahli.
- DACUM/CBC package memuat worksheet untuk expert review.
- Validation package berisi template gold labeling dan metrik, tetapi hasil `not_measured` tidak boleh digunakan sebagai klaim validitas.

### 12. Validasi Ilmiah dan Kekhawatiran Akurasi

Karena pembimbing menekankan akurasi rekomendasi, proposal harus memperkuat bagian validasi.

Tambahkan atau revisi agar mencakup:

- gold-label sentence relevance,
- gold-label extraction span/type,
- cluster coherence review,
- competency validity review,
- top-rank usefulness review,
- inter-rater reliability,
- Delphi/expert panel,
- stability across seeds,
- ablation model/provider,
- ranking ablation,
- cost/runtime reporting,
- pass/fail report.

Jelaskan bahwa sistem tidak akan mengklaim akurasi sebelum metrik benar-benar diukur.

---

## Bagian Proposal yang Perlu Direvisi

### Bab I - Pendahuluan

Revisi seperlunya:

- Hilangkan placeholder seperti "Jelaskan GAP penelitian".
- Rumuskan gap penelitian secara eksplisit:
  - penyelarasan kurikulum vokasi masih banyak berbasis judgment manual,
  - DACUM tradisional kuat tetapi mahal dan terbatas panel,
  - sistem rekomendasi pendidikan umumnya berfokus pada materi/siswa, bukan competency recommendation berbasis demand kerja,
  - pendekatan LLM berisiko halusinasi jika tidak diberi lineage, grounding, dan expert review,
  - belum banyak sistem yang menggabungkan CBC/DACUM, job-posting evidence, KKNI/SKKNI/Spektrum, dan audit validitas korpus.
- Rumusan masalah dan tujuan perlu disejajarkan dengan:
  - pengembangan pipeline evidence-first,
  - audit validitas korpus,
  - kualitas rekomendasi kompetensi,
  - validasi ahli dan stabilitas.

### Bab II - Kajian Pustaka

Revisi seperlunya:

- Naikkan Competency-Based Curriculum/CBET sebagai grand theory.
- Naikkan DACUM sebagai framework utama curriculum development.
- Human Capital Theory dan Signaling Theory tetap boleh dipertahankan sebagai teori pendukung untuk menjelaskan hubungan pendidikan-pasar kerja.
- Revisi bagian NLP/LLM agar tidak overclaim:
  - LLM sebagai alat ekstraksi dan drafting berbasis evidence,
  - bukan pembuat kebenaran demand,
  - membutuhkan audit, grounding, dan review ahli.
- Revisi evaluasi sistem rekomendasi agar memuat lineage, grounding, cluster validity, theme audit, recommendation status, stability, dan expert validation.

### Bab III - Metode Penelitian

Revisi lebih substantif:

- Borg & Gall tetap dapat digunakan sebagai kerangka R&D makro.
- Agile/iterative development boleh disebut sebagai proses pengembangan software di dalam Borg & Gall.
- Ganti deskripsi pipeline lama yang terlalu spesifik dan sebagian tidak sesuai:
  - jangan menulis pipeline utama sebagai "LLM-only full-text + Hybrid JobBERT sebagai jalur utama" kecuali memang diposisikan sebagai ablation/eksperimen terpisah;
  - jangan menulis output sebagai Title/Description;
  - jangan menulis formula priority lama sebagai basis publik utama;
  - jangan menyatakan future weight berbasis LLM sebagai bukti empiris.
- Gunakan pipeline terbaru:
  1. raw corpus ingestion,
  2. corpus lineage audit,
  3. sentence preparation and relevance audit,
  4. Skill-LLM extraction,
  5. deterministic extraction typing,
  6. Phase-1 evidence conversion,
  7. clustering and cluster validity audit,
  8. demand theme audit,
  9. evidence-bound competency generation,
  10. anti-hallucination and evidence repair,
  11. deduplication,
  12. KKNI/education/occupation mapping,
  13. recommendation contract,
  14. dashboard and research packages,
  15. expert validation and stability/ablation.
- Populasi/sampel data harus hati-hati:
  - target final adalah korpus Indonesia bidang RPL/IT,
  - korpus US hanya engineering/test run,
  - jumlah data harus ditulis sebagai rencana atau data aktual hanya jika sudah dikonfirmasi.
- Teknik analisis data harus memuat metrik dan audit terbaru.

---

## Hal yang Harus Dihapus atau Dilemahkan

Hapus, revisi, atau beri caveat pada klaim berikut jika muncul:

- Sistem menghasilkan rekomendasi resmi tanpa validasi ahli.
- Semua output LLM dianggap benar.
- Semua `SKILL` dari extractor adalah hard skill.
- Future weight dari LLM dianggap evidence.
- US corpus digunakan untuk klaim kebutuhan kurikulum Indonesia.
- Unknown/mixed corpus digunakan untuk rekomendasi formal.
- Kompetensi yang tidak muncul di korpus tetap dipertahankan karena dianggap penting secara kurikuler.
- Validasi yang masih `not_measured` diperlakukan sebagai hasil.

---

## Gaya Penulisan yang Diinginkan

Gunakan Bahasa Indonesia akademik yang tegas, hati-hati, dan tidak berlebihan.

Hindari:

- klaim terlalu besar,
- istilah marketing,
- "AI akan menggantikan ahli",
- istilah teknis yang tidak dijelaskan,
- daftar pipeline terlalu panjang tanpa makna metodologis.

Utamakan:

- akuntabilitas bukti,
- keterlacakan data,
- validitas korpus,
- pemisahan sinyal eksploratif dan rekomendasi formal,
- CBC/DACUM sebagai fondasi kurikulum,
- expert review sebagai syarat legitimasi.

---

## Output yang Saya Harapkan dari Anda

Silakan revisi proposal lama dengan mode berikut:

1. Pertahankan struktur utama Bab I, Bab II, Bab III.
2. Revisi paragraf yang tidak lagi sesuai dengan pipeline terbaru.
3. Tambahkan subbagian jika diperlukan, khususnya untuk:
   - audit validitas korpus,
   - recommendation contract,
   - evidence package,
   - DACUM/CBC evidence layer,
   - validasi akurasi rekomendasi.
4. Jangan menghapus teori lama yang masih berguna; reposisikan jika perlu.
5. Tandai bagian yang membutuhkan data aktual dengan `[PERLU KONFIRMASI]`.
6. Tandai rujukan yang perlu diverifikasi dengan `[PERLU RUJUKAN]`.
7. Setelah revisi, berikan ringkasan perubahan per bab.

Jika Anda dapat menghasilkan `.docx`, buat file:

`Proposal_Disertasi_Revisi_Pipeline_Terbaru.docx`

Jika tidak dapat menghasilkan `.docx`, berikan teks revisi lengkap dengan struktur bab dan subbab yang dapat saya salin ke Word.

---

## Ringkasan Diagnosis Proposal Lama

Gunakan diagnosis ini sebagai pegangan awal:

- Bab I sudah relevan, tetapi gap penelitian perlu dirumuskan lebih kuat dan placeholder "Jelaskan GAP penelitian" harus dihapus.
- Bab II sudah memiliki bagian DACUM dan CBET, tetapi posisi grand theory perlu dibuat lebih eksplisit: CBC/CBET sebagai grand theory, DACUM sebagai framework curriculum development.
- Bab III adalah bagian yang paling perlu diperbarui karena pipeline lama masih memuat beberapa detail yang tidak selaras dengan pipeline terbaru, seperti output Title/Description, formula prioritas lama, future weight sebagai bukti, dan jalur JobBERT/hybrid yang harus diposisikan hati-hati.
- Proposal perlu lebih jujur tentang status korpus: US untuk engineering validation, Indonesia untuk klaim kurikulum setelah audit dan validasi.
- Proposal perlu memperkuat mekanisme akurasi rekomendasi: lineage audit, grounding, evidence repair, cluster validity, demand theme audit, expert review, dan stability/ablation.

