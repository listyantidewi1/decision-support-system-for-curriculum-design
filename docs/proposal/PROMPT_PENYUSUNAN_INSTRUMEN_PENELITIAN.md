# Prompt untuk Menyusun Instrumen Penelitian Disertasi

Gunakan file ini sebagai prompt utama di **claude.ai** (chat). Lampirkan proposal berikut ketika memakai prompt ini:

- `Proposal_Disertasi_23032026.docx`

Jika model tidak dapat membaca lampiran, minta model berhenti dan meminta proposal diunggah ulang. Jangan biarkan model menyusun instrumen tanpa membaca proposal. Karena keluaran sangat panjang, mintalah model bekerja **bertahap per kelompok lampiran** (mis. Matriks Induk + Definisi Operasional dulu, lalu lembar validasi ahli, lalu gold-labeling, dan seterusnya), bukan sekaligus.

---

## Prompt Utama

Anda adalah tim ahli multidisipliner yang terdiri atas:

1. ahli metodologi penelitian pendidikan vokasi;
2. ahli pengembangan kurikulum berbasis kompetensi atau Competency-Based Curriculum/CBET;
3. ahli DACUM atau pengembangan kurikulum vokasi berbasis okupasi;
4. ahli evaluasi pendidikan dan psikometri;
5. ahli NLP, LLM, data mining, dan sistem rekomendasi;
6. ahli bidang Rekayasa Perangkat Lunak/Teknologi Informasi;
7. reviewer disertasi yang teliti terhadap validitas instrumen, validitas klaim, dan keterlacakan bukti.

Saya melampirkan proposal disertasi berjudul:

**"Analisis Kebutuhan Pasar Tenaga Kerja Bidang Rekayasa Perangkat Lunak Menggunakan Natural Language Processing (NLP) dan Large Language Model (LLM)"**

Tugas Anda adalah membaca proposal tersebut secara menyeluruh, lalu menyusun **paket instrumen penelitian yang lengkap, siap pakai, dan siap diisi**. Instrumen harus sesuai dengan proposal hasil revisi terbaru — yang **berfokus pada pengembangan *pipeline* (alur kerja) analisis kebutuhan pasar tenaga kerja berbasis NLP dan LLM**, dengan sistem rekomendasi kompetensi sebagai **luaran pipeline** dan dashboard sebagai **pelaksana pipeline sekaligus penyaji hasil** — pendekatan Research and Development Borg and Gall, serta kerangka teori CBET (grand theory) dan DACUM (landasan kerja sekaligus kerangka pengembangan kurikulum di hilir).

Orientasi utama tugas ini adalah menyusun **lampiran instrumen untuk proposal penelitian/disertasi**. Jadi keluaran harus berbentuk naskah lampiran yang lengkap, formal, dan siap ditempel ke bagian lampiran proposal. Jangan hanya memberi daftar instrumen, konsep, atau rencana implementasi.

Walaupun pelaksanaan validasi nantinya dapat menggunakan dashboard, formulir digital, spreadsheet, atau sistem tertentu, instrumen dalam proposal harus tetap lengkap secara dokumen. Artinya setiap instrumen harus dapat dipahami, diisi, dihitung, dan diinterpretasikan tanpa bergantung pada dashboard tertentu.

Hasil harus berupa dokumen instrumen yang dapat langsung dipindahkan ke lampiran proposal/disertasi, Word, Google Docs, atau Excel. Setiap instrumen wajib memuat:

1. identitas instrumen;
2. tujuan penggunaan instrumen;
3. responden/validator yang mengisi;
4. waktu penggunaan dalam tahapan penelitian;
5. petunjuk pengisian untuk validator/responden;
6. skala penilaian dan rubrik setiap skala;
7. kisi-kisi instrumen;
8. lembar kosong siap isi;
9. kolom skor;
10. kolom catatan/rekomendasi;
11. cara menghitung skor;
12. rumus analisis yang digunakan;
13. contoh format rekapitulasi hasil;
14. cara interpretasi hasil;
15. kriteria keputusan/tindak lanjut.

Jangan menulis "instrumen ini dapat dikembangkan kemudian". Semua instrumen yang dibutuhkan proposal harus disusun sekarang dalam bentuk siap lampir.

---

## Prinsip Wajib Tambahan (Selaras dengan Standar Revisi Proposal)

Empat prinsip berikut **wajib** dan setara pentingnya dengan kelengkapan instrumen.

### 1. Aksesibilitas lintas-bidang

Instrumen ini akan (a) diisi oleh validator yang sebagian adalah **ahli kurikulum dan pendidikan tanpa latar belakang IT**, dan (b) dilampirkan pada proposal yang diuji oleh penguji lintas-bidang. Karena itu:

- Setiap butir, petunjuk, dan definisi harus dapat dipahami pembaca non-IT. Jelaskan istilah teknis dengan bahasa Indonesia yang lugas pada kemunculan pertamanya, baru sebut istilah aslinya (cetak miring atau dalam kurung). Tulis berlapis: satu kalimat yang dipahami ahli kurikulum, diikuti istilah presisi agar validator teknis tetap tahu maksudnya.
- Kode/label sistem (mis. `KEEP_REQUIREMENT`, `valid_cluster`, `hard_technical`, `official_ready`, `split_required`) boleh tetap dipakai **pada lembar anotasi teknis untuk annotator terlatih**, tetapi pada **lembar validasi ahli kurikulum dan pada seluruh prosa petunjuk**, setiap kode harus disertai padanan Indonesia yang jelas. Sediakan **glosarium ringkas** di awal setiap lembar yang memuat kode.
- Gunakan analogi ke dunia yang dikenal ahli kurikulum (panel/lokakarya DACUM, kartu tugas, berkas bukti) bila membantu pemahaman.
- Jangan menyederhanakan sampai dangkal — layani validator teknis dan validator kurikuler sekaligus, jangan mengorbankan salah satu.

### 2. Suara manusia akademik, bukan mesin

Petunjuk pengisian, pertanyaan wawancara/FGD/Delphi, lembar informasi penelitian, dan narasi instrumen harus terbaca seperti tulisan akademisi manusia: kalimat bervariasi, tanpa pola "Pertama… Kedua… Ketiga…" yang kaku di setiap paragraf, tanpa penguat kosong ("sangat penting", "krusial"), dan tanpa pembuka paragraf yang monoton ("Selain itu," / "Dengan demikian," berulang). Butir penilaian boleh ringkas dan terstruktur, tetapi narasi di sekitarnya harus alami.

### 3. Kala rencana (sistem belum dibangun)

Proposal ini adalah **rencana penelitian**; belum ada sistem maupun hasil. Instrumen dirancang terhadap **luaran sistem yang direncanakan**. Tandai setiap contoh data, kompetensi, kalimat, atau cluster sebagai `[contoh ilustratif]` agar tidak disalahartikan sebagai hasil nyata. Jangan menulis angka hasil, metrik capaian, atau klaim keberhasilan apa pun.

### 4. Penyelarasan dengan kisi-kisi proposal

Instrumen harus konsisten dengan kisi-kisi yang sudah ada di Bab III proposal:

- **Tabel 3.1 (Kisi-kisi Wawancara Ahli Kurikulum & Praktisi Industri):** relevansi okupasi; kesesuaian DACUM; kesesuaian CBET; pemetaan KKNI/Spektrum/SKKNI; kegunaan evidence package; risiko rekomendasi.
- **Tabel 3.2 (Kisi-kisi Kuesioner & Lembar Validasi Produk):** validitas korpus; kualitas ekstraksi; koherensi cluster; kualitas competency statement; recommendation contract; coverage analyzer; usability dashboard.

Pastikan **setiap indikator pada kedua tabel memiliki butir instrumen yang bersesuaian**, dan kaitkan tiap instrumen ke pertanyaan penelitian (PP1–PP5) serta tahap Borg & Gall di Matriks Induk (bagian A). Bila proposal yang dilampirkan belum memuat PP secara eksplisit, turunkan dari tujuan penelitian dan nyatakan asumsi penurunannya. **Bila kisi-kisi pada proposal hasil revisi berbeda dari Tabel 3.1/3.2 di atas, ikuti proposal hasil revisi.**

### 5. Selaras dengan reframe proposal (PIPELINE-FIRST) — WAJIB

Proposal telah direvisi dengan reframe yang disetujui promotor. Seluruh instrumen mengikuti bingkai ini:

- **Produk yang divalidasi adalah *pipeline* analisis**, bukan "sebuah sistem". **Dashboard** hanyalah alat untuk menjalankan pipeline dan menyajikan hasil. Sesuaikan setiap penyebutan "sistem" menjadi **"pipeline"** (kecuali ketika memang merujuk dashboard sebagai pelaksana/penyaji).
- **CBET = grand theory**; **DACUM diulas dwiperan** — (a) landasan kerja yang menjelaskan mengapa rekomendasi pipeline layak menjadi dasar kurikulum, dan (b) kerangka pengembangan kurikulum di hilir untuk institusi — **tanpa istilah "DACUM digital"**.
- **Tanpa Taksonomi Bloom.** Jenjang KKNI ditetapkan **langsung** melalui suatu metode (rencana: pencocokan makna pernyataan kompetensi ke deskriptor resmi KKNI, lalu divalidasi ahli), **tanpa** lebih dulu dipetakan ke Bloom. Instrumen validasi KKNI **tidak** menilai "Bloom level".
- **Future-aware di level keterampilan.** Pembobotan tren (*future weight*) berdasarkan Laporan *Future of Jobs* WEF dihitung pada **keterampilan**, lalu diagregasi ke kompetensi. Sediakan butir validasi atas ketepatan dan kejujuran pembobotan ini (bobot bersumber bukti WEF, bukan tebakan model).
- **Anotasi gold empat-tag.** Penelitian melatih & menguji model JobBERT dengan **data gold beranotasi sendiri** memakai **empat tag BIO: skill-hard, skill-soft, knowledge, attitude** (perluasan dari SkillSpan yang hanya skill/knowledge), pada **dua korpus (Indonesia & Global)**, dengan **eksperimen** menentukan korpus pelatihan terbaik. Instrumen anotasi & evaluasi ekstraksi mengikuti skema empat-tag ini (lihat Bagian H).

---

## Konteks Penelitian yang Harus Dipertahankan

Penelitian ini mengembangkan **pipeline analisis kebutuhan pasar tenaga kerja** bidang Rekayasa Perangkat Lunak/Pengembangan Gim/Teknologi Informasi; **sistem rekomendasi kompetensi adalah luaran pipeline**, disajikan melalui dashboard. Pipeline menggunakan pendekatan evidence-first, yaitu setiap rekomendasi kompetensi harus dapat ditelusuri dari:

`raw job CSV -> prepared sentences -> Skill-LLM extraction JSONL -> Phase-1 evidence CSV -> clustering -> cluster validity audit -> demand theme audit -> evidence-bound competency generation -> evidence repair -> deduplication -> KKNI/occupation/SKKNI/Spektrum mapping -> recommendation contract -> dashboard/evidence package/validation package`

Grand theory penelitian:

1. **Competency-Based Curriculum/CBET** sebagai teori utama format kompetensi: output harus berupa competency statement yang observable, assessable, performance-based, dan bukan sekadar daftar topik/tools.
2. **DACUM** berperan ganda (tanpa istilah "DACUM digital"): (a) **landasan kerja** — premis DACUM bahwa kebutuhan kerja nyata mendefinisikan kompetensi menjadi alasan mengapa rekomendasi pipeline layak jadi dasar kurikulum; dan (b) **kerangka pengembangan kurikulum di hilir** yang dipakai institusi: occupation/spektrum -> duty area -> task evidence -> skills/knowledge/soft skills -> competency statement -> curriculum recommendation.
3. Human Capital Theory dan Signaling Theory hanya teori pendukung, bukan dasar utama penyusunan kurikulum.

Prinsip ilmiah yang wajib dijaga:

1. Korpus US/non-Indonesia hanya boleh digunakan untuk engineering validation, bukan klaim kebutuhan kurikulum Indonesia.
2. Klaim kurikulum Indonesia hanya boleh menggunakan korpus Indonesia yang valid, lengkap lineage-nya, dan telah melalui validasi ahli.
3. LLM tidak boleh menjadi otoritas kurikulum. LLM hanya membantu ekstraksi, perumusan, ringkasan, dan audit yang tetap harus divalidasi manusia.
4. Rekomendasi harus memiliki status: `exploratory`, `review_candidate`, atau `official_ready`.
5. Status `official_ready` hanya boleh diberikan jika korpus Indonesia valid, pemetaan resmi terkonfirmasi, validasi ahli lulus, dan uji stabilitas/ablation memenuhi ambang.
6. Jika bukti lemah, campuran, atau tidak cukup, sistem harus menyatakan `not_recommended_from_this_corpus`, bukan memaksakan rekomendasi.
7. Instrumen harus mengukur validitas komputasional dan validitas kurikuler sekaligus.

---

## Tujuan Output

Susun paket instrumen untuk mengukur dan memvalidasi hal-hal berikut:

1. validitas kebutuhan dan **desain pipeline** (sebagai produk);
2. validitas korpus, data lineage, dan denominator (korpus Indonesia & Global);
3. kualitas sentence relevance filtering;
4. **kualitas anotasi gold empat-tag (skill-hard, skill-soft, knowledge, attitude)** dan ekstraksi keterampilan/pengetahuan;
5. kualitas penggolongan (typing) hasil ekstraksi;
6. **ketepatan eksperimen korpus pelatihan JobBERT** (Indonesia / Global / gabungan-transfer) dan evaluasi ekstraksi per tag (precision/recall/F1, confusion matrix);
7. koherensi cluster dan keputusan split/merge/drop;
8. validitas demand theme audit;
9. kualitas competency statement berdasarkan CBET;
10. kesesuaian evidence bundle dengan competency statement;
11. **ketepatan penetapan jenjang KKNI** (metode langsung tanpa pemetaan Bloom) serta pemetaan SKKNI, Spektrum Keahlian, dan/atau skema okupasi;
12. **ketepatan dan kejujuran pembobotan tren (*future weight*) berbasis Laporan *Future of Jobs* WEF** (dihitung di level keterampilan, diagregasi ke kompetensi);
13. kesesuaian DACUM (sebagai landasan kerja dan kerangka hilir): occupation -> duty -> task -> skills/knowledge -> competency -> curriculum recommendation;
14. kualitas ranking, deduplication, dan recommendation contract;
15. usefulness, trust, transparency, dan usability dashboard sebagai **pelaksana pipeline + penyaji hasil**;
16. validitas coverage analyzer kurikulum;
17. reliabilitas antar-penilai;
18. **stabilitas pipeline** lintas seed/model/provider/korpus;
19. cost/runtime feasibility;
20. kelayakan pipeline (beserta luaran dan dashboard) dalam uji coba terbatas.

---

## Keluaran Wajib

Hasil akhir harus berisi semua bagian berikut.

Sajikan seluruh keluaran sebagai **paket lampiran proposal** dengan penomoran yang jelas, misalnya:

- Lampiran 1. Matriks Induk Instrumen Penelitian
- Lampiran 2. Definisi Operasional Konstruk
- Lampiran 3. Lembar Validasi Ahli Kurikulum CBET/DACUM
- Lampiran 4. Lembar Validasi Ahli RPL/TI dan Praktisi Industri
- dan seterusnya.

Untuk setiap lampiran instrumen, berikan tiga bagian:

1. **Panduan Pengisian**: siapa yang mengisi, apa yang dinilai, bagaimana memberi skor, dan bagaimana memberi komentar.
2. **Formulir Siap Isi**: tabel kosong yang dapat langsung digunakan oleh validator/responden.
3. **Panduan Pengolahan dan Interpretasi**: rumus, contoh rekap, kategori interpretasi, dan kriteria keputusan.

### A. Matriks Induk Instrumen Penelitian

Buat tabel yang memetakan:

- tujuan penelitian;
- pertanyaan penelitian;
- variabel/konstruk yang diukur;
- indikator;
- sumber data;
- responden/penilai;
- tahap Borg and Gall;
- instrumen yang digunakan;
- bentuk data;
- teknik analisis;
- kriteria keputusan.

Gunakan format tabel siap pakai.

### B. Definisi Operasional Konstruk

Buat definisi operasional untuk semua konstruk penting, minimal:

- validitas korpus;
- data lineage;
- sentence relevance;
- extraction quality;
- hard skill (skill-hard);
- soft skill (skill-soft);
- knowledge;
- attitude (sikap kerja);
- skema anotasi empat-tag BIO dan reliabilitas antar-annotator;
- korpus pelatihan dan eksperimen korpus (Indonesia vs Global vs gabungan-transfer);
- post-extraction typing;
- cluster coherence;
- mixed cluster;
- demand signal;
- job prevalence;
- evidence grounding;
- future weight / bobot tren (berbasis Laporan Future of Jobs WEF, dihitung di level keterampilan);
- hallucination/fabrication risk;
- competency statement quality;
- CBET alignment;
- DACUM alignment;
- KKNI alignment;
- SKKNI/Spektrum alignment;
- recommendation status;
- dashboard usefulness;
- dashboard transparency;
- curriculum coverage;
- trust in recommendation;
- official readiness.

Setiap definisi harus memuat:

- definisi konseptual;
- definisi operasional;
- indikator;
- contoh bukti/data;
- cara pengukuran;
- kriteria interpretasi.

### C. Instrumen Validasi Ahli Kurikulum CBET/DACUM

Susun lembar validasi siap isi untuk ahli kurikulum vokasi. Minimal mencakup:

- identitas validator;
- petunjuk pengisian;
- skala penilaian 1-4 tanpa pilihan netral;
- rubrik skala;
- butir validasi;
- kolom skor;
- kolom komentar;
- kolom rekomendasi revisi.

Aspek yang harus dinilai:

1. kesesuaian competency statement dengan prinsip CBET;
2. keterukuran dan keteramatiannya;
3. kesesuaian action verb, objek kerja, konteks, dan standar performa;
4. kesesuaian DACUM duty-task-skill-knowledge;
5. kelayakan sebagai bahan pengembangan kurikulum;
6. pemisahan kompetensi inti teknis, process/leadership, employability/support, dan out-of-domain;
7. risiko overclaim dari evidence pasar kerja;
8. kejelasan status rekomendasi.

Sediakan juga bagian keputusan akhir:

- layak tanpa revisi;
- layak dengan revisi kecil;
- layak dengan revisi besar;
- tidak layak digunakan.

### D. Instrumen Validasi Ahli RPL/TI dan Praktisi Industri

Susun lembar validasi untuk praktisi industri, software engineer senior, tech lead, HR/recruiter teknis, atau asesor kompetensi bidang RPL/TI.

Aspek minimal:

1. relevansi kompetensi dengan pekerjaan RPL/TI;
2. ketepatan task evidence dari lowongan kerja;
3. kecukupan keterampilan teknis terkait;
4. kecukupan knowledge terkait;
5. kecukupan soft skill yang memang relevan terhadap kerja;
6. ketepatan pemetaan okupasi/fungsi kerja;
7. ketepatan prioritas demand;
8. apakah kompetensi terlalu luas, terlalu sempit, redundant, atau missing;
9. apakah kompetensi layak masuk kurikulum SMK/Diploma/Sarjana/Vokasi;
10. apakah rekomendasi perlu status `review_candidate`, `not_recommended_from_this_corpus`, atau dapat dipertimbangkan untuk `official_ready` setelah syarat formal terpenuhi.

### E. Instrumen Validasi Ahli NLP/AI/Data Mining

Susun lembar validasi untuk ahli NLP/AI/data mining. Aspek minimal:

1. validitas pipeline evidence-first;
2. kecukupan data lineage;
3. sentence relevance filtering;
4. extraction span quality;
5. hard/soft/knowledge typing;
6. cluster validity;
7. demand theme audit;
8. anti-hallucination/grounding;
9. deduplication;
10. ranking demand-first;
11. ablation/stability design;
12. keterbatasan LLM dan risiko bias;
13. ketepatan **skema anotasi empat-tag** (skill-hard/skill-soft/knowledge/attitude) dan rancangan **eksperimen korpus pelatihan** (Indonesia/Global/gabungan-transfer);
14. ketepatan dan **kejujuran pembobotan *future weight*** berbasis Laporan *Future of Jobs* WEF (dihitung di level keterampilan, diagregasi ke kompetensi).

Instrumen harus membantu ahli menilai apakah **pipeline** layak dipercaya secara teknis, bukan hanya apakah dashboard terlihat baik.

### F. Instrumen Validasi Ahli Pengukuran/Evaluasi

Susun lembar validasi untuk ahli evaluasi/psikometri/metodologi. Aspek minimal:

1. kesesuaian instrumen dengan tujuan penelitian;
2. kejelasan konstruk;
3. validitas isi;
4. kejelasan butir;
5. kejelasan skala;
6. kelayakan teknik analisis;
7. reliabilitas antar-penilai;
8. kecukupan sampel/validator;
9. kriteria keputusan pass/fail;
10. risiko bias penilaian.

### G. Instrumen Gold Labeling: Sentence Relevance

Buat panduan dan lembar anotasi untuk menilai apakah suatu kalimat lowongan kerja relevan sebagai evidence. Harus memuat:

- definisi label;
- contoh positif dan negatif;
- petunjuk ketika kalimat ambigu;
- aturan konservatif: kalimat requirement yang tidak pasti lebih baik diberi keep/uncertain daripada dibuang diam-diam;
- format tabel anotasi.

Label minimal:

- `KEEP_REQUIREMENT`;
- `KEEP_TASK`;
- `KEEP_TOOL_OR_TECH`;
- `KEEP_CREDENTIAL_OR_EDUCATION`;
- `KEEP_SOFT_SKILL`;
- `DROP_COMPANY_DESCRIPTION`;
- `DROP_BENEFIT`;
- `DROP_LOGISTICS`;
- `DROP_DUPLICATE`;
- `UNCERTAIN_KEEP`.

Sediakan format kolom:

- sentence_id;
- job_id;
- sentence_text;
- label_gold;
- confidence;
- rationale;
- annotator_id;
- notes.

### H. Instrumen Gold Labeling: Anotasi Span Empat-Tag (untuk Melatih & Menguji JobBERT)

Buat **pedoman anotasi (annotation guideline) dan lembar anotasi siap isi** untuk membangun data gold yang akan **melatih dan menguji** model JobBERT. Skema memperluas SkillSpan (yang hanya menandai skill dan knowledge) menjadi **empat tag** dalam **notasi BIO** (Begin–Inside–Outside):

- **skill-hard** — keterampilan teknis (mis. "mengembangkan REST API", "menulis kueri SQL"). `[contoh ilustratif]`
- **skill-soft** — keterampilan lunak yang relevan dengan kerja (mis. "komunikasi dengan pemangku kepentingan teknis"). `[contoh ilustratif]`
- **knowledge** — pengetahuan/konsep (mis. "prinsip basis data relasional"). `[contoh ilustratif]`
- **attitude** — sikap kerja (mis. "ketelitian terhadap detail keamanan"). `[contoh ilustratif]`

Pedoman wajib memuat:

- definisi tiap tag + batas (mana yang masuk skill-hard vs knowledge vs skill-soft vs attitude);
- **aturan batas span** dan **format BIO** (B-skill-hard, I-skill-hard, B-knowledge, …, O);
- aturan untuk tool/framework/bahasa pemrograman, frasa kerja, soft skill, dan domain pengetahuan;
- aturan kasus ambigu dan **penyelesaian konflik** antar-annotator;
- contoh benar dan salah (beberapa `[contoh ilustratif]`);
- **lembar anotasi siap isi** (kolom: `sentence_id`, `job_id`, `token`, `tag_BIO`, `annotator_id`, `catatan`).

Dua hal khusus yang harus ditegaskan:

1. **Dua korpus terpisah** — sediakan pedoman/lembar untuk **korpus Indonesia** dan **korpus Global**, masing-masing dibagi **train/dev/test**. Pedoman boleh sama, tetapi contoh disesuaikan dengan bahasa korpus.
2. **Reliabilitas antar-annotator (IAA)** — sediakan prosedur menghitung kesepakatan antar-annotator pada anotasi span/tag (mis. Cohen/Fleiss Kappa atau F1 antar-annotator), beserta ambang dan aturan adjudikasi.

Tambahkan **lembar evaluasi model ekstraksi** (untuk eksperimen korpus pelatihan): bandingkan konfigurasi pelatihan (Indonesia / Global / gabungan-transfer) terhadap data uji menggunakan **precision/recall/F1 per tag** dan **confusion matrix** (lembar rekap siap isi; tanpa angka hasil — hanya format).

> Catatan keterkaitan: skema empat-tag ini adalah versi **gold/terlatih** dari penggolongan (typing) di pipeline. Bila pipeline memakai kategori penggolongan yang lebih rinci di hilir (mis. memisahkan *credential*, *education requirement*, *out-of-domain*, *invalid fragment*), jelaskan pemetaan dari empat-tag gold ke kategori hilir tersebut dalam pedoman, agar tidak rancu.

### I. Instrumen Validasi Cluster

Buat lembar validasi cluster untuk ahli/annotator. Cluster harus dinilai dengan status:

- `valid_cluster`;
- `split_required`;
- `mixed_domain`;
- `support_process_cluster`;
- `out_of_domain_cluster`;
- `weak_low_frequency_theme`;
- `drop_invalid`.

Aspek penilaian:

1. koherensi semantik;
2. kesamaan fungsi kerja;
3. kesamaan domain teknis;
4. kecukupan source jobs;
5. ada/tidaknya campuran soft skill, credential, atau out-of-domain;
6. kebutuhan split atau merge;
7. kelayakan masuk competency generation.

Sediakan rubrik skor, tabel siap isi, dan aturan keputusan.

### J. Instrumen Validasi Competency Statement dan Evidence Package

Buat instrumen inti untuk menilai setiap rekomendasi kompetensi. Instrumen harus menampilkan satu kompetensi per baris atau per form, dengan elemen:

- competency_id;
- competency_statement;
- competency_class;
- recommendation_status;
- job_prevalence;
- source_job_count;
- evidence sentences;
- related skills;
- related knowledge;
- related soft skills;
- KKNI label;
- occupation/skema mapping;
- blockers;
- not_recommended reason jika ada.

Aspek yang dinilai:

1. relevansi dengan bidang RPL/TI;
2. grounding terhadap evidence;
3. tidak hallucinated;
4. tidak terlalu luas;
5. tidak terlalu sempit;
6. tidak redundant;
7. assessable;
8. observable;
9. sesuai CBET;
10. sesuai DACUM;
11. tepat untuk jenjang KKNI;
12. tepat untuk pemetaan okupasi/SKKNI/Spektrum;
13. layak sebagai bahan telaah kurikulum;
14. status rekomendasi sudah jujur;
15. perlu direvisi, digabung, dipisah, dipindah kelas, atau dihapus.

Sediakan format keputusan:

- accept_as_review_candidate;
- revise_statement;
- merge_with_other;
- split_required;
- demote_to_support;
- mark_out_of_domain;
- not_recommended_from_this_corpus;
- candidate_for_official_ready_after_validation.

### K. Instrumen Validasi KKNI, SKKNI, Spektrum, dan Okupasi

Buat instrumen untuk menilai ketepatan pemetaan:

- **ketepatan jenjang KKNI** kompetensi (kompleksitas, otonomi, tanggung jawab) — **tanpa menilai "Bloom level"** (penelitian tidak memakai pemetaan Bloom; jenjang KKNI ditetapkan langsung lewat metode pencocokan ke deskriptor KKNI, lalu divalidasi ahli);
- **ketepatan metode penetapan KKNI** itu sendiri: apakah pencocokan pernyataan kompetensi ke deskriptor resmi KKNI sudah masuk akal menurut ahli;
- kesesuaian dengan deskriptor KKNI resmi;
- okupasi/fungsi kerja;
- SKKNI/unit kompetensi jika tersedia;
- Spektrum Keahlian jika konteks SMK;
- status mapping resmi vs community mapping.

Harus ada peringatan bahwa community mapping tidak boleh dianggap resmi sebelum divalidasi dan dirujuk ke dokumen resmi versi tertentu.

### L. Worksheet DACUM/CBET

Buat worksheet siap isi untuk expert review berbasis DACUM:

`occupation/skema -> duty area -> task evidence -> tools/technology -> hard skill -> knowledge -> soft skill -> competency statement -> curriculum implication -> assessment idea`

Worksheet harus bisa digunakan dalam workshop ahli atau Delphi.

### M. Instrumen Validasi Dashboard dan Pengalaman Pengguna

Buat instrumen untuk pengguna dashboard publik dan administratif. Minimal:

1. skenario tugas;
2. lembar observasi tugas;
3. task success checklist;
4. System Usability Scale (SUS) versi Bahasa Indonesia;
5. perceived usefulness;
6. perceived transparency;
7. trust in recommendation;
8. intention to use;
9. open-ended feedback.

Pastikan instrumen dashboard tidak digunakan untuk mengklaim validitas rekomendasi kurikulum. Usability hanya mengukur kegunaan antarmuka.

### N. Instrumen Validasi Coverage Analyzer Kurikulum

Buat instrumen untuk menilai apakah sistem benar saat membandingkan dokumen kurikulum dengan rekomendasi kompetensi.

Harus mencakup:

- gold curriculum coverage;
- covered competency;
- partially covered competency;
- uncovered competency;
- false positive;
- false negative;
- precision;
- recall;
- F1;
- komentar ahli.

### O. Panduan Wawancara dan FGD

Buat panduan wawancara/FGD untuk:

1. ahli kurikulum;
2. praktisi industri;
3. guru SMK RPL/Pengembangan Gim;
4. dosen vokasi/TI;
5. pengguna dashboard;
6. panel Delphi jika digunakan.

Pertanyaan harus menggali:

- kebutuhan sistem;
- kepercayaan terhadap rekomendasi;
- kecukupan evidence package;
- risiko salah interpretasi;
- relevansi kurikulum;
- kesiapan implementasi;
- revisi yang diperlukan.

### P. Instrumen Delphi atau Expert Panel

Jika proposal menggunakan Delphi atau expert panel, buat paket instrumen untuk beberapa putaran:

- putaran 1: penilaian awal dan komentar bebas;
- putaran 2: revisi dan rating konsensus;
- putaran 3 jika diperlukan: konfirmasi final.

Sediakan:

- format rating;
- format komentar;
- kriteria konsensus;
- aturan keputusan retain/revise/drop;
- cara melaporkan perubahan antar-putaran.

### Q. Paket Analisis Validitas dan Reliabilitas

Berikan rumus, tabel perhitungan, dan interpretasi untuk:

- Aiken's V;
- I-CVI dan S-CVI/Ave;
- Lawshe CVR jika relevan;
- Cronbach's alpha atau McDonald's omega untuk skala persepsi;
- Cohen's Kappa untuk dua penilai;
- Fleiss' Kappa untuk lebih dari dua penilai;
- percent agreement;
- precision, recall, F1-score;
- confusion matrix;
- Wilson confidence interval;
- Jaccard top-k stability;
- Spearman rank correlation untuk stabilitas ranking;
- pass/fail report.

Sediakan ambang interpretasi awal yang wajar, tetapi tandai sebagai "dapat disesuaikan dengan keputusan promotor/ahli metodologi" jika belum ditetapkan dalam proposal.

Untuk setiap teknik analisis, berikan:

1. kapan teknik itu digunakan;
2. input data yang diperlukan;
3. rumus;
4. struktur tabel rekap;
5. contoh cara membaca hasil;
6. kategori interpretasi;
7. keputusan tindak lanjut terhadap instrumen/produk.

Jangan hanya menyebut nama teknik analisis. Proposal harus memiliki cara hitung dan cara interpretasi yang dapat dipahami pembaca.

### R. Template Rekap Data Excel

Buat struktur sheet Excel yang diperlukan, minimal:

1. `validator_identity`;
2. `instrument_blueprint`;
3. `expert_scores`;
4. `aiken_v`;
5. `cvi_cvr`;
6. `sentence_gold_labels`;
7. `span_gold_labels` (empat-tag BIO; kolom korpus: Indonesia/Global; split train/dev/test);
8. `cluster_review`;
9. `competency_review`;
10. `kkni_mapping_review`;
11. `dacum_worksheet`;
12. `dashboard_sus`;
13. `dashboard_task_observation`;
14. `coverage_gold`;
15. `reliability_iaa`;
16. `stability_ablation`;
17. `revision_log`;
18. `extraction_model_eval` (precision/recall/F1 per tag untuk tiap konfigurasi korpus pelatihan: Indonesia/Global/gabungan);
19. `future_weight_review` (validasi pembobotan tren berbasis WEF di level keterampilan).

Untuk setiap sheet, berikan daftar kolom, tipe data, contoh nilai, dan catatan validasi.

### S. Template Berita Acara dan Lembar Persetujuan

Buat template:

- lembar informasi penelitian untuk validator/responden;
- informed consent;
- berita acara validasi ahli;
- berita acara FGD/Delphi;
- log revisi berdasarkan masukan ahli.

Gunakan bahasa Indonesia formal, tetapi tetap praktis dan siap pakai.

---

## Format Penulisan

Gunakan Bahasa Indonesia akademik. Susun sebagai dokumen lengkap dengan heading yang jelas.

Setiap sitasi dan rujukan yang muncul dalam instrumen (mis. sumber SUS, rumus statistik, deskriptor KKNI, Laporan *Future of Jobs* WEF) ditulis **gaya APA edisi ke-7** — baik sitasi dalam teks (Penulis, tahun; *dkk.* untuk tiga penulis atau lebih) maupun daftar rujukan (alfabetis, indentasi gantung, DOI/URL bila ada). Selaraskan dengan Pedoman Pascasarjana UM bila menetapkan gaya lain.

Untuk setiap instrumen, gunakan format berikut:

1. Nama instrumen.
2. Tujuan instrumen.
3. Responden/validator.
4. Waktu penggunaan dalam tahap Borg and Gall.
5. Konstruk yang diukur.
6. Petunjuk pengisian.
7. Skala dan rubrik.
8. Kisi-kisi instrumen.
9. Lembar penilaian siap isi.
10. Pertanyaan terbuka/komentar.
11. Cara penskoran.
12. Kriteria interpretasi.
13. Keputusan tindak lanjut.
14. Format rekapitulasi skor.
15. Contoh interpretasi hasil.

Gunakan tabel siap salin. Format tabel butir minimal:

| No  | Aspek | Indikator | Butir Pernyataan/Penilaian | Skala | Skor | Catatan/Rekomendasi |
| --- | ----- | --------- | -------------------------- | ----- | ---- | ------------------- |

Untuk validasi ahli, gunakan skala 1-4:

- 1 = tidak valid/tidak sesuai/tidak jelas;
- 2 = kurang valid/kurang sesuai/kurang jelas;
- 3 = valid/sesuai/jelas;
- 4 = sangat valid/sangat sesuai/sangat jelas.

Untuk usability atau acceptance, boleh gunakan skala 1-5 jika lebih sesuai, tetapi jelaskan alasannya.

Setelah setiap tabel instrumen, tambahkan bagian:

### Cara Pengolahan Skor

Jelaskan cara menghitung skor per aspek, skor total, rerata, persentase kelayakan, atau indeks validitas yang sesuai.

### Cara Interpretasi

Berikan kategori interpretasi, misalnya sangat valid, valid, kurang valid, tidak valid; atau sangat layak, layak, perlu revisi, tidak layak. Jika kategori menggunakan ambang tertentu, tuliskan ambangnya secara eksplisit.

### Keputusan Revisi

Jelaskan keputusan yang harus diambil berdasarkan hasil, misalnya:

- diterima tanpa revisi;
- diterima dengan revisi minor;
- diterima dengan revisi mayor;
- perlu validasi ulang;
- tidak digunakan.

### Format Rekapitulasi

Berikan tabel rekap yang dapat dipindahkan ke Excel atau lampiran hasil penelitian.

---

## Batasan yang Harus Ditaati

1. Jangan mengklaim instrumen sudah valid sebelum diuji.
2. Jangan menganggap output LLM sebagai kebenaran tanpa validasi ahli.
3. Jangan menyatukan validasi teknis pipeline dengan validasi kurikulum; keduanya harus dipisah tetapi saling terhubung.
4. Jangan membuat status `official_ready` untuk data US/non-Indonesia.
5. Jangan menghapus kemungkinan hasil negatif. Instrumen harus memungkinkan ahli menyatakan bahwa suatu kompetensi tidak direkomendasikan dari korpus tertentu.
6. Jangan membuat butir yang terlalu umum seperti "sistem sudah baik". Setiap butir harus mengukur aspek spesifik.
7. Jika proposal mengandung bagian `[PERLU KONFIRMASI]`, jadikan itu daftar kebutuhan konfirmasi dan hubungkan ke instrumen atau keputusan metodologis.
8. Jangan menyerahkan detail pengisian, penghitungan, atau interpretasi kepada dashboard. Dashboard boleh disebut sebagai media pelaksanaan, tetapi instrumen proposal harus berdiri sendiri sebagai dokumen akademik.
9. Jangan menunda penyusunan instrumen sampai tahap pelaksanaan. Semua instrumen utama, termasuk petunjuk pengisian, scoring, rumus, dan interpretasi, harus ditulis sebagai lampiran proposal.
10. Setiap butir dan petunjuk harus dapat dipahami validator non-IT. Kode sistem selalu disertai padanan Indonesia dan glosarium ringkas; jangan menampilkan jargon teknis telanjang pada lembar yang diisi ahli kurikulum (lihat Prinsip Wajib Tambahan §1).
11. Tulis petunjuk dan narasi instrumen dengan suara akademik manusia, bukan pola mesin. Jangan mencantumkan angka hasil atau klaim capaian; sistem belum dibangun, jadi contoh ditandai `[contoh ilustratif]` (lihat Prinsip Wajib Tambahan §2 dan §3).

---

## Output Akhir yang Saya Inginkan

Berikan hasil dalam urutan berikut:

1. daftar lampiran instrumen yang akan masuk ke proposal;
2. ringkasan fungsi setiap instrumen;
3. matriks penelitian-instrumen-metrik;
4. definisi operasional konstruk;
5. semua instrumen siap isi;
6. semua panduan anotasi/gold labeling;
7. semua worksheet DACUM/CBET;
8. semua panduan wawancara/FGD/Delphi;
9. rumus, cara perhitungan, dan cara interpretasi;
10. template rekap skor dan sheet Excel;
11. template berita acara dan informed consent;
12. checklist kesiapan validasi;
13. daftar hal yang masih perlu dikonfirmasi kepada peneliti/promotor.

Jika jawaban terlalu panjang untuk satu respons, bagi menjadi beberapa bagian dan lanjutkan secara sistematis. Jangan meringkas bagian instrumen. Prioritaskan kelengkapan dan kesiapan pakai.
