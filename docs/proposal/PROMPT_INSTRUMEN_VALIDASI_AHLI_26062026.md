# Prompt untuk Menyusun Instrumen Validasi Ahli (Lampiran Proposal) — Edisi 26 Juni 2026

Versi **terfokus**: hanya **instrumen yang diisi oleh ahli/validator** (lembar validasi, telaah, wawancara/FGD, Delphi, worksheet DACUM) beserta **scaffolding** yang membuatnya dapat dipakai dan diskor (matriks induk, definisi operasional, paket validitas/reliabilitas, berita acara/consent).

**Sengaja TIDAK termasuk** (bukan diisi ahli): pelabelan emas/anotasi span (diisi annotator terlatih), SUS/usability dashboard (diisi pengguna akhir), coverage analyzer & metrik komputasi sistem (precision/recall/F1, span-F1, confusion matrix, Wilson, NDCG — dihitung dari data/sistem, bukan diisi ahli), dan template rekap Excel mentah. Instrumen non-ahli ini disusun terpisah pada tahap pengembangan, di luar lampiran ini.

**Lampirkan saat memakai prompt ini:** `Proposal_Disertasi_26062026.docx` (versi terbaru, wajib).

Catatan penggunaan:
1. Jika model tak dapat membaca lampiran, minta model **berhenti** dan meminta proposal diunggah ulang. Jangan menyusun instrumen tanpa membaca proposal.
2. Keluaran dirancang **bertahap**: prompt memuat *Protokol Penyampaian Bertahap* yang memerintah model memproduksi **satu Bagian per respons** (Bagian 0–8) lalu berhenti menunggu Anda mengetik **"lanjut"**. Anda cukup membalas "lanjut" tiap kali untuk bagian berikutnya — tidak perlu menyusun ulang permintaan.

---

## Prompt Utama (salin mulai dari sini ke claude.ai)

Anda adalah tim ahli multidisipliner: (1) ahli metodologi R&D pendidikan vokasi; (2) ahli kurikulum berbasis kompetensi (CBET); (3) ahli DACUM/pengembangan kurikulum vokasi berbasis okupasi; (4) ahli evaluasi pendidikan dan psikometri; (5) ahli NLP/LLM/data mining/sistem rekomendasi; (6) ahli Rekayasa Perangkat Lunak/Teknologi Informasi; (7) reviewer disertasi yang teliti terhadap validitas instrumen, validitas klaim, dan keterlacakan bukti.

Saya melampirkan proposal disertasi berjudul **"Analisis Kebutuhan Pasar Tenaga Kerja Bidang Rekayasa Perangkat Lunak Menggunakan Natural Language Processing (NLP) dan Large Language Model (LLM)"** (Listyanti Dewi Astuti, Pascasarjana Pendidikan Kejuruan, Universitas Negeri Malang, 2026).

Tugas Anda: **baca proposal secara menyeluruh, lalu susun paket lembar/instrumen yang akan DIISI OLEH AHLI/VALIDATOR**, lengkap dan siap pakai, sebagai **lampiran proposal/disertasi**. Fokus **hanya** pada instrumen yang diisi ahli (lembar validasi, telaah kompetensi/cluster/pemetaan, wawancara/FGD, Delphi, worksheet DACUM) ditambah scaffolding yang membuatnya dapat dipakai dan diskor (matriks induk, definisi operasional, rumus validitas/reliabilitas, berita acara/consent).

**Jangan** menyusun: lembar anotasi/pelabelan emas (diisi annotator terlatih, bukan ahli validasi); kuesioner SUS/usability dashboard (diisi pengguna akhir); instrumen coverage analyzer dan metrik komputasi sistem (dihitung dari data, bukan diisi ahli); template rekap Excel mentah. Bila perlu menyebutnya, cukup sebagai rujukan silang, bukan sebagai instrumen yang disusun di sini.

Keluaran berbentuk **naskah lampiran** formal, siap ditempel. Setiap instrumen harus dapat dipahami, diisi, dihitung, dan diinterpretasikan **tanpa bergantung pada dashboard apa pun**. Setiap instrumen wajib memuat: (1) identitas; (2) tujuan; (3) responden/validator; (4) waktu penggunaan (tahap Borg & Gall); (5) petunjuk pengisian; (6) skala + rubrik tiap skala; (7) kisi-kisi; (8) lembar kosong siap isi; (9) kolom skor; (10) kolom catatan/rekomendasi; (11) cara menghitung skor; (12) rumus analisis; (13) contoh rekapitulasi; (14) cara interpretasi; (15) kriteria keputusan/tindak lanjut. Jangan menulis "dapat dikembangkan kemudian" — susun **sekarang**, siap lampir.

---

## Protokol Penyampaian Bertahap (WAJIB)

Keluaran ini panjang. Sampaikan secara **bertahap**, **satu *Bagian* per respons**, agar mudah ditinjau dan tidak terpotong. Patuhi aturan berikut.

**Aturan penyampaian:**
1. Hasilkan **satu Bagian per respons**; jangan menggabung beberapa Bagian dalam satu respons.
2. Mulai tiap respons dengan judul Bagian dan daftar Lampiran yang dicakupnya.
3. Akhiri tiap respons dengan penanda "*Bagian X selesai*", satu kalimat tentang isi Bagian berikutnya, lalu kalimat **"Ketik 'lanjut' untuk melanjutkan."** Setelah itu **berhenti dan tunggu** — jangan meneruskan sebelum diminta.
4. **Jangan meringkas atau memadatkan** isi instrumen demi memuat lebih banyak dalam satu respons; kelengkapan lebih utama daripada keringkasan.
5. Bila satu Bagian masih terlalu panjang untuk satu respons, pecah lagi (mis. *Bagian 5a*, *5b*), nyatakan di mana Anda berhenti, lalu minta "lanjut".
6. Saat diminta "lanjut", teruskan dari Lampiran yang **belum** selesai; jangan mengulang yang sudah dibuat.

**Peta Bagian (rencana baku; boleh disesuaikan bila ada Bagian yang terlalu berat):**
- **Bagian 0 — Pendahuluan:** daftar lampiran (Lampiran 1–15) + ringkasan fungsi tiap instrumen + peta penyampaian ini.
- **Bagian 1 — Fondasi:** Lampiran 1 (Matriks Induk) + Lampiran 2 (Definisi Operasional Konstruk).
- **Bagian 2 — Validasi Kurikulum & Industri:** Lampiran 3 + Lampiran 4.
- **Bagian 3 — Validasi Teknis & Skema Pelabelan:** Lampiran 5 + Lampiran 6.
- **Bagian 4 — Psikometri & Telaah Cluster:** Lampiran 7 + Lampiran 8.
- **Bagian 5 — Instrumen Inti:** Lampiran 9 (Competency Statement & Evidence Package), disajikan tersendiri karena paling besar.
- **Bagian 6 — Pemetaan & DACUM:** Lampiran 10 + Lampiran 11.
- **Bagian 7 — Wawancara & Delphi:** Lampiran 12 + Lampiran 13.
- **Bagian 8 — Analisis & Administrasi:** Lampiran 14 + Lampiran 15 + checklist kesiapan validasi + daftar hal yang perlu dikonfirmasi.

Mulailah sekarang dengan **Bagian 0**, lalu berhenti dan tunggu "lanjut".

---

## Prinsip Wajib (setara penting dengan kelengkapan)

**1. Aksesibilitas lintas-bidang & ragam bahasa.** Sebagian validator adalah **ahli kurikulum/pendidikan tanpa latar IT**, dan proposal diuji penguji lintas-bidang. **Ragam bahasa wajib: Indonesia akademik yang formal tetapi lugas** — kalimat efektif dan langsung, hindari jargon teknis telanjang, dan jelaskan tiap istilah teknis dengan bahasa sehari-hari pada kemunculan pertamanya sebelum menyebut istilah aslinya. Tolok ukur keterbacaan: seorang **ahli kurikulum, guru, atau penguji tanpa latar TI harus dapat memahami setiap butir, petunjuk, dan definisi tanpa bantuan tambahan** — tanpa mengorbankan ketepatan bagi validator teknis. Tulis **berlapis**: satu kalimat yang dipahami ahli kurikulum, diikuti istilah presisi (cetak miring/kurung). Kode sistem (mis. `valid_cluster`, `tool-technology`, `siap ditelaah formal`) selalu disertai padanan Indonesia; sediakan **glosarium ringkas** di awal tiap lembar yang memuat kode. Gunakan analogi yang dikenal ahli kurikulum (panel/lokakarya DACUM, kartu tugas, berkas perkara/bukti). Jangan menyederhanakan sampai dangkal — layani validator kurikuler dan validator teknis sekaligus.

**2. Suara manusia akademik.** Petunjuk, pertanyaan wawancara/FGD/Delphi, lembar informasi, dan narasi harus terbaca seperti tulisan akademisi: kalimat bervariasi; tanpa pola "Pertama… Kedua… Ketiga…" yang kaku; tanpa penguat kosong ("sangat penting", "krusial"); tanpa pembuka monoton berulang. Butir penilaian boleh ringkas; narasi di sekitarnya alami.

**3. Kala rencana (sistem belum dibangun).** Proposal ini **rencana**; belum ada sistem/hasil. Instrumen dirancang terhadap **luaran yang direncanakan**. Tandai tiap contoh kompetensi/kalimat/cluster sebagai `[contoh ilustratif]`. **Jangan** menulis angka hasil/metrik capaian/klaim keberhasilan.

**4. Penyelarasan dengan kisi-kisi Bab III proposal.** Rujuk **berdasarkan judul tabel + subbab** (naskah memuat penomoran tabel berulang — ada "Tabel 3.1"/"Tabel 3.2" di Subbab 3.3.2 *dan* di Subbab 3.5):
- **Kisi-Kisi Wawancara Ahli Kurikulum & Praktisi Industri (Subbab 3.5)** — relevansi okupasi; kesesuaian DACUM; kesesuaian CBET; pemetaan KKNI/Spektrum/SKKNI; kegunaan berkas bukti; risiko rekomendasi.
- **Kisi-Kisi Kuesioner & Lembar Validasi Produk (Subbab 3.5)** — keabsahan korpus; kualitas ekstraksi; koherensi pengelompokan; kualitas pernyataan kompetensi; kontrak rekomendasi; analisis cakupan; kegunaan papan informasi. *(Untuk lampiran ini, ambil hanya indikator yang dinilai ahli; sisihkan yang dihitung sistem.)*
- **Pemetaan Metrik Evaluasi → Tahap Borg & Gall (Subbab 3.5)** — sumber kolom "Tahap Borg & Gall" pada Matriks Induk.

Pastikan tiap indikator yang relevan punya butir bersesuaian, dan kaitkan tiap instrumen ke **tujuan penelitian (Subbab 1.2)** dan **pertanyaan penelitian (PP1–PP4)**. Bila PP tak tertulis eksplisit, turunkan dari empat tujuan dan nyatakan asumsinya. Bila kisi-kisi naskah berbeda dari ringkasan ini, **ikuti naskah**.

**5. Bingkai PIPELINE-FIRST + LIMA-TAG (sebagai objek penilaian ahli).**
- Produk yang divalidasi adalah **pipeline analisis**, bukan "sebuah sistem". **Dashboard/papan informasi** = pelaksana pipeline + penyaji hasil. Gunakan "pipeline/alur kerja", kecuali saat memang merujuk dashboard.
- **CBET = grand theory.** **DACUM dwiperan** (tanpa istilah "DACUM digital"): (a) landasan kerja — kebutuhan kerja nyata mendefinisikan kompetensi → alasan rekomendasi pipeline layak jadi dasar kurikulum; (b) kerangka hilir institusi: `okupasi/spektrum → duty area → task evidence → tools/technology → hard skill/knowledge/soft skill → work behavior → competency statement → curriculum recommendation → assessment idea`.
- **Penetapan KKNI langsung, tanpa Taksonomi Bloom sebagai jembatan.** Jenjang KKNI ditetapkan via pencocokan makna (*semantic matching* dengan *sentence embedding* seperti SBERT) antara pernyataan kompetensi dan deskriptor resmi KKNI (kompleksitas, otonomi, tanggung jawab), lalu **divalidasi ahli**. Instrumen KKNI **tidak** menilai "Bloom level". Ahli justru menilai **apakah metode pencocokan ini masuk akal** dan apakah jenjang hasilnya tepat.
- **Future-aware di level keterampilan.** *Future weight* berbasis **Laporan Future of Jobs WEF (2025)** dihitung pada **keterampilan**, lalu diagregasi ke kompetensi. **Aturan kejujuran:** bobot bersumber bukti WEF, **bukan tebakan model**; bobot dari saran model ditandai & tak dipakai dalam pemeringkatan publik. Sediakan butir bagi ahli untuk menilai **ketepatan dan kejujuran** pembobotan ini.
- **Skema ekstraksi LIMA-TAG** — `skill-hard`, `skill-soft`, `knowledge`, `tool-technology`, `work-behavior` (notasi BIO/BIOES di tingkat span). Dalam lampiran ini, lima-tag muncul sebagai **objek yang dinilai ahli** (mis. ahli NLP/AI menilai ketepatan skema; ahli RPL/TI & kurikulum menilai apakah keterampilan/pengetahuan/tools/soft skill/sikap kerja yang ditautkan ke suatu kompetensi sudah tepat). `tool-technology` sengaja terpisah agar istilah teknologi (Python, Docker, React, PostgreSQL, AWS — `[contoh ilustratif]`) **tidak** langsung diperlakukan sebagai kompetensi.

---

## Konteks Penelitian yang Harus Dipertahankan

**Rantai bukti (evidence-first, Subbab 3.3.1)** — setiap rekomendasi kompetensi dapat ditelusuri mundur:
`korpus mentah → audit lineage → penyiapan & audit relevansi kalimat → ekstraksi lima-tag → penggolongan deterministik → tabel bukti → pengelompokan + audit cluster → audit tema permintaan → competency statement terikat bukti → antihalusinasi → perbaikan bukti → dedup → pemetaan KKNI/pendidikan/okupasi/SKKNI/Spektrum → kontrak rekomendasi → dashboard + evidence package + paket validasi`.
Ahli menilai mata rantai ini sejauh tampak pada berkas bukti yang disajikan kepadanya.

**Posisi teori:** CBET (utama) → competency statement *observable, assessable, performance-based*, diawali kata kerja, bukan daftar topik/tools. DACUM (dwiperan). Human Capital & Signaling Theory hanya pendukung.

**Prinsip ilmiah:** (1) korpus US/non-Indonesia hanya uji rekayasa, bukan klaim kurikulum Indonesia; (2) klaim Indonesia hanya dari korpus Indonesia valid + lineage lengkap + validasi ahli; (3) LLM bukan otoritas kurikulum; (4) kontrak rekomendasi **tiga kelas** (Subbab 3.3.5): **Eksploratif** (sinyal awal, bukti terbatas — hanya pemantauan); **Kandidat tinjauan** (korpus valid, evidence package lengkap, cluster koheren, grounding memadai — bahan diskusi ahli); **Siap ditelaah formal** (lolos audit korpus/ekstraksi/grounding/pemetaan + validasi ahli/Delphi + stabilitas — bahan telaah formal, bukan keputusan otomatis); (5) "siap ditelaah formal" **tak pernah** untuk korpus non-Indonesia; (6) bila bukti lemah/campuran/kurang → **tidak direkomendasikan dari korpus tersebut**; status dapat naik/turun; (7) instrumen mengukur **validitas komputasional dan kurikuler sekaligus** (di lampiran ini: sisi yang dinilai ahli).

**Protokol benchmark model (Subbab 3.3.2)** — ahli NLP/AI menilai kelayakan rancangan perbandingan 6 pendekatan: rule-based/dictionary; encoder sequence labeling (lima-tag BIO); domain-adapted encoder (JobBERT); open-type IE; prompted LLM tanpa retrieval; **retrieval-verified schema-constrained LLM** (kandidat utama). Juga **eksperimen korpus pelatihan**: Indonesia / Global-multibahasa / gabungan-transfer (korpus global tak menjadi dasar rekomendasi formal Indonesia).

**Subjek & panel (Subbab 3.4):** sekurang-kurangnya **3 reviewer ahli** untuk penilaian kualitas ekstraksi & kompetensi; **panel Delphi 7–9 orang** (menyeimbangkan ahli kurikulum vokasi, praktisi industri perangkat lunak, asesor kompetensi). **Empat unit analisis** (perjelas denominator): dokumen (lowongan), kalimat, span (entitas lima-tag), rekomendasi (kompetensi).

---

## Konstruk yang Divalidasi Ahli (untuk Definisi Operasional & butir)

Validitas korpus & lineage (sejauh tampak pada berkas bukti); kualitas ekstraksi & ketepatan **typing lima-tag**; ketepatan **skema anotasi lima-tag** & rancangan benchmark/eksperimen korpus; koherensi cluster & keputusan split/merge/drop; validitas audit tema permintaan; **kualitas competency statement (CBET)**; **grounding** (kesesuaian evidence ↔ statement); ketepatan **jenjang KKNI** & **metode penetapannya** (tanpa Bloom); pemetaan **SKKNI/Spektrum/okupasi** (resmi vs komunitas); **ketepatan & kejujuran future weight** (WEF, level keterampilan); kesesuaian **DACUM** (duty→task→skill/knowledge→competency); ketepatan **ranking sadar-golongan** & **kontrak rekomendasi**; risiko **overclaim/halusinasi/pencampuran domain/bias korpus**; kelayakan luaran sebagai **bahan telaah kurikulum** (SMK/Diploma/Sarjana vokasi). Tiap definisi memuat: konseptual; operasional; indikator; contoh bukti `[contoh ilustratif]`; cara pengukuran; kriteria interpretasi. (Selaraskan dengan **Tabel 1.1 Definisi Operasional** proposal.)

---

## Keluaran Wajib (paket lampiran bernomor)

Untuk **setiap** lampiran beri tiga bagian: (1) **Panduan Pengisian**; (2) **Formulir Siap Isi** (tabel kosong); (3) **Panduan Pengolahan & Interpretasi** (rumus, contoh rekap, kategori interpretasi, kriteria keputusan).

**Lampiran 1 — Matriks Induk Instrumen (Ahli).** Tabel: tujuan penelitian → PP1–PP4 → konstruk → indikator → sumber data → responden/penilai → **tahap Borg & Gall** → instrumen (hanya yang diisi ahli) → bentuk data → teknik analisis → kriteria keputusan.

**Lampiran 2 — Definisi Operasional Konstruk.** Untuk konstruk pada daftar di atas; format: konseptual, operasional, indikator, contoh bukti `[contoh ilustratif]`, cara ukur, kriteria interpretasi.

**Lampiran 3 — Lembar Validasi Ahli Kurikulum CBET/DACUM.** Identitas validator; petunjuk; **skala 1–4 tanpa netral** + rubrik; butir; kolom skor; komentar; rekomendasi revisi. Aspek: (1) kesesuaian competency statement dgn CBET; (2) keterukuran & keteramatan; (3) ketepatan action verb/objek kerja/konteks/standar performa; (4) kesesuaian DACUM duty-task-skill-knowledge; (5) kelayakan sebagai bahan kurikulum; (6) pemisahan kompetensi inti teknis vs proses/leadership vs employability/support vs out-of-domain; (7) risiko overclaim dari evidence pasar kerja; (8) kejelasan status rekomendasi. Keputusan akhir: layak tanpa revisi / revisi kecil / revisi besar / tidak layak.

**Lampiran 4 — Lembar Validasi Ahli RPL/TI & Praktisi Industri.** Untuk software engineer senior/tech lead/HR teknis/asesor RPL/TI. Aspek: (1) relevansi kompetensi dgn kerja RPL/TI; (2) ketepatan task evidence dari lowongan; (3) kecukupan keterampilan teknis; (4) kecukupan knowledge; (5) kecukupan soft skill relevan; (6) ketepatan pemetaan okupasi/fungsi kerja; (7) ketepatan prioritas demand; (8) terlalu luas/sempit/redundant/missing; (9) kelayakan jenjang SMK/Diploma/Sarjana vokasi; (10) status yang sesuai (kandidat tinjauan / tidak direkomendasikan dari korpus / dapat dipertimbangkan "siap ditelaah formal" setelah syarat formal).

**Lampiran 5 — Lembar Validasi Ahli NLP/AI/Data Mining.** Aspek: (1) validitas pipeline evidence-first; (2) kecukupan lineage; (3) penyaringan relevansi kalimat; (4) kualitas span ekstraksi; (5) **typing lima-tag**; (6) cluster validity; (7) demand theme audit; (8) antihalusinasi/grounding; (9) deduplication; (10) ranking demand-first sadar-golongan; (11) desain ablation/stability; (12) keterbatasan LLM & risiko bias; (13) rancangan **eksperimen korpus** + **protokol benchmark 6 pendekatan** (penilaian rinci atas skema pelabelan lima-tag dilakukan pada Lampiran 6); (14) ketepatan & **kejujuran future weight** (WEF, level keterampilan). Instrumen menilai apakah **pipeline** layak dipercaya secara teknis.

**Lampiran 6 — Lembar Validasi Ahli atas Skema Pelabelan Lima-Tag (pedoman anotasi/codebook).** Menilai **rancangan skema pelabelan** (definisi tag, aturan batas, contoh, aturan adjudikasi) **sebelum** dipakai annotator — **bukan** menilai hasil anotasi (anotasi span diisi annotator terlatih dan berada di luar lampiran ini). Validator: ahli NLP/linguistik komputasi, ahli RPL/TI, dan ahli kurikulum (untuk keterkaitan hilir). Skala 1–4 + rubrik; kolom skor, komentar, usulan revisi (definisi/contoh/aturan batas).
- *Aspek per-tag* (dinilai untuk SETIAP tag — skill-hard, skill-soft, knowledge, tool-technology, work-behavior): (a) kejelasan & ketepatan definisi; (b) kecukupan & keterwakilan contoh positif/negatif `[contoh ilustratif]`; (c) kejelasan batas terhadap tag lain (*mutual exclusivity*); (d) keterterapan lintas korpus Indonesia & Global (definisi tetap valid lintas bahasa).
- *Aspek tingkat skema (menyeluruh):* (1) kelengkapan/cakupan — apakah lima tag cukup mencakup entitas kebutuhan kerja pada lowongan RPL/TI tanpa celah besar; (2) risiko tumpang tindih/ambiguitas antar-tag, khususnya **tool-technology vs knowledge vs skill-hard** dan **skill-soft vs work-behavior**; (3) ketepatan & kejelasan aturan batas span + notasi BIO/BIOES; (4) kejelasan aturan kasus ambigu & prosedur adjudication; (5) justifikasi perluasan dari SkillSpan (skill/knowledge) menjadi lima tag — apakah penambahan tool-technology & work-behavior beralasan dan tidak redundan; (6) kesesuaian skema dengan tujuan penelitian: memisahkan bukti teknologi dari kompetensi serta mendukung penggolongan deterministik & perumusan kompetensi CBET di hilir; (7) prediksi keterandalan anotasi — apakah definisi cukup operasional untuk dianotasi reliabel oleh ≥2 annotator (dugaan risiko IAA rendah); (8) kejelasan pemetaan ke kategori hilir bila pipeline memecah lebih lanjut (mis. credential/education requirement, out-of-domain, invalid fragment).
- *Bagian khusus batas tag:* untuk tiap pasangan tag yang rawan tertukar, sediakan baris "kejelasan aturan pembeda" + kolom usulan perbaikan aturan batas.
- *Keputusan skema:* (a) diterima tanpa revisi — boleh **dibekukan** (freeze versi) & diserahkan ke annotator; (b) revisi kecil — perjelas definisi/tambah contoh; (c) revisi besar — gabung/pisah tag atau tulis ulang aturan batas → validasi ulang; (d) tidak memadai.
- *Pengolahan:* validitas isi per butir/per tag dengan **Aiken's V** dan/atau **I-CVI**, serta **S-CVI/Ave** untuk skema keseluruhan (rujuk Lampiran Analisis Validitas & Reliabilitas). Skema hanya boleh **dibekukan dan diserahkan ke annotator** setelah mencapai ambang validitas isi yang ditetapkan.

**Lampiran 7 — Lembar Validasi Ahli Pengukuran/Evaluasi (Psikometri).** Aspek: (1) kesesuaian instrumen dgn tujuan; (2) kejelasan konstruk; (3) validitas isi; (4) kejelasan butir; (5) kejelasan skala; (6) kelayakan teknik analisis; (7) reliabilitas antar-penilai; (8) kecukupan validator; (9) kriteria pass/fail; (10) risiko bias penilaian.

**Lampiran 8 — Lembar Telaah Cluster (oleh ahli).** Status: `valid_cluster`, `split_required`, `mixed_domain`, `support_process_cluster`, `out_of_domain_cluster`, `weak_low_frequency_theme`, `drop_invalid` (sertai padanan Indonesia + glosarium). Aspek: koherensi semantik; kesamaan fungsi kerja; kesamaan domain teknis; kecukupan source jobs; campuran soft skill/credential/out-of-domain; kebutuhan split/merge; kelayakan masuk perumusan kompetensi. Sertakan rubrik skor, tabel siap isi, aturan keputusan.

**Lampiran 9 — Lembar Validasi Competency Statement & Evidence Package (instrumen inti).** Satu kompetensi per baris/form, elemen tampil: `competency_id`, `competency_statement`, `competency_class`, `recommendation_status`, `job_prevalence`, `source_job_count`, kalimat bukti, related skills/knowledge/soft skills/**tools-technology**, label KKNI, pemetaan okupasi/SKKNI/Spektrum, future_weight (+ sumber: WEF/saran model), blockers, alasan "tidak direkomendasikan" bila ada. Aspek (skala 1–4): relevansi RPL/TI; grounding; tidak hallucinated; tidak terlalu luas/sempit; tidak redundant; assessable; observable; sesuai CBET; sesuai DACUM; tepat jenjang KKNI; tepat pemetaan okupasi/SKKNI/Spektrum; layak bahan telaah kurikulum; status sudah jujur. Keputusan: `accept_as_review_candidate`, `revise_statement`, `merge_with_other`, `split_required`, `demote_to_support`, `mark_out_of_domain`, `not_recommended_from_this_corpus`, `candidate_for_official_ready_after_validation` (semua dengan padanan Indonesia).

**Lampiran 10 — Lembar Validasi KKNI, SKKNI, Spektrum, Okupasi.** Menilai: ketepatan **jenjang KKNI** (kompleksitas/otonomi/tanggung jawab) — **tanpa "Bloom level"**; **ketepatan metode penetapan KKNI** (apakah pencocokan makna ke deskriptor resmi masuk akal); kesesuaian deskriptor KKNI resmi; okupasi/fungsi kerja; SKKNI/unit kompetensi; Spektrum Keahlian (konteks SMK); status **pemetaan resmi vs komunitas**. Peringatan: pemetaan komunitas tak boleh dianggap resmi sebelum divalidasi & dirujuk ke dokumen resmi versi tertentu.

**Lampiran 11 — Worksheet DACUM/CBET (lokakarya ahli).** Siap isi: `okupasi/skema → duty area → task evidence → tools/technology → hard skill → knowledge → soft skill → work behavior → competency statement → curriculum implication → assessment idea`. Dapat dipakai di workshop ahli atau Delphi.

**Lampiran 12 — Panduan Wawancara & FGD Ahli.** Untuk ahli kurikulum; praktisi industri; guru SMK RPL/Pengembangan Gim; dosen vokasi/TI; (dan pengguna sebagai informan kualitatif bila relevan). Pakai aspek **Kisi-Kisi Wawancara §3.5**: relevansi okupasi, kesesuaian DACUM, kesesuaian CBET, pemetaan KKNI/Spektrum/SKKNI, kegunaan berkas bukti, risiko rekomendasi. Gali: kepercayaan terhadap rekomendasi; kecukupan evidence package; risiko salah interpretasi; relevansi kurikulum; kesiapan implementasi; revisi yang diperlukan. Sediakan pertanyaan pembuka, lanjutan/probe, dan penutup; format catatan.

**Lampiran 13 — Paket Delphi / Panel Ahli (multi-putaran).** Panel **7–9 orang** (ahli kurikulum vokasi, praktisi industri, asesor kompetensi). Putaran 1 (penilaian awal + komentar bebas); putaran 2 (revisi + rating konsensus); putaran 3 (konfirmasi final, bila perlu). Sediakan format rating, format komentar, **kriteria konsensus** (mis. persentase kesepakatan, IQR, Kendall's W) ditetapkan **sebelum** analisis, aturan retain/revise/drop, cara melaporkan perubahan antar-putaran. Rekomendasi yang tak mencapai konsensus **diturunkan** ke kandidat tinjauan/eksploratif, bukan dipaksakan.

**Lampiran 14 — Paket Analisis Validitas & Reliabilitas (skor ahli).** Rumus + tabel hitung + interpretasi untuk teknik yang memproses **penilaian ahli**: **Aiken's V**; **I-CVI & S-CVI/Ave**; **Lawshe CVR** (bila relevan); **Cohen's Kappa** (2 penilai); **Fleiss' Kappa** (>2 penilai); **percent agreement**; **Kendall's W** (konsensus Delphi); **Cronbach's alpha / McDonald's omega** (bila ada skala persepsi ahli). Untuk tiap teknik: kapan dipakai; input data; rumus; struktur tabel rekap; contoh membaca hasil; kategori interpretasi; keputusan tindak lanjut. Sertakan **ambang awal yang wajar**, tandai "dapat disesuaikan dengan keputusan promotor/ahli metodologi" bila proposal belum menetapkan. *(Metrik komputasi sistem — span-F1, confusion matrix, Wilson, Jaccard, NDCG — di luar lampiran ini karena tak diisi ahli; cukup dirujuk silang.)*

**Lampiran 15 — Template Berita Acara & Informed Consent (sesi ahli).** Lembar informasi penelitian untuk validator; informed consent; berita acara validasi ahli; berita acara FGD/Delphi; log revisi berdasarkan masukan ahli. Selaraskan dengan **Etika Penelitian §3.7** (kerahasiaan identitas, hak menarik diri, data hanya untuk penelitian). Bahasa Indonesia formal, praktis, siap pakai.

---

## Format Penulisan

Bahasa Indonesia akademik yang **formal namun lugas dan mudah dipahami pembaca non-teknis** (lihat Prinsip Wajib §1); heading jelas. Sitasi/rujukan (sumber rumus statistik, deskriptor KKNI, Laporan Future of Jobs WEF) **gaya APA edisi ke-7** (dalam teks: Penulis, tahun; *dkk.* untuk ≥3 penulis; daftar rujukan alfabetis, indentasi gantung, DOI/URL bila ada). Selaraskan dgn Pedoman Pascasarjana UM bila berbeda.

Tiap instrumen ikuti urutan: (1) Nama; (2) Tujuan; (3) Responden/validator; (4) Waktu (tahap Borg & Gall); (5) Konstruk; (6) Petunjuk; (7) Skala & rubrik; (8) Kisi-kisi; (9) Lembar siap isi; (10) Pertanyaan terbuka/komentar; (11) Cara penskoran; (12) Kriteria interpretasi; (13) Keputusan tindak lanjut; (14) Format rekapitulasi; (15) Contoh interpretasi.

Format tabel butir minimal:

| No | Aspek | Indikator | Butir Penilaian | Skala | Skor | Catatan/Rekomendasi |
| -- | ----- | --------- | --------------- | ----- | ---- | ------------------- |

**Skala validasi ahli 1–4 (tanpa netral):** 1 = tidak valid/sesuai/jelas; 2 = kurang; 3 = valid/sesuai/jelas; 4 = sangat valid/sesuai/jelas.

Setelah tiap tabel instrumen, tambahkan: **Cara Pengolahan Skor** (per aspek, total, rerata, persentase kelayakan/indeks validitas); **Cara Interpretasi** (kategori + ambang eksplisit); **Keputusan Revisi** (diterima tanpa revisi / revisi minor / revisi mayor / validasi ulang / tidak digunakan); **Format Rekapitulasi** (tabel siap pindah ke pengolah angka).

---

## Batasan yang Harus Ditaati

1. Susun **hanya** instrumen yang diisi ahli/validator + scaffolding-nya. Jangan menyusun lembar anotasi/pelabelan emas, SUS/usability, coverage analyzer, metrik komputasi sistem, atau rekap Excel mentah.
2. Jangan mengklaim instrumen sudah valid sebelum diuji.
3. Jangan menganggap output LLM kebenaran tanpa validasi ahli.
4. Pisahkan validitas teknis pipeline dari validitas kurikulum, tetapi hubungkan keduanya.
5. Jangan beri status "siap ditelaah formal" untuk data US/non-Indonesia.
6. Pertahankan kemungkinan hasil negatif: ahli harus dapat menyatakan kompetensi **tidak direkomendasikan dari korpus tertentu**.
7. Tiap butir mengukur aspek **spesifik** (hindari "sistem sudah baik").
8. Instrumen berdiri sendiri sebagai dokumen akademik; jangan bergantung pada dashboard.
9. Susun **sekarang**, lengkap dengan petunjuk, scoring, rumus, interpretasi.
10. Dapat dipahami validator non-IT; kode sistem selalu + padanan Indonesia + glosarium.
11. Suara akademik manusia; tanpa angka hasil/klaim capaian; contoh `[contoh ilustratif]`.
12. Gunakan skema **LIMA-TAG** (skill-hard, skill-soft, knowledge, tool-technology, work-behavior) secara konsisten saat lima-tag menjadi objek penilaian ahli; jangan memakai skema empat-tag lama.

---

## Output Akhir yang Saya Inginkan (urutan)

1. daftar lampiran instrumen (Lampiran 1–15) yang masuk proposal;
2. ringkasan fungsi tiap instrumen;
3. Matriks Induk (penelitian–instrumen–metrik);
4. definisi operasional konstruk;
5. semua lembar validasi/telaah ahli siap isi (Lampiran 3–10), termasuk **validasi skema pelabelan lima-tag (Lampiran 6)**;
6. worksheet DACUM/CBET (Lampiran 11);
7. panduan wawancara/FGD/Delphi (Lampiran 12–13);
8. paket validitas & reliabilitas — rumus, cara hitung, cara interpretasi (Lampiran 14);
9. template berita acara & informed consent (Lampiran 15);
10. checklist kesiapan validasi;
11. daftar hal yang perlu dikonfirmasi ke peneliti/promotor: penomoran tabel ganda di Bab III; ambang konsensus Delphi; rumus agregasi future weight; ambang Aiken's V / Kappa / CVI; jumlah & komposisi final validator.

Sampaikan seluruh keluaran **bertahap sesuai *Protokol Penyampaian Bertahap*** di atas — satu Bagian per respons (Bagian 0–8), akhiri tiap respons dengan permintaan "lanjut", lalu berhenti dan tunggu. **Jangan meringkas** bagian instrumen. Prioritaskan kelengkapan dan kesiapan pakai.
