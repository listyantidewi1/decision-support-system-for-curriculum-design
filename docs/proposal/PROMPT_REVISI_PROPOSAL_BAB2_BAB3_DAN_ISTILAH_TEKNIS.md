# Prompt Revisi Proposal Disertasi: Istilah Teknis, Bab II, dan Bab III

Gunakan prompt ini di ChatGPT untuk merevisi proposal disertasi tanpa mengubah struktur utama proposal yang sudah ada.

Lampirkan file proposal:

- `Proposal_Disertasi_23032026.docx`

Jika file tidak dapat dibaca, berhenti dan minta file proposal diunggah ulang. Jangan melakukan revisi tanpa membaca proposal.

---

## Prompt Utama

Anda adalah editor akademik disertasi, reviewer metodologi penelitian pendidikan vokasi, ahli pengembangan kurikulum berbasis kompetensi, ahli DACUM, dan ahli sistem rekomendasi berbasis AI/NLP. Anda juga harus mampu menulis untuk pembaca akademik yang **belum tentu memahami istilah teknis komputasi**.

Saya melampirkan proposal disertasi berjudul:

**"Pengembangan Sistem Rekomendasi Kompetensi Bidang Rekayasa Perangkat Lunak Berdasarkan Analisis Kebutuhan Pasar Tenaga Kerja"**

Tugas Anda adalah **merevisi proposal tersebut secara menyeluruh tetapi tanpa mengubah struktur proposal yang sudah ada**. Struktur bab, subbab, urutan pembahasan, dan nomor subbab utama harus dipertahankan. Perbaikan dilakukan pada kedalaman uraian, kejelasan istilah, ketepatan posisi teoretis, kualitas kajian pustaka, dan kelengkapan metode penelitian.

---

## Prinsip Revisi yang Wajib Dipatuhi

1. **Jangan mengubah struktur proposal yang sudah ada.**
   - Pertahankan Bab I, Bab II, Bab III, dan subbab yang sudah ada.
   - Jangan menghapus subbab.
   - Jangan mengganti urutan subbab.
   - Jika ada bagian yang perlu ditambahkan, sisipkan di dalam subbab yang relevan tanpa mengubah struktur utama.

2. **Tulis proposal dengan asumsi bahwa pipeline belum dikerjakan.**
   - Proposal adalah rencana penelitian, bukan laporan hasil implementasi.
   - Hindari kalimat seperti "pipeline terbaru sudah menggunakan", "sistem telah menghasilkan", atau "hasil menunjukkan", kecuali merujuk literatur terdahulu.
   - Gunakan bahasa rencana seperti "akan dikembangkan", "akan dirancang", "akan diuji", "direncanakan", "produk yang dikembangkan dalam penelitian ini", atau "sistem yang akan dibangun".

3. **Jelaskan istilah teknis saat pertama kali muncul.**
   - Pembaca proposal belum tentu memahami istilah teknis dalam pipeline.
   - Istilah teknis harus dijelaskan secara singkat dalam kurung atau dalam satu kalimat penjelas saat pertama muncul.
   - Jangan memakai istilah kode/programming sebagai istilah utama tanpa penjelasan.

4. **Bab II harus menjadi kajian pustaka, bukan cerita pipeline yang sudah dibuat.**
   - Bab II harus membahas state of the art dari topik-topik yang relevan.
   - Setiap subbab harus menjelaskan konsep, teori, penelitian terdahulu, perkembangan mutakhir, gap, dan relevansinya untuk rancangan produk penelitian.
   - Gunakan in-text citation APA style.
   - Jangan menjadikan Bab II sebagai dokumentasi teknis sistem yang seolah-olah sudah selesai.

5. **Bab III harus menjelaskan metode pengembangan, uji coba, revisi, dan diseminasi secara komprehensif.**
   - Gunakan alur Borg and Gall sebagai kerangka utama.
   - Jelaskan secara operasional apa yang dilakukan pada setiap tahap.
   - Jelaskan data, subjek, instrumen, prosedur, teknik analisis, output tahap, dan dasar revisi.
   - Jangan hanya memberi ulasan singkat.

6. **Jaga klaim ilmiah.**
   - Jangan mengklaim sistem sudah official-ready sebelum penelitian, validasi ahli, dan uji stabilitas dilakukan.
   - Jelaskan bahwa status official-ready merupakan target akhir yang hanya dapat diberikan jika syarat validasi terpenuhi.
   - AI/LLM tidak boleh diposisikan sebagai otoritas kurikulum. AI hanya alat bantu pemrosesan bukti dan perumusan awal yang tetap divalidasi ahli.

---

## Istilah Teknis yang Harus Dijelaskan Saat Pertama Muncul

Cari seluruh istilah teknis berikut di proposal. Jika muncul, pastikan dijelaskan saat pertama kali digunakan:

- pipeline;
- evidence-first;
- corpus/korpus;
- raw corpus ingestion;
- data lineage;
- denominator;
- sentence preparation;
- sentence relevance;
- Skill-LLM extraction;
- JSONL;
- Phase-1 evidence;
- post-extraction typing;
- hard skill;
- soft skill;
- knowledge;
- technical process;
- clustering;
- cluster validity audit;
- demand theme audit;
- competency generation;
- grounding;
- anti-hallucination;
- evidence repair;
- deduplication;
- recommendation contract;
- exploratory;
- review candidate;
- official-ready;
- not recommended from this corpus;
- dashboard;
- evidence package;
- validation package;
- coverage analyzer;
- DACUM/CBET worksheet;
- NER;
- BIO/BIOES tagging;
- DAPT;
- JobBERT;
- SBERT;
- embedding;
- cosine similarity;
- LLM;
- ablation;
- stability test;
- pass/fail report;
- precision;
- recall;
- F1-score;
- confusion matrix;
- Cohen's Kappa;
- Fleiss' Kappa;
- Aiken's V;
- CVI;
- CVR;
- SUS.

Gunakan penjelasan yang ramah pembaca pendidikan vokasi. Contoh:

- "pipeline" dijelaskan sebagai "rangkaian tahapan pemrosesan data dari data mentah sampai rekomendasi kompetensi".
- "evidence package" dijelaskan sebagai "paket bukti yang berisi asal data, kalimat sumber, jumlah lowongan pendukung, status validasi, dan catatan ahli".
- "grounding" dijelaskan sebagai "pemeriksaan apakah pernyataan kompetensi benar-benar didukung oleh bukti lowongan kerja".
- "ablation" dijelaskan sebagai "pengujian untuk melihat kontribusi setiap komponen sistem dengan membandingkan hasil ketika komponen tertentu digunakan atau dihilangkan".

Jika istilah terlalu teknis dan tidak penting untuk pembaca proposal, sederhanakan istilah tersebut dan letakkan istilah teknisnya dalam kurung.

---

## Revisi Bab I

Pertahankan struktur Bab I yang sudah ada. Perbaiki Bab I agar:

1. latar belakang menjelaskan masalah mismatch kompetensi dan kebutuhan penyelarasan kurikulum secara runtut;
2. urgensi bidang RPL/TI dijelaskan dengan bahasa akademik yang tidak terlalu teknis;
3. posisi produk akhir jelas sebagai sistem rekomendasi kompetensi berbasis bukti;
4. kontribusi AI/NLP dijelaskan sebagai alat bantu analisis bukti pasar kerja, bukan pengganti ahli;
5. kebaruan penelitian dirumuskan sebagai kontribusi metodologis dan produk pendidikan, bukan klaim bahwa pipeline sudah selesai;
6. tujuan penelitian tetap selaras dengan R&D: mengidentifikasi kebutuhan, merancang produk, mengembangkan produk, memvalidasi, menguji coba, merevisi, dan mendiseminasikan secara terbatas;
7. manfaat penelitian disesuaikan dengan produk akhir yang diharapkan.

Gunakan kalimat rencana penelitian. Jangan memakai gaya laporan hasil.

---

## Revisi Bab II: Kajian Pustaka dan State of the Art

Bab II harus direvisi agar benar-benar menjadi kajian pustaka. Untuk setiap subbab, lakukan pola berikut:

1. jelaskan konsep utama;
2. bahas teori atau model yang relevan;
3. bahas perkembangan state of the art;
4. tampilkan temuan penelitian terdahulu dengan in-text citation APA style;
5. jelaskan gap yang masih ada;
6. jelaskan bagaimana kajian tersebut dipakai untuk merancang produk sistem rekomendasi kompetensi dalam penelitian ini.

Jangan menulis Bab II seolah-olah pipeline penelitian ini sudah berjalan. Gunakan frasa:

- "Dalam penelitian ini, konsep tersebut akan digunakan untuk..."
- "Kajian ini menjadi dasar bagi rancangan..."
- "Temuan tersebut menunjukkan perlunya..."
- "Berdasarkan kajian tersebut, produk yang dikembangkan akan..."

### Arahan Subbab Bab II

#### 2.1 Competency-Based Curriculum/CBET

Perkuat kajian tentang:

- definisi kompetensi;
- prinsip kurikulum berbasis kompetensi;
- kompetensi sebagai integrasi pengetahuan, keterampilan, sikap, dan performa;
- karakteristik competency statement yang observable, assessable, dan performance-based;
- relevansi CBET untuk pendidikan vokasi;
- bagaimana CBET digunakan sebagai dasar format output rekomendasi.

Gunakan rujukan APA style yang relevan.

#### 2.2 Teori Sinyal Pasar Tenaga Kerja

Perjelas bahwa teori ini mendukung alasan mengapa lowongan kerja dapat diperlakukan sebagai sinyal kebutuhan pasar kerja. Jelaskan batasannya:

- lowongan kerja bukan representasi sempurna semua kebutuhan industri;
- perlu validasi dan audit korpus;
- sinyal pasar kerja harus diterjemahkan menjadi kompetensi melalui kerangka kurikulum.

#### 2.3 Kurikulum Merdeka, KSP/KOSP, Spektrum Keahlian

Jelaskan konteks regulasi Indonesia dan relevansinya:

- fleksibilitas kurikulum;
- kebutuhan data aktual;
- posisi RPL/TI;
- kaitannya dengan pengembangan produk rekomendasi kompetensi.

#### 2.4 KKNI

Jelaskan KKNI sebagai kerangka jenjang kualifikasi. Hindari menjadikan KKNI sekadar output otomatis sistem. Jelaskan bagaimana KKNI akan dipakai sebagai rujukan pemetaan kompleksitas, otonomi, dan tanggung jawab kompetensi.

#### 2.5 Desain Kurikulum Pendidikan Kejuruan

Perkuat DACUM sebagai kerangka utama pengembangan kurikulum vokasi. Jelaskan:

- sejarah dan prinsip DACUM;
- duty, task, skill, knowledge, attitude/tools;
- keterbatasan DACUM tradisional;
- gagasan digital DACUM sebagai adaptasi, bukan pengganti ahli;
- Tyler dan Taba sebagai kerangka pendukung operasional;
- hubungan DACUM dengan CBET.

#### 2.6 Relevansi dan Penyelarasan Kurikulum

Bahas literatur tentang curriculum alignment, skill mismatch, gap antara pendidikan dan industri, dan pentingnya data pasar kerja.

#### 2.7 Sistem Rekomendasi Berbasis AI dalam Pendidikan

Pastikan kajian ini membedakan:

- sistem rekomendasi untuk individu/pembelajar;
- sistem rekomendasi untuk institusi/kurikulum;
- gap penelitian bahwa banyak sistem rekomendasi pendidikan belum berfokus pada rekomendasi kompetensi institusional berbasis evidence package.

#### 2.8 NLP, LLM, dan Ekstraksi Keterampilan Berbasis Bukti

Ubah bagian ini menjadi kajian state of the art. Jelaskan:

- bagaimana NLP digunakan untuk mengekstraksi informasi dari teks lowongan kerja;
- potensi LLM;
- risiko hallucination;
- perlunya grounding, audit bukti, dan validasi ahli;
- posisi LLM sebagai alat bantu, bukan sumber kebenaran.

#### 2.9 Ekstraksi Persyaratan Pendidikan dan Agregasi Permintaan per Jenjang

Jelaskan sebagai rencana analisis untuk memahami jenjang pendidikan/KKNI yang sering muncul dalam lowongan kerja. Tekankan keterbatasan metadata dan perlunya pelaporan coverage.

#### 2.10 NER dan Pelabelan Sekuensial

Jelaskan NER sebagai teknik untuk mengenali entitas dalam teks, misalnya skill, knowledge, dan soft skill. Jelaskan BIO/BIOES dengan sederhana. Jangan terlalu teknis.

#### 2.11 DAPT dan JobBERT

Jelaskan DAPT sebagai adaptasi model bahasa ke domain tertentu. Jelaskan JobBERT/SkillSpan sebagai kajian terdahulu yang relevan, bukan sebagai sesuatu yang sudah pasti digunakan tanpa validasi. Kaitkan dengan kemungkinan pengembangan jalur hybrid.

#### 2.12 LLM sebagai Komponen Ekstraksi dan Perumusan Berbasis Bukti

Bahas literature tentang LLM, manfaat dan risikonya. Tegaskan:

- LLM dapat membantu draft;
- output LLM harus dibatasi evidence;
- output LLM harus divalidasi;
- tidak boleh ada klaim official dari LLM saja.

#### 2.13 SBERT, Deduplikasi, dan Pemetaan Domain

Jelaskan konsep semantic similarity dengan bahasa sederhana. Jelaskan bagaimana konsep ini akan dipakai untuk mengelompokkan/menyamakan istilah kompetensi, tetapi tetap perlu validasi.

#### 2.14 Kerangka Evaluasi Sistem Rekomendasi Evidence-First

Perkuat sebagai kajian evaluasi, bukan laporan pipeline. Bahas state of the art evaluasi:

- data validity;
- extraction evaluation;
- cluster evaluation;
- recommendation evaluation;
- expert validation;
- usability;
- stability/ablation.

Setiap subbagian 2.14 harus menjelaskan konsep, metrik, alasan pemilihan, dan bagaimana akan digunakan dalam penelitian.

#### 2.15 Kerangka Konseptual Digital DACUM Berbasis Evidence Package

Jelaskan kerangka konseptual akhir Bab II yang menyatukan:

- CBET;
- DACUM;
- bukti pasar kerja;
- evidence package;
- validasi ahli;
- rekomendasi kompetensi;
- pengembangan kurikulum.

Pastikan kerangka ini future-state, bukan deskripsi pipeline yang sudah selesai.

---

## Revisi Bab III: Metode Penelitian R&D Borg and Gall

Bab III harus menjadi uraian metodologi yang komprehensif. Jangan hanya memberi daftar tahapan. Untuk setiap tahap Borg and Gall, jelaskan:

1. tujuan tahap;
2. aktivitas yang dilakukan;
3. data yang digunakan;
4. subjek/validator/responden yang terlibat;
5. instrumen yang digunakan;
6. output tahap;
7. teknik analisis;
8. dasar keputusan untuk lanjut/revisi;
9. hubungan dengan tahap berikutnya.

### 3.1 Rancangan Penelitian

Jelaskan bahwa penelitian menggunakan R&D Borg and Gall sebagai kerangka makro, sedangkan pengembangan perangkat lunak menggunakan pendekatan agile sebagai proses teknis internal. Jelaskan hubungan keduanya:

- Borg and Gall mengatur logika penelitian pendidikan;
- agile mengatur iterasi teknis pembuatan produk;
- validasi dan uji coba menentukan revisi produk.

### 3.2 Tahapan Penelitian

Perluas setiap tahap berikut:

1. **Research and Information Collecting**
   - kajian literatur;
   - analisis kebutuhan;
   - studi dokumen;
   - identifikasi korpus;
   - wawancara awal jika direncanakan.

2. **Planning**
   - spesifikasi produk;
   - desain sistem;
   - rancangan evidence package;
   - rancangan instrumen;
   - rencana validasi;
   - kriteria keberhasilan.

3. **Develop Preliminary Form of Product**
   - pengembangan prototipe sistem;
   - pengembangan pipeline rencana;
   - pengembangan dashboard;
   - pengembangan instrumen dan template validasi;
   - dokumentasi produk awal.

4. **Preliminary Field Testing**
   - uji teknis awal;
   - validasi awal instrumen;
   - review keterbacaan evidence package;
   - engineering validation jika menggunakan korpus US/non-Indonesia;
   - jelaskan bahwa tahap ini belum menghasilkan klaim kurikulum Indonesia.

5. **Main Product Revision**
   - revisi berdasarkan hasil uji awal;
   - perbaikan instrumen;
   - perbaikan produk;
   - perbaikan alur validasi.

6. **Main Field Testing**
   - uji pada korpus Indonesia yang valid;
   - validasi ahli;
   - validasi rekomendasi kompetensi;
   - validasi KKNI/SKKNI/Spektrum/okupasi;
   - Delphi/expert panel jika digunakan.

7. **Operational Product Revision**
   - revisi produk berdasarkan uji utama;
   - penyesuaian rekomendasi, status, dashboard, dan coverage analyzer.

8. **Operational Field Testing**
   - uji pengguna terbatas;
   - usability;
   - coverage analysis;
   - FGD/wawancara pengguna;
   - stability/ablation jika direncanakan.

9. **Final Product Revision**
   - penyempurnaan akhir;
   - penyusunan dokumentasi;
   - pass/fail report;
   - batasan klaim;
   - produk final.

10. **Dissemination and Implementation**
   - diseminasi terbatas;
   - panduan penggunaan;
   - policy brief jika layak;
   - publikasi ilmiah;
   - batas penggunaan resmi.

### 3.3 Deskripsi Produk yang Dikembangkan

Jelaskan produk sebagai target akhir yang akan dikembangkan:

- sistem rekomendasi kompetensi;
- dashboard publik;
- dashboard administratif;
- evidence package;
- validation package;
- DACUM/CBET worksheet;
- coverage analyzer;
- export laporan.

Tekankan fungsi produk bagi pengembang kurikulum. Jangan terlalu fokus pada detail kode.

### 3.4 Populasi dan Sampel

Perjelas kelompok yang terlibat:

- ahli kurikulum vokasi;
- praktisi industri RPL/TI;
- ahli NLP/AI;
- ahli evaluasi/metodologi;
- guru/dosen/pengembang kurikulum;
- pengguna dashboard;
- dokumen/korpus sebagai objek nonmanusia.

Jika jumlah belum pasti, tuliskan sebagai rencana dengan kriteria pemilihan dan rentang yang realistis.

### 3.5 Teknik dan Instrumen Pengumpulan Data

Perkuat bagian instrumen. Jelaskan:

- instrumen studi dokumen/korpus;
- pedoman wawancara;
- instrumen gold labeling;
- lembar validasi ahli;
- worksheet DACUM/CBET;
- instrumen Delphi;
- SUS/usability;
- instrumen coverage analyzer;
- log revisi dan pass/fail report.

Setiap instrumen harus dijelaskan fungsi, responden, bentuk data, dan penggunaannya.

### 3.6 Teknik Analisis Data

Perjelas setiap teknik analisis:

- validitas isi: Aiken's V/CVI/CVR;
- reliabilitas antar-penilai: Cohen's/Fleiss' Kappa;
- evaluasi ekstraksi: precision, recall, F1;
- evaluasi coverage analyzer: precision, recall, F1;
- usability: SUS dan statistik deskriptif;
- validasi ahli: rerata skor, kategori kelayakan, komentar kualitatif;
- Delphi: tingkat konsensus;
- stability/ablation: Jaccard/top-k/rank correlation jika digunakan;
- analisis kualitatif: reduksi data, kategorisasi, tema revisi.

Tuliskan cara analisis secara operasional, bukan hanya menyebut nama metrik.

---

## Gaya Sitasi dan Daftar Pustaka

Gunakan in-text citation APA style, misalnya:

- (Norton, 1997)
- (Mulder et al., 2007)
- (Anderson & Krathwohl, 2001)
- (Reimers & Gurevych, 2019)
- (Gururangan et al., 2020)

Jika menambahkan rujukan baru, pastikan:

1. relevan dengan subtopik;
2. tidak fiktif;
3. konsisten dengan daftar pustaka;
4. ditambahkan ke daftar pustaka dalam format APA.

Jika tidak yakin terhadap detail bibliografi, beri tanda `[PERLU VERIFIKASI REFERENSI]` daripada mengarang.

---

## Output Revisi yang Diminta

Berikan hasil dalam format berikut:

1. **Ringkasan masalah utama proposal saat ini** berdasarkan pembacaan file.
2. **Daftar prinsip revisi yang diterapkan**.
3. **Versi revisi proposal lengkap** dengan struktur yang sama seperti proposal asli.
4. **Daftar istilah teknis yang sudah dijelaskan** beserta letak penjelasannya.
5. **Daftar bagian Bab II yang diubah menjadi kajian pustaka/state of the art**.
6. **Daftar bagian Bab III yang diperkuat secara metodologis**.
7. **Daftar rujukan baru atau rujukan yang perlu diverifikasi**.
8. **Catatan perubahan penting** tanpa mengubah struktur proposal.

Jika output terlalu panjang, lanjutkan per bab. Jangan meringkas Bab II dan Bab III secara berlebihan. Prioritaskan kualitas akademik, kejelasan istilah, dan kelengkapan metodologi.

---

## Checklist Akhir

Sebelum mengakhiri revisi, pastikan:

1. struktur proposal tetap sama;
2. proposal ditulis sebagai rencana penelitian, bukan laporan hasil pipeline;
3. istilah teknis dijelaskan saat pertama muncul;
4. Bab II berisi kajian pustaka dan state of the art;
5. Bab II memiliki in-text citation APA style;
6. setiap subbab Bab II menjelaskan relevansi kajian bagi rancangan produk;
7. Bab III menjelaskan pengembangan, uji coba, revisi, dan diseminasi sesuai Borg and Gall;
8. agile hanya dijelaskan sebagai proses teknis internal, bukan pengganti Borg and Gall;
9. AI/LLM tidak diposisikan sebagai pengambil keputusan kurikulum;
10. official-ready diposisikan sebagai target yang hanya dapat dicapai setelah validasi lengkap;
11. tidak ada klaim bahwa produk sudah selesai atau sudah valid sebelum penelitian dilakukan;
12. bahasa dapat dipahami pembaca pendidikan vokasi yang tidak berlatar belakang teknis komputasi.
