# Audit Sumber Data & Legalitas Akuisisi Korpus OJP (Workstream A1)

**Status:** v1.7 — keputusan K1–K12 tercatat (K1–K5: 29 Okt 2026; K6–K11: 2 Nov 2026; K12 re-probe TechInAsia: 8 Nov 2026, lihat §5); tanda tangan formal promotor menyusul
**Tanggal:** 20 Oktober 2026 (rev. 8 November 2026)
**Rujukan proposal:** §3.4.1 (akuisisi korpus), Tabel 3.9 (inklusi/eksklusi), Tabel 3.10 (skema metadata), §3.7 (etika penelitian)
**Penyusun:** Listyanti Dewi Astuti (dengan asistensi teknis)

---

## 1. Ringkasan eksekutif

Penelitian ini membutuhkan korpus iklan lowongan kerja daring (*online job postings*, OJP) tiga strata:
Indonesia 10.000–20.000 unik, ASEAN non-Indonesia ≥5.000, OECD ≥5.000. Dokumen ini mengaudit
setiap sumber data kandidat dari sisi (a) metode akses, (b) posisi syarat layanan (*Terms of
Service*, ToS), (c) kelengkapan metadata terhadap Tabel 3.10, (d) status empiris per Oktober 2026,
dan (e) keputusan pemakaian. Keputusan akhir memerlukan persetujuan promotor karena menyangkut
bagian etika penelitian (§3.7).

**Rekomendasi keputusan (ringkas):**

| Sumber | Keputusan yang diusulkan |
|---|---|
| Indeed (via pustaka JobSpy) | **PAKAI, dengan catatan metode** — kanal utama ketiga strata (K2 ✓) |
| Google Jobs (via JobSpy) | **PARKIR** — rusak secara teknis per Okt 2026; kontribusi historis ≈0,2% |
| LinkedIn (via JobSpy) | **DIPERTAHANKAN dalam desain akuisisi** (keputusan K1, 2026-10-29 — menolak usulan drop); dengan pagar operasional §3.3 dan risiko ToS terdokumentasi |
| ZipRecruiter (via JobSpy) | **PARKIR** — cakupan AS saja; Indeed sudah memadai untuk strata OECD |
| Jobstreet by SEEK (ID/MY/SG/PH) | **PAKAI** — lolos evaluasi K3 (2026-10-29): metadata terlengkap dari semua kanal, 34% berbahasa Indonesia, 1% tumpang-tindih (lihat §3.5) |
| Glints (ID/SEA) | **EVALUASI** — kandidat penambah volume Indonesia |
| Kalibrr (ID/PH) | **EVALUASI** — kandidat penambah volume Indonesia |
| Adzuna API (OECD) | **AJUKAN AKSES** — API resmi, legal paling bersih untuk strata OECD |
| USAJOBS API (AS, sektor publik) | **OPSIONAL** — API resmi pemerintah AS, legal bersih, cakupan sempit |
| Arbeitnow API (Eropa) | **OPSIONAL** — API publik gratis, volume kecil |
| Dataset berlisensi (mis. Lightcast) | **TIDAK** — di luar anggaran; dicatat sebagai keterbatasan |

---

## 2. Prinsip legal–etis yang dipegang

1. **Hanya data publik.** Semua akuisisi terbatas pada halaman iklan lowongan yang dapat dilihat
   publik tanpa login. Tidak ada penerobosan autentikasi, paywall, atau CAPTCHA.
2. **Kesantunan teknis (politeness).** *(Dikoreksi 2026-11-15 agar sesuai praktik nyata —
   rumusan sebelumnya menyebut "jeda ≥5 detik + acak", padahal adapter tidak pernah berjalan
   sebesar itu; klaim yang tidak sesuai praktik dikoreksi, bukan dipertahankan.)*
   Pengendali beban yang **benar-benar dipakai** adalah:
   (a) **jeda acak antar-permintaan 1–4 detik** per adapter — `sleep_seconds` 2–5 detik antar
   kueri pada config batch, dan jeda acak di dalam adapter: Glints 1,2–2,5 s (pencarian) /
   1,5–3,0 s (detail), Kitalulus 1,0–2,0 s / 1,2–2,6 s, Jobstreet sejenis;
   (b) **pagu volume per sesi** (mis. Jobstreet ≤25 istilah, lihat §3.5) — ini pengendali beban
   yang paling menentukan, bukan panjang jeda;
   (c) **jeda antar-sesi ≥24 jam** per platform;
   (d) **circuit breaker** yang menghentikan sesi saat terdeteksi rentetan 403/429.
   Akuisisi bersifat berkala (bulanan/dwibulanan), bukan pengambilan terus-menerus. Volume
   total rendah: satu sesi Jobstreet ±1.000–2.000 pengambilan detail, sekali dalam ≥24 jam.
3. **Minimisasi data pribadi.** OJP pada dasarnya data korporat, namun dapat memuat data pribadi
   (nama/email narahubung). Mitigasi: kolom `emails` tidak pernah dipakai di lapisan turunan;
   identitas pemberi kerja disimpan sebagai `employer_hash` pada lapisan analisis; lapisan mentah
   (*raw immutable*) berakses terbatas (peneliti saja) dan tidak didistribusikan ulang. Selaras
   dengan UU No. 27/2022 (Pelindungan Data Pribadi) dan, untuk iklan kawasan Eropa, prinsip GDPR.
4. **Tujuan riset non-komersial** dan tidak mereproduksi basis data sumber secara utuh — korpus
   dipakai untuk analisis agregat kebutuhan kompetensi; teks iklan tidak dipublikasikan ulang.
5. **Dokumentasi jujur.** Metode, versi pustaka, parameter kueri, dan keterbatasan setiap kanal
   dicatat dalam `acquisition_method/version` (Tabel 3.10) dan manifest per batch — termasuk fakta
   bahwa papan lowongan hanya menyajikan iklan yang masih tayang sehingga korpus 36 bulan harus
   diakumulasi maju (lihat `docs/IMPLEMENTATION_PLAN.md` §1).
6. **ToS dihormati secara proporsional.** Sumber yang ToS-nya melarang scraping secara eksplisit
   dan menegakkannya secara hukum (LinkedIn) tidak dipakai. Sumber yang menyajikan data publik
   terbuka diakses dengan kesantunan teknis dan didokumentasikan; bila tersedia API resmi, API
   diprioritaskan.

---

## 3. Matriks audit per sumber

### 3.1 Indeed — via pustaka `python-jobspy`

- **Metode akses:** halaman/endpoint daftar lowongan publik (tanpa login) melalui pustaka
  open-source JobSpy v1.4.0.
- **Posisi ToS:** ToS Indeed secara umum membatasi akses otomatis; namun data yang diambil adalah
  iklan publik, volume rendah, berkala, non-komersial, untuk riset — posisi yang lazim
  dipertahankan dalam riset analisis pasar kerja (banyak studi OJP akademik memakai sumber serupa).
  Didokumentasikan jujur; bukan API resmi.
- **Metadata vs Tabel 3.10:** judul, perusahaan, lokasi, `date_posted`, deskripsi penuh, URL unik,
  gaji (sebagian), tipe pekerjaan. Lengkap untuk kebutuhan skema; `first_seen/last_seen` dikelola
  oleh harness lintas batch.
- **Status empiris (uji 2026-10-20):** BERFUNGSI di ketiga strata — Indonesia 13 iklan/kueri uji,
  Malaysia 20/20, AS 20/20; deskripsi lengkap (median 2–3,4 ribu karakter); 100% bertanggal.
  Kanal ini menyumbang 99,8% batch-001 (113 ribu iklan, Maret 2026).
- **Keputusan diusulkan:** **PAKAI** sebagai kanal utama ketiga strata, dengan catatan metode di §3.7 proposal.

### 3.2 Google Jobs — via JobSpy

- **Metode akses:** agregator hasil pencarian lowongan Google (halaman publik).
- **Status empiris:** RUSAK pada JobSpy 1.4.0 (“no job data returned” pada ketiga strata, uji
  2026-10-20). Kontribusi historis hanya 190 dari 113 ribu baris batch-001.
- **Keputusan diusulkan:** **PARKIR** (tidak dipakai; dicek ulang bila JobSpy memperbaiki).

### 3.3 LinkedIn — via JobSpy

- **Posisi ToS:** User Agreement LinkedIn **melarang eksplisit** scraping/akses otomatis; LinkedIn
  aktif menegakkannya secara hukum. Preseden *hiQ v. LinkedIn* berakhir dengan larangan permanen
  bagi hiQ (2022) — risiko hukum dan etika paling tinggi di antara semua kandidat.
- **Status empiris:** belum diuji ulang pada Okt 2026. Catatan penting dari batch-000: seluruh
  1.832 baris LinkedIn Indonesia **tanpa deskripsi** (JobSpy tidak mengambil deskripsi LinkedIn
  kecuali `linkedin_fetch_description=True`) sehingga gagal syarat metadata Tabel 3.9 — scraping
  LinkedIn tanpa opsi tersebut tidak menghasilkan data yang dapat dipakai.
- **KEPUTUSAN (K1, 2026-10-29): DIPERTAHANKAN dalam desain akuisisi** — usulan drop DITOLAK oleh
  peneliti. Pagar operasional yang diadopsi: (a) volume rendah dan berkala (politeness §2),
  (b) `linkedin_fetch_description` diaktifkan agar data memenuhi Tabel 3.10, (c) risiko ToS/hukum
  tetap terdokumentasi jujur di §3.7 proposal, (d) kanal ditinjau ulang bila terjadi pemblokiran
  atau keberatan dari platform.
- **Catatan data historis:** batch-000 (arsip Nov–Des 2025, ditemukan 2026-10-20) memuat ±10,5 ribu
  baris LinkedIn dan ±5,6 ribu baris Glassdoor yang telanjur terkumpul sebelum rekomendasi ini.
  Lapisan mentah menyimpannya apa adanya (prinsip kekal); apakah baris tersebut ikut masuk
  N_eligible diputuskan di lapisan tata kelola → keputusan K5.

### 3.4 ZipRecruiter — via JobSpy

- Cakupan efektif AS/Kanada saja; strata OECD sudah terpenuhi lewat Indeed (42 ribu iklan AS pada
  batch-001). **Keputusan diusulkan: PARKIR** (dipakai hanya bila Indeed AS terganggu).

### 3.5 Jobstreet by SEEK (Indonesia, Malaysia, Singapura, Filipina)

- **Metode akses:** halaman pencarian publik; terdapat endpoint JSON publik yang dipakai situsnya
  sendiri. Perlu adapter khusus (belum ada di JobSpy).
- **Posisi ToS:** ToS SEEK membatasi akses otomatis; data publik, prinsip yang sama dengan Indeed.
- **Nilai:** papan lowongan terbesar untuk pasar kerja formal Indonesia/ASEAN — kandidat terkuat
  untuk menutup kekurangan strata Indonesia.
- **HASIL EVALUASI (K3, batch-002d, 2026-10-29):** **LOLOS — PAKAI.** Purwarupa adapter
  (`acquisition/jobstreet_adapter.py`, API pencarian v5 publik + status halaman detail) pada
  12 istilah × 150 hasil: 826 lowongan unik; **100% berdeskripsi penuh** (median 1.858 karakter),
  100% bertanggal (timestamp ISO penuh), **100% memuat `expiresAt`** (bukti langsung `last_seen`
  Tabel 3.10), 93% nama perusahaan, 33% gaji; **34% berbahasa Indonesia** (280 — bandingkan 294
  pada SELURUH korpus eligible sebelumnya); **hanya 1% tumpang-tindih konten** dengan korpus
  Indonesia yang ada; **62% lolos aturan deterministik Tabel 3.9** (vs ±32% Indeed ID) dan yang
  tersisih memang di luar lingkup (sales/civil/MEP engineer). Kelengkapan metadata TERBAIK dari
  semua kanal yang diaudit. Lanjut: akuisisi skala penuh Indonesia, lalu adaptasi MY/SG/PH.
- **Pagar operasional (empiris, batch-002e 2026-10-29):** WAF SEEK memblokir (HTTP 403) setelah
  ±33 istilah berat / ±2.000 pengambilan detail dalam satu sesi; blokir berlangsung berjam-jam.
  Anggaran per sesi: **≤25 istilah atau ≤1.500 detail**, jeda antar-sesi ≥24 jam; harness kini
  menghentikan diri (circuit breaker) saat mendeteksi rentetan 403/429.

### 3.6 Glints (Indonesia/SEA) dan 3.7 Kalibrr (Indonesia/Filipina)

- Platform lowongan teknologi populer di Indonesia; halaman publik dengan endpoint JSON.
- **HASIL EVALUASI (K3, 2026-10-30):** keduanya **PAKAI** — adapter dibangun
  (`acquisition/glints_adapter.py`, `acquisition/kalibrr_adapter.py`), deskripsi penuh 100%.
  - **Glints (batch-002h):** 835 lowongan unik; platform fokus-teknologi → **66% lolos Tabel 3.9**
    (554 eligible). State Next.js tertanam + detail per lowongan (deskripsi Draft.js); memuat
    `educationLevel`, rentang pengalaman, dan daftar `skills` pra-ekstraksi. Risiko rate-limit sama
    dengan Jobstreet → circuit breaker + anggaran sesi.
  - **Kalibrr (batch-002g):** 1.402 lowongan unik, deskripsi penuh (median 637 karakter), tetapi
    **hanya ~104 lolos Tabel 3.9 (≈7%)** — Kalibrr adalah papan lowongan UMUM; pencarian "programmer"
    mengembalikan banyak lowongan non-RPL (bartender, auditor, business development, teknisi jaringan)
    yang benar disaring. Metadata kaya (`application_end_date` → last_seen, `education_level`), API
    cepat tanpa detail-fetch. Nilai per lowongan rendah tetapi sah; dipakai untuk melengkapi.
  - **Catatan keterbatasan:** sebagian kecil peran QA/testing berdeskripsi pendek pada Kalibrr
    mungkin tersaring sebagai false-negative (ambang 2 sinyal deskripsi); diterima sebagai
    konservatif (lebih baik under-count) dan ditinjau saat validasi gold-standard.

### 3.8 API resmi (legal paling bersih)

| API | Cakupan | Catatan |
|---|---|---|
| **Adzuna** | OECD + SG (BUKAN Indonesia) | API resmi, kuota gratis; **deskripsi dipotong 500 karakter** (lihat evaluasi) |
| **USAJOBS** | AS (pemerintah federal) | API resmi; relevansi RPL sempit tapi bersih |
| **Arbeitnow** | Eropa | API publik gratis; volume kecil |

API resmi memperkuat posisi etis korpus OECD dan menjadi pembanding independen terhadap Indeed.

**HASIL EVALUASI Adzuna (K4, 2026-10-30):** kunci API berfungsi (HTTP 200). Cakupan baik untuk
strata pembanding — US 156.209, GB 10.949, CA 10.643, AU 4.174, NZ 246 (OECD); SG 5.223 (ASEAN);
DE/FR/NL juga tersedia. **Indonesia TIDAK dicakup (HTTP 404)** sehingga tidak membantu strata utama.
**Keterbatasan kritis: deskripsi dipotong tepat 500 karakter** (diakhiri elipsis) — API gratis
tidak menyediakan teks lowongan penuh. Konsekuensi: field `description_raw` Tabel 3.10 menjadi
parsial, sehingga **tidak layak untuk ekstraksi span lima-tag** (persyaratan/skill biasanya berada
di luar 500 karakter pertama). **Keputusan: PAKAI TERBATAS sebagai sumber pembanding/validasi
independen** (hitungan permintaan, audit representativitas lintas-sumber, penguatan posisi etis
korpus OECD) — **bukan** sumber pembangun korpus ekstraksi. Rekaman Adzuna, bila diakuisisi, ditandai
`description_truncated=true` dan dipisahkan dari korpus ekstraksi utama. Strata OECD sudah >11× target
dari Indeed, sehingga nilai tambah korpus dari Adzuna rendah; nilainya murni pada legalitas/validasi.

### 3.9 Dataset berlisensi (Lightcast/Burning Glass, Indeed Hiring Lab)

- Standar emas industri untuk OJP historis (bisa menjawab kebutuhan 36 bulan retrospektif), namun
  biaya lisensi jauh di luar anggaran penelitian. **Keputusan: TIDAK DIPAKAI**; dicatat di bagian
  keterbatasan (§1.4 proposal) sebagai alasan desain akumulasi maju.

---

## 4. Implikasi desain yang sudah diadopsi harness

1. **Lapisan mentah kekal (raw immutable):** teks disimpan apa adanya, tanpa pembersihan; setiap
   batch bersifat *append-only* dengan manifest (`DATA/corpus_raw/batch-XXX/`). (Catatan: batch-001
   warisan Maret 2026 telanjur dibersihkan sebelum disimpan — dicatat sebagai deviasi pada manifest.)
2. **Skema Tabel 3.10:** `source_record_id` (hash URL), `acquisition_timestamp`, `acquisition_method/
   version`, `posting_date`, `first_seen/last_seen` (dari registry lintas batch), negara/strata,
   `employer_hash`, `description_raw`, provenance kueri (`query_term/query_family`).
3. **Akumulasi maju:** batch-001 = arsip Maret 2026; batch-002 = Oktober 2026; selanjutnya bulanan
   (Indonesia) / dwibulanan (ASEAN, OECD). Klaim temporal 36 bulan bersifat kondisional/eksploratif
   sesuai §1.4 proposal.
4. **Aturan sanitas tanggal:** `posting_date` < 2021 atau > tanggal akuisisi ditandai tidak valid
   pada lapisan tata kelola (ditemukan nilai 1983/1990 pada batch-001).

---

## 5. Keputusan yang dimintakan persetujuan promotor

| # | Keputusan | Usulan | **Keputusan (2026-10-29)** |
|---|---|---|---|
| K1 | Menghapus LinkedIn dari seluruh desain akuisisi | Setuju (drop) | **DITOLAK — LinkedIn dipertahankan** dengan pagar operasional §3.3 |
| K2 | Melanjutkan akuisisi Indeed via JobSpy dengan prinsip §2 dan dokumentasi jujur di §3.7 proposal | Setuju (pakai) | **SETUJU** |
| K3 | Mengembangkan adapter Jobstreet → Glints → Kalibrr (urutan prioritas) untuk menutup target Indonesia | Setuju (evaluasi bertahap) | **SETUJU** |
| K4 | Mengajukan akses API resmi Adzuna (+ opsional USAJOBS/Arbeitnow) untuk strata OECD | Setuju (ajukan) | **SETUJU** |
| K5 | Perlakuan baris LinkedIn/Glassdoor historis pada batch-000 (Nov–Des 2025): ikut N_eligible dengan dokumentasi, atau dikecualikan di lapisan tata kelola | Usulan: ikut, dengan dokumentasi platform per baris | **SETUJU — ikut, dengan dokumentasi platform per baris** |
| K6 | Membangun adapter **Dealls** (dealls.com, ID) sebagai kanal baru | Di-*greenlight* pengguna 2026-11-02 ("bangun ketiganya") | **PARKIR (2026-11-02)** — API pencarian `/v1/explore-job/job` terbuka, tetapi halaman detail (sumber deskripsi) dijaga anti-bot (403 → pengalih `sh.dealls.com`); deskripsi wajib Tabel 3.10 tidak dapat diambil tanpa *evasion* (dilarang, dan tidak dapat dipertanggungjawabkan di §3.7). Pencarian juga kabur (mirip Kalibrr). Dicek ulang bila ada jalur deskripsi yang sah. |
| K7 | Membangun adapter **Arbeitnow API** untuk strata OECD/EU | Di-*greenlight* pengguna 2026-11-02 | **TIDAK DIPAKAI (2026-11-02)** — *feed* tanpa pencarian kata kunci (`?search=` diabaikan), lokasi teks-bebas tanpa kode negara (sulit atribusi strata), lintas banyak negara termasuk di luar himpunan pantauan (mis. CH), dan OECD sudah >10× target (59k). Biaya integrasi + dilusi domain tidak sepadan. Dicek ulang hanya bila celah OECD spesifik perlu ditutup; bila dipakai, berperan sebagai sumber pelengkap, bukan tulang punggung strata. |
| K8 | Membangun adapter **Kitalulus** (ID) via JobPosting JSON-LD | Di-*greenlight* pengguna 2026-11-02 | **PAKAI (2026-11-02 — promotor OK; persetujuan promotor dikonfirmasi peneliti dan dicatat 2026-11-19)** — JSON-LD schema.org pada halaman detail (data terstruktur publik untuk SEO, dibaca apa adanya, **tanpa evasion**); payload Tabel 3.10 lengkap (deskripsi + pendidikan/pengalaman/gaji). **Catatan etika:** robots `User-Agent: *` mengizinkan `/lowongan` (hanya `/auth/`, `/my/` dilarang — tidak disentuh); ada blok khusus `GPTBot: Disallow /` — kami bukan GPTBot, penggunaan untuk analisis riset (bukan redistribusi pelatihan model), volume rendah, didokumentasikan transparan. **Catatan kualitas:** listing promosi "Misi Seru - KitaLulus" bukan lowongan nyata → ditolak di tahap konten/domain Tabel 3.9; yield riil rendah, nilainya pada katalog baru + metadata kaya. |
| K9 | Memperluas kerangka sampel strata **OECD** dari "negara berbahasa Inggris" (US/GB/CA/AU/NZ/IE) menjadi **ekonomi utama OECD** dengan menambahkan Jepang, Jerman, Prancis, Korea Selatan, Italia, Spanyol, Belanda, Meksiko ke korpus **UTAMA** (bukan sekadar *benchmark*); deskripsi non-Inggris **diterima**, diterjemahkan kemudian setelah korpus mentah lengkap | Diminta pengguna 2026-11-02 | **SETUJU (2026-11-02 — promotor OK)** — `batch-002l` (Indeed/JobSpy, 17 istilah inti, rw 250) + *top-up* NZ (15 istilah segar; NZ 985 → **1.080** lolos gerbang 1k). Mengubah kerangka terdokumentasi "IDN + 7-negara Inggris" (IMPLEMENTATION_PLAN 3.4.1 / KAJIAN_AKADEMIK §"Korpus saat ini berbahasa Inggris") → klaim "OECD" kini = **ekonomi utama OECD, bukan seluruh 38 anggota** (didokumentasikan di tabel representativitas). Hasil: OECD 59.069 → **67.454** eligible. Lolos gerbang 1k: DE 2.979 / NL 2.426 / FR 2.200 / ES 1.062 / MX 1.001. **Belum** lolos: JP 267 / IT 266 / KR 87 — data mentah tertangkap (JP 1.515 / IT 2.709 / KR 618) tetapi iklan berbahasa lokal gugur di filter domain Tabel 3.9 (berbasis kata kunci Inggris/Indonesia); **diselamatkan oleh langkah terjemah-lalu-saring-ulang** yang direncanakan (DE/NL/FR/ES lolos karena iklan tekniknya ditulis dalam bahasa Inggris). |
| K10 | Membangun adapter **Kemnaker Karirhub** (ID) sebagai sumber **resmi pemerintah** — SISNAKER, Kementerian Ketenagakerjaan RI | Diminta pengguna 2026-11-02 | **PAKAI (2026-11-02 — promotor OK)** — API publik tanpa autentikasi: LIST `GET api.kemnaker.go.id/karirhub/vacancy/v2/published-industrial-vacancies?keyword=&page=&limit=` (paginasi Laravel) → DETAIL `/{id}` memberi payload Tabel 3.10 lengkap (`description`+`qualification`, `min_education`, `min_year_experience`, gaji, `published_at`/`expires_at`, `employer`, `region`, + array **`skills` terstruktur** sebagai bonus). **Etika (terbersih sejauh ini):** host API `api.kemnaker.go.id` tak punya robots.txt (404 → tidak dibatasi); robots situs publik berjudul *"SEO & AI AGENT READINESS"* dan **mengizinkan ClaudeBot/GPTBot eksplisit** (`Allow: /`; hanya /admin,/dashboard,/api,/auth pada host itu yang dilarang — tidak disentuh). Sumber resmi pemerintah, hanya-baca, tanpa autentikasi, **tanpa evasion**, dibatasi laju + volume rendah. **Catatan teknis:** host API labil di bawah beban (500/koneksi putus sesekali) → adapter me-*retry* transient 5xx/koneksi dengan *backoff*, PER_PAGE=10 konservatif. **Catatan volume:** portal tenaga kerja **umum** (~21 rb lowongan, mayoritas non-RPL: perawat/laundry/sales/operator); irisan RPL kecil & saling tumpang-tindih (developer 117 / software 128 / perangkat lunak 107 / android 40 / IT support 40) → perkiraan net-new eligible ID ≈ **150–350**. Nilai utama = **provenans resmi + bahasa Indonesia + metadata kaya** (termasuk `skills` terstruktur, calon sinyal weak-supervision bagi model), bukan volume mentah. `acquisition/kemnaker_adapter.py` (kemnaker-0.1.0); batch-002m. |
| K11 | Recon kumpulan papan lowongan ID tambahan (Karir.com, TopKarir, TechInAsia, EKRUT, Jobs.id, Urbanhire) untuk menambah korpus Indonesia | Diminta pengguna 2026-11-02 ("more ID boards") | **SEMUA PARKIR/DROP (2026-11-02) — tidak ada PAKAI baru.** **EKRUT**: lowongan di balik login talenta (`/talent/job-opportunities`); sitemap publik hanya halaman auth/marketing → tak terakses tanpa akun (buat akun palsu dilarang). **Karir.com**: robots `Disallow: /jobs/` + `/karir/` (jalur lowongan) → konten lowongan dilarang; tak ada rute lowongan yang diizinkan. **TechInAsia**: terbuka secara teknis (robots izinkan `/jobs`, detail punya JobPosting JSON-LD, enumerable via sitemap `/sitemaps/jobs/`) TETAPI total papan hanya **~360 lowongan se-ASEAN** & condong Singapura (16× SG vs 1× Jakarta pada listing default); irisan ID-RPL ~puluhan, sitemap tak memuat negara (harus fetch tiap detail untuk filter) → ROI rendah. PARKIR (dicek ulang hanya bila fokus ASEAN/SG diperlukan; ASEAN sudah 2,5× target). **TopKarir/Jobs.id/Urbanhire**: tak terjangkau/mati (404 / SSL mati / Cloudflare 530). **Temuan korpus:** ekosistem papan lowongan publik ID yang mudah diakses sudah **habis tergarap** — kanal yang dipakai (Indeed/Glints/Kalibrr/Jobstreet/Kitalulus/LinkedIn/Kemnaker) mencakup sumber publik layak; sisanya auth-walled / robots-blocked / mati / condong asing. Pertumbuhan ID selanjutnya = kadens bulanan (akumulasi maju) + kedalaman Jobstreet, bukan papan baru. |
| K12 | Membangun adapter **TechInAsia** (revisit K11 — pengguna minta "build TechInAsia now") sebagai kanal ID bersih | Di-*greenlight* pengguna 2026-11-08 | **PARKIR LAGI (2026-11-08)** — re-probe langsung menunjukkan situs berubah sejak K11. **Izin: terbersih dari semua kandidat** — robots.txt memuat grup AI-crawler eksplisit yang MENGIZINKAN ClaudeBot/Claude-User/anthropic-ai/GPTBot dll. (`Allow: /` minus /login,/core,/vendor,/register,*/feed — tak disentuh); `/jobs` + `/sitemaps/jobs/sitemap.xml` diizinkan; indeks peta-situs pekerjaan ada (171 pecahan). **Namun akuisisi bersih tidak lagi mungkin:** situs kini SPA sisi-klien penuh (bundel `static.techinasia.com/assets/v5/app.*.js`). Setiap halaman detail `/jobs/<uuid>` mengembalikan shell 337 KB identik dengan **0 kemunculan `JobPosting`/`datePosted`/`jobLocation`**; satu-satunya JSON-LD = blok `Organization`. Klaim K11 "detail punya JobPosting JSON-LD" **sudah tidak berlaku** (migrasi ke SPA v5). Data lowongan datang lewat gateway internal `/gateway-api-express/techinasia`; jalurnya tidak ada di HTML statis dan bundel memuat **kunci reCaptcha** → endpoint terlindungi. Menjangkaunya menuntut balik-rekayasa bundel + meniru token/reCaptcha browser = *evasion* (dilarang §3.7) — kelas yang sama dengan **Dealls (K6)**. Satu-satunya jalur bersih tersisa = *headless browser* (Playwright) merender tiap detail — dependensi berat yang tidak sepadan untuk **irisan ID-RPL ~20–40 rekaman** (peta-situs ~171×~2 ≈ ~340 total se-ASEAN, condong Singapura). **Keputusan: PARKIR, tanpa adapter.** Probe bersifat **hanya-baca & volume rendah** (robots.txt, indeks peta-situs, 1 peta-anak, 2 shell detail, 1 app.js, beberapa cek endpoint) — **tidak ada data lowongan yang diakuisisi atau disimpan**. Dicek ulang hanya bila TIA memulihkan data terstruktur sisi-server (JSON-LD/JSON terbuka) atau merilis API lowongan resmi. **Menegaskan kembali K11:** ekosistem papan ID publik mudah-akses habis tergarap; pertumbuhan ID = kadens bulanan + kedalaman Jobstreet. |

**Persetujuan promotor:**

- Nama / tanda tangan / tanggal: ______________________________
- Catatan revisi: ______________________________

---

## Riwayat perubahan

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0 (draf) | 2026-10-20 | Draf awal: matriks 11 sumber, prinsip legal–etis, 4 butir keputusan |
| 1.1 (draf) | 2026-10-20 | +K5: perlakuan baris LinkedIn/Glassdoor historis setelah penemuan arsip batch-000 (Nov–Des 2025) |
| 1.2 | 2026-10-29 | Keputusan K1–K5 tercatat: K1 ditolak (LinkedIn dipertahankan + pagar operasional), K2–K5 disetujui |
| 1.3 | 2026-11-02 | +K6–K8 sumber baru (hasil recon): Dealls PARKIR (detail anti-bot), Arbeitnow TIDAK DIPAKAI (misfit stratifikasi, OECD sudah terpenuhi), Kitalulus PAKAI (JSON-LD JobPosting; catatan etika GPTBot + spam "Misi Seru"). Adapter + perbaikan bug `-I` pada probe/runbook Jobstreet. |
| 1.4 | 2026-11-02 | +K9: perluasan kerangka OECD ke ekonomi utama non-Inggris (JP/DE/FR/KR/IT/ES/NL/MX) + *top-up* NZ via `batch-002l`; deskripsi non-Inggris diterima untuk diterjemahkan kemudian; promotor OK. OECD 59.069 → 67.454 eligible; 5 ekonomi baru lolos gerbang 1k, 3 (JP/IT/KR) menunggu terjemahan. Juga tercatat (temuan 2026-11-02): JobSpy `hours_old` tidak dapat *backfill* historis (menyaring indeks aktif, bukan iklan kedaluwarsa); 3 CSV `DATA/preprocessing` = sumber batch-000 (sudah diingest). |
| 1.5 | 2026-11-02 | +K10: adapter **Kemnaker Karirhub** (sumber resmi pemerintah SISNAKER) via API publik `published-industrial-vacancies`; robots terbersih (host API tanpa robots.txt; situs publik "AI AGENT READINESS"); metadata Tabel 3.10 lengkap + `skills` terstruktur; volume RPL modest (~150–350 eligible); adapter me-*retry* host API yang labil. Promotor OK. |
| 1.6 | 2026-11-02 | +K11: recon papan ID tambahan (Karir.com/TopKarir/TechInAsia/EKRUT/Jobs.id/Urbanhire) → **semua PARKIR/DROP** (auth-walled / robots-blocked / mati / TechInAsia ~360 total condong SG). Ekosistem papan lowongan publik ID dinyatakan **habis tergarap**; pertumbuhan ID lewat kadens bulanan + kedalaman Jobstreet. |
| 1.7 | 2026-11-08 | +K12: re-probe **TechInAsia** (revisit K11; pengguna green-light "build now") → **PARKIR LAGI**. Izin robots terbersih (grup AI-crawler eksplisit mengizinkan ClaudeBot/anthropic-ai; `/jobs` + sitemap OK) TETAPI situs kini SPA v5 sisi-klien: detail `/jobs/<uuid>` = shell 337 KB tanpa JobPosting JSON-LD (klaim K11 gugur); data via gateway internal + reCaptcha → terlindungi (kelas Dealls K6). Jalur bersih hanya *headless-browser*, tidak sepadan untuk ~20–40 rekaman ID-RPL. Probe hanya-baca, tanpa akuisisi. |
| 1.8 | 2026-11-19 | K8 (Kitalulus): persetujuan promotor dicatat. Baris K8 sebelumnya hanya mencatat *greenlight* peneliti dan audit etika; peneliti mengonfirmasi pada 2026-11-19 bahwa promotor menyetujui sumber ini. Tidak ada perubahan pada cara akuisisi. |
