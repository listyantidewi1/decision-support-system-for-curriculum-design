# Audit Sumber Data & Legalitas Akuisisi Korpus OJP (Workstream A1)

**Status:** v1.2 — keputusan K1–K5 tercatat 29 Oktober 2026 (lihat §5); tanda tangan formal promotor menyusul
**Tanggal:** 20 Oktober 2026 (rev. 29 Oktober 2026)
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
2. **Kesantunan teknis (politeness).** Jeda antar-permintaan (≥5 detik + acak), volume per kueri
   dibatasi, akuisisi berkala (bulanan/dwibulanan) — bukan pengambilan terus-menerus; tidak
   membebani infrastruktur sumber.
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
- **Keputusan diusulkan:** **EVALUASI** dengan prosedur yang sama dengan Jobstreet (purwarupa →
  audit metadata → keputusan). Prioritas setelah Jobstreet.

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
