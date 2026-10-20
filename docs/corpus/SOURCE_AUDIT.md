# Audit Sumber Data & Legalitas Akuisisi Korpus OJP (Workstream A1)

**Status:** DRAF v1.0 — menunggu tinjauan dan persetujuan promotor
**Tanggal:** 20 Oktober 2026
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
| Indeed (via pustaka JobSpy) | **PAKAI, dengan catatan metode** — kanal utama ketiga strata |
| Google Jobs (via JobSpy) | **PARKIR** — rusak secara teknis per Okt 2026; kontribusi historis ≈0,2% |
| LinkedIn (via JobSpy) | **JANGAN PAKAI** — ToS melarang eksplisit; risiko hukum/etika tertinggi |
| ZipRecruiter (via JobSpy) | **PARKIR** — cakupan AS saja; Indeed sudah memadai untuk strata OECD |
| Jobstreet by SEEK (ID/MY/SG/PH) | **EVALUASI** — kandidat kuat penambah volume Indonesia & ASEAN |
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
- **Status empiris:** sengaja TIDAK diuji.
- **Keputusan diusulkan:** **JANGAN PAKAI** (drop permanen). Konsekuensi volume kecil karena
  batch-001 praktis Indeed-only.

### 3.4 ZipRecruiter — via JobSpy

- Cakupan efektif AS/Kanada saja; strata OECD sudah terpenuhi lewat Indeed (42 ribu iklan AS pada
  batch-001). **Keputusan diusulkan: PARKIR** (dipakai hanya bila Indeed AS terganggu).

### 3.5 Jobstreet by SEEK (Indonesia, Malaysia, Singapura, Filipina)

- **Metode akses:** halaman pencarian publik; terdapat endpoint JSON publik yang dipakai situsnya
  sendiri. Perlu adapter khusus (belum ada di JobSpy).
- **Posisi ToS:** ToS SEEK membatasi akses otomatis; data publik, prinsip yang sama dengan Indeed.
- **Nilai:** papan lowongan terbesar untuk pasar kerja formal Indonesia/ASEAN — kandidat terkuat
  untuk menutup kekurangan strata Indonesia (saat ini 2,2 ribu vs target 10–20 ribu).
- **Keputusan diusulkan:** **EVALUASI** — bangun purwarupa adapter kecil, ukur kelengkapan metadata
  terhadap Tabel 3.10, lalu putuskan bersama promotor.

### 3.6 Glints (Indonesia/SEA) dan 3.7 Kalibrr (Indonesia/Filipina)

- Platform lowongan teknologi populer di Indonesia; halaman publik dengan endpoint JSON.
- **Keputusan diusulkan:** **EVALUASI** dengan prosedur yang sama dengan Jobstreet (purwarupa →
  audit metadata → keputusan). Prioritas setelah Jobstreet.

### 3.8 API resmi (legal paling bersih)

| API | Cakupan | Catatan |
|---|---|---|
| **Adzuna** | ~16 negara OECD + SG | API resmi, kuota riset gratis, metadata baik; **ajukan kunci API** |
| **USAJOBS** | AS (pemerintah federal) | API resmi; relevansi RPL sempit tapi bersih |
| **Arbeitnow** | Eropa | API publik gratis; volume kecil |

API resmi memperkuat posisi etis korpus OECD dan menjadi pembanding independen terhadap Indeed.

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

| # | Keputusan | Usulan |
|---|---|---|
| K1 | Menghapus LinkedIn dari seluruh desain akuisisi | Setuju (drop) |
| K2 | Melanjutkan akuisisi Indeed via JobSpy dengan prinsip §2 dan dokumentasi jujur di §3.7 proposal | Setuju (pakai) |
| K3 | Mengembangkan adapter Jobstreet → Glints → Kalibrr (urutan prioritas) untuk menutup target Indonesia | Setuju (evaluasi bertahap) |
| K4 | Mengajukan akses API resmi Adzuna (+ opsional USAJOBS/Arbeitnow) untuk strata OECD | Setuju (ajukan) |

**Persetujuan promotor:**

- Nama / tanda tangan / tanggal: ______________________________
- Catatan revisi: ______________________________

---

## Riwayat perubahan

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0 (draf) | 2026-10-20 | Draf awal: matriks 11 sumber, prinsip legal–etis, 4 butir keputusan |
