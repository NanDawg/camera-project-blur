# FOTO KITA BLUR DAWGGGGGGGG

Kamera desktop real-time yang otomatis mengaktifkan efek blur ketika kamu
menunjukkan gesture ✌️ PEACE, dibangun dengan OpenCV + MediaPipe Tasks
Hand Landmarker API.

---

## 1. Fitur

### Deteksi gesture
- Mendeteksi 5 gesture berbeda dari landmark tangan: **PEACE ✌️**, **OPEN HAND ✋**,
  **FIST 👊**, **ONE FINGER ☝️**, **THUMBS UP 👍**, dan `NORMAL` kalau tidak ada
  tangan terdeteksi.
- Hanya gesture **PEACE** yang mengaktifkan blur secara otomatis.
- Pakai perhitungan rasio jarak tip-ke-wrist vs pip-ke-wrist (dinormalisasi
  ukuran telapak), jadi tidak sekadar hitung jumlah jari — bisa membedakan
  kombinasi jari yang beda dengan jelas.

### Anti-flicker (fast-attack / slow-release)
- Blur **langsung nyala** begitu PEACE terdeteksi stabil selama beberapa
  frame (`GESTURE_ON_FRAMES = 2`, ±60ms) — responsif, tidak perlu "lewat"
  gesture lain dulu.
- Blur **hanya mati** setelah gesture non-PEACE bertahan beberapa frame
  (`GESTURE_OFF_FRAMES = 5`) — supaya tidak berkedip-kedip saat tangan
  sedikit goyang.

### Blur effect
- Transisi blur halus (interpolasi nilai, bukan langsung on/off mentah).
- 3 mode kekuatan blur: **Light**, **Medium** (default), **Heavy**.
- Bisa dikontrol otomatis via gesture, atau manual lewat hotkey.

### Kamera & tampilan
- Resolusi target 1920x1080, otomatis fallback ke resolusi terbaik yang
  didukung webcam kalau tidak tersedia.
- Mirror / selfie mode.
- Fullscreen yang benar-benar memenuhi layar (aspect-ratio preserving
  resize + center crop) — video tidak pernah gepeng atau ada area kosong,
  meskipun resolusi kamera dan layar beda rasio.
- UI dark/cinematic minimalis dengan panel semi-transparent, tidak
  menutupi wajah:
  - **Tengah atas**: mode `AUTO GESTURE` / `MANUAL BLUR`
  - **Tengah bawah**: satu panel gabungan berisi label gesture saat ini +
    status `BLUR ON` / `BLUR OFF`
  - **Kiri atas**: indikator `REC` (muncul hanya saat sedang merekam)
  - UI bisa disembunyikan sepenuhnya kalau mau tampilan bersih.

### Screenshot & recording
- Screenshot (`SPACE`) menyimpan frame yang sedang tampil apa adanya —
  termasuk blur (kalau aktif), mirror, dan crop fullscreen.
- Recording (`R`) merekam hasil frame setelah efek blur diterapkan, dengan
  fallback codec otomatis kalau codec utama tidak didukung sistem.

### Kontrol manual
- Manual override blur (`B`) tanpa mematikan deteksi gesture di baliknya —
  gesture tetap dibaca terus, cuma output blur-nya yang di-override.

---

## 2. Struktur Project

```
camera_project/
├── main.py
├── hand_landmarker.task   <- kamu tambahkan sendiri (lihat langkah 4)
├── requirements.txt
├── README.md
├── screenshots/           <- otomatis dibuat / terisi saat screenshot
└── recordings/            <- otomatis dibuat / terisi saat recording
```

---

## 3. Install Dependency

Semua perintah di bawah dijalankan di **terminal** (Command Prompt /
PowerShell), bukan di dalam Python.

1. Buka folder `camera_project` di File Explorer, ketik `cmd` di address
   bar lalu Enter (atau klik kanan → *Open in Terminal*).
2. Buat & aktifkan virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   Setelah aktif, prompt akan ada tulisan `(venv)` di depannya.
3. Install semua library yang dibutuhkan:
   ```
   pip install -r requirements.txt
   ```

**Catatan kompatibilitas Python 3.14:** paket `mediapipe` secara historis
hanya mem-publish wheel resmi untuk Python 3.9–3.12. Kalau instalasi gagal
di Python 3.14, pakai interpreter Python 3.11/3.12 untuk virtual
environment ini — kode `main.py` sendiri tidak bergantung versi Python
tertentu.

---

## 4. Menyiapkan hand_landmarker.task

1. Download model resmi dari:
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
2. Simpan sebagai `hand_landmarker.task`
3. Taruh tepat di dalam folder `camera_project/`, sejajar dengan `main.py`

Kalau file tidak ditemukan, aplikasi akan berhenti dan menampilkan pesan
error yang jelas beserta lokasi yang diharapkan.

---

## 5. Menjalankan

Pastikan `(venv)` masih aktif di terminal (kalau baru buka terminal lagi,
jalankan `venv\Scripts\activate` dulu), lalu:

```
python main.py
```

Pastikan tidak ada aplikasi lain (Zoom, Teams, browser tab, OBS, dll.)
yang sedang memakai webcam.

---

## 6. Hotkeys

| Tombol | Fungsi |
|---|---|
| `F` | Toggle fullscreen (gunakan ini kalau `F11` tidak terdeteksi di sistemmu) |
| `F11` | Toggle fullscreen (best-effort, tergantung platform/build OpenCV) |
| `ESC` | Keluar dari fullscreen |
| `Q` | Keluar aplikasi |
| `H` | Sembunyikan / tampilkan overlay UI |
| `B` | Cycle: AUTO GESTURE → MANUAL BLUR ON → MANUAL BLUR OFF → AUTO GESTURE |
| `M` | Toggle mirror / selfie mode |
| `1` | Blur mode: Light |
| `2` | Blur mode: Medium (default) |
| `3` | Blur mode: Heavy |
| `R` | Start / stop recording |
| `C` | Reset semua setting ke default |
| `SPACE` | Ambil screenshot |
| `[` / `]` | Kurangi / tambah sensitivitas deteksi gesture |

> **Catatan:** instruksi awal sempat minta `R` untuk dua fungsi (Recording
> dan Reset). Karena satu tombol tidak bisa aman untuk dua aksi berbeda,
> `R` dipakai untuk **Recording**, dan **Reset** dipindah ke `C`.

---

## 7. Gesture yang Dikenali

`detect_gesture()` di `main.py` mengembalikan salah satu dari:

- `"peace"` – ✌️ index + middle terangkat, ring + pinky terlipat → **mengaktifkan blur**
- `"open"` – ✋ semua jari terangkat
- `"fist"` – 👊 semua jari terlipat
- `"one"` – ☝️ hanya index terangkat
- `"thumbs_up"` – 👍 hanya ibu jari terangkat
- `"unknown"` – kombinasi lain
- `None` – tidak ada tangan terdeteksi

Kalau gesture terasa kurang akurat, sesuaikan langsung saat aplikasi
berjalan dengan tombol `[` (kurangi sensitivitas) / `]` (tambah
sensitivitas), atau ubah nilai `FINGER_EXTENSION_THRESHOLD` /
`THUMB_EXTENSION_THRESHOLD` di `main.py`. Kalau blur terasa terlalu cepat
mati/nyala, sesuaikan `GESTURE_ON_FRAMES` (delay nyala) dan
`GESTURE_OFF_FRAMES` (delay mati) di bagian CONFIGURATION.

---

## 8. Troubleshooting Webcam

- **"Camera not found."** → Pastikan webcam terpasang, tidak dipakai
  aplikasi lain, dan izin kamera untuk Python diizinkan di
  Windows Settings > Privacy > Camera.
- **Gambar hitam / freeze** → Lepas-pasang ulang webcam, atau ganti index
  kamera di `cv2.VideoCapture(0, ...)` menjadi `1`/`2` kalau ada lebih dari
  satu kamera.
- **Resolusi tidak sesuai target 1920x1080** → Aplikasi otomatis fallback
  ke resolusi terbaik yang didukung webcam kamu; ini normal dan aman.

---

## 9. Troubleshooting MediaPipe

- **`hand_landmarker.task not found.`** → Pastikan file model berada
  persis di folder yang sama dengan `main.py`, bukan di subfolder lain.
- **ImportError terkait `mp.tasks`** → Pastikan `mediapipe` yang terinstal
  sudah mendukung Tasks API (>= 0.10.x atau 1.0.0). Cek versi dengan:
  ```
  python -c "import mediapipe as mp; print(mp.__version__)"
  ```
- **Instalasi gagal di Python 3.14** → lihat catatan di bagian 3, gunakan
  Python 3.11/3.12 sebagai alternatif sementara.
- **Warning "unicode/basestring/long is not defined" di `six.py`** → aman,
  diabaikan saja. Itu warning linter (Pylance) untuk library dependency
  bawaan (`six`), bukan kode kamu, dan tidak memengaruhi jalannya program.

---

## 10. Catatan Performa

- `HandLandmarker` diinisialisasi **satu kali** saat aplikasi start (bukan
  per-frame), berjalan dalam `RunningMode.VIDEO` dengan timestamp yang
  selalu meningkat, sesuai rekomendasi MediaPipe untuk streaming real-time.
- Blur mode **Heavy** di resolusi tinggi lebih memakan CPU — turunkan ke
  Light/Medium (`1` / `2`) kalau performa terasa turun di laptopmu.
