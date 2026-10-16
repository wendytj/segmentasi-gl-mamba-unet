import os
import sys
import time
import ssl
import urllib.request
import urllib.parse
import urllib.error

# --- KONFIGURASI ENDPOINT MULTI-PROVIDER ---
DOWNLOAD_ENDPOINTS = [
    "https://speed.cloudflare.com/__down?bytes=10000000", # 10 MB Cloudflare Stream
    "https://httpbin.org/bytes/1048576"                   # Fallback 1 MB Stream
]

# Endpoint Upload Publik Resilient
UPLOAD_ENDPOINTS = [
    "https://speed.cloudflare.com/__up",   # Cloudflare Speed Test Upload Endpoint
    "https://httpbin.org/post",            # Standard HTTP POST
    "https://postman-echo.com/post"        # Fallback Postman
]

DUMMY_SIZE_MB = 2           # Ukuran tes upload di RAM (2 MB)
TIMEOUT_SECONDS = 10        # Timeout maksimal per tes
MIN_UPLOAD_SPEED_MBPS = 0.5 # Syarat minimal upload (500 KB/s)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def create_flexible_ssl_context():
    """Membuat SSL Context yang fleksibel di Windows maupun Linux."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def test_download_speed():
    """Menguji kecepatan download menggunakan multi-endpoint resilient."""
    context = create_flexible_ssl_context()
    
    for url in DOWNLOAD_ENDPOINTS:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS, context=context) as response:
                data = response.read()
                elapsed = time.time() - start_time
                size_mb = len(data) / (1024 * 1024)
                if size_mb > 0 and elapsed > 0:
                    return size_mb / elapsed
        except Exception:
            continue  # Jika satu endpoint gagal, coba endpoint berikutnya
            
    return 0.0


def test_upload_speed():
    """Menguji kecepatan upload (outbound) dengan payload multipart terformat."""
    # Buat dummy data berbentuk teks/padding biasa agar tidak dianggap virus/malware oleh WAF
    dummy_text = "X" * (DUMMY_SIZE_MB * 1024 * 1024)
    boundary = "----WebKitFormBoundarySanityCheck7MA4YWxkTrZu0gW"
    
    # Buat format multipart/form-data standar
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.bin"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
        f"{dummy_text}\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')

    context = create_flexible_ssl_context()

    for url in UPLOAD_ENDPOINTS:
        try:
            req_headers = {
                **HEADERS,
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Content-Length': str(len(body))
            }
            req = urllib.request.Request(url, data=body, method='POST', headers=req_headers)
            
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS, context=context) as response:
                _ = response.read()
                elapsed = time.time() - start_time
                if elapsed > 0:
                    return DUMMY_SIZE_MB / elapsed
        except Exception:
            continue  # Coba endpoint upload berikutnya jika satu gagal
            
    return 0.0


def run_sanity_check():
    os_name = "Linux" if sys.platform.startswith("linux") else "Windows"
    print("=" * 55)
    print(f"🚀 NETWORK SANITY CHECK ({os_name.upper()} ENVIRONMENT)")
    print("=" * 55)

    # 1. Tes Download
    print("1️⃣  Menguji Download Speed (Inbound)...", end="", flush=True)
    dl_speed = test_download_speed()
    print(f"\r  ✅ Download Speed : {dl_speed:.2f} MB/s                    ")

    # 2. Tes Upload
    print("2️⃣  Menguji Upload Speed (Outbound)...", end="", flush=True)
    ul_speed = test_upload_speed()
    print(f"\r  ✅ Upload Speed   : {ul_speed:.2f} MB/s                    ")

    # 3. Evaluasi
    print("-" * 55)
    if ul_speed < MIN_UPLOAD_SPEED_MBPS:
        print("🚨 KEPUTUSAN : REJECT / NETWORK THROTTLED!")
        print(f"⚠️  Outbound speed ({ul_speed:.2f} MB/s) di bawah batas ({MIN_UPLOAD_SPEED_MBPS} MB/s).")
    else:
        print("🎉 KEPUTUSAN : PASS / SAFE!")
        print("✅ Jaringan normal. Aman untuk training & auto-upload.")
    print("=" * 55)


if __name__ == '__main__':
    run_sanity_check()