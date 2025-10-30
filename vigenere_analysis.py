#!/usr/bin/env python3
"""
vigenere_analysis.py

Implementasi Vigenère cipher + analisis frekuensi + Kasiski + Friedman.
Cocok dijalankan di VSCode / terminal.

Fitur:
- encrypt / decrypt
- frequency analysis (tabel & optional plot)
- kasiski examination (mencari n-gram berulang dan faktor jarak)
- friedman test (index of coincidence + estimasi panjang kunci)

Author: ChatGPT (penyesuaian untuk tugas praktikum)
"""

from collections import Counter, defaultdict
import argparse
import math
import re
import textwrap
import sys

# Jika ingin plotting, matplotlib diperlukan
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def clean_text(s: str) -> str:
    """Bersihkan teks: ambil huruf A-Z saja dan konversi ke uppercase."""
    return "".join([c for c in s.upper() if c.isalpha()])


def repeat_key_to_length(key: str, length: int) -> str:
    """Ulangi key sampai panjang length."""
    key = clean_text(key)
    if not key:
        raise ValueError("Key tidak boleh kosong atau tidak mengandung huruf.")
    return (key * ((length // len(key)) + 1))[:length]


def vigenere_encrypt(plaintext: str, key: str) -> str:
    p = clean_text(plaintext)
    k = repeat_key_to_length(key, len(p))
    res_chars = []
    for pc, kc in zip(p, k):
        pi = ord(pc) - 65
        ki = ord(kc) - 65
        ci = (pi + ki) % 26
        res_chars.append(chr(ci + 65))
    return "".join(res_chars)


def vigenere_decrypt(ciphertext: str, key: str) -> str:
    c = clean_text(ciphertext)
    k = repeat_key_to_length(key, len(c))
    res_chars = []
    for cc, kc in zip(c, k):
        ci = ord(cc) - 65
        ki = ord(kc) - 65
        pi = (ci - ki) % 26
        res_chars.append(chr(pi + 65))
    return "".join(res_chars)


def frequency_analysis(text: str) -> dict:
    """Kembalikan Counter huruf (A-Z) sebagai frekuensi absolut."""
    t = clean_text(text)
    cnt = Counter(t)
    # ensure all letters present (maybe zero)
    for ch in ALPHABET:
        cnt.setdefault(ch, 0)
    return dict(sorted(cnt.items()))


def plot_frequency(freq: dict, title: str = "Letter Frequency"):
    """Plot frekuensi huruf menggunakan matplotlib (jika tersedia)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("Matplotlib tidak tersedia. Install dengan `pip install matplotlib`.")
    letters = list(freq.keys())
    counts = [freq[ch] for ch in letters]
    plt.figure(figsize=(10, 4))
    plt.bar(letters, counts)
    plt.title(title)
    plt.xlabel("Letter")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def index_of_coincidence(text: str) -> float:
    """Hitung Index of Coincidence (IC) untuk teks (A-Z)."""
    t = clean_text(text)
    N = len(t)
    if N <= 1:
        return 0.0
    counts = Counter(t)
    numerator = sum(v * (v - 1) for v in counts.values())
    denominator = N * (N - 1)
    return numerator / denominator


def friedman_estimate_key_length(text: str) -> float:
    """
    Estimasi panjang kunci berdasarkan Friedman test.
    Rumus sederhana: (0.0265 * N) / ((IC * (N - 1)) - 0.0385 * N + 0.065)
    (Konstanta berdasarkan bahasa Inggris; formula umum).
    """
    t = clean_text(text)
    N = len(t)
    if N <= 1:
        return 0.0
    IC = index_of_coincidence(t)
    # Constants for English language approx
    K0 = 0.066  # IC of English
    Kr = 0.0385  # IC of random text
    # Friedman estimation formula:
    numerator = (K0 - Kr) * N
    denominator = ((IC - Kr) * (N - 1)) + 1e-12  # avoid div by zero
    if denominator == 0:
        return float('inf')
    est = numerator / denominator
    return est


def kasiski_examination(text: str, min_len: int = 3, max_len: int = 5) -> dict:
    """
    Cari pola n-gram berulang (n dari min_len sampai max_len).
    Untuk setiap pola yang muncul >1, hitung jarak antar kemunculan.
    Kembalikan dict: pattern -> list of distances
    """
    t = clean_text(text)
    results = {}
    for n in range(min_len, max_len + 1):
        positions = defaultdict(list)
        for i in range(len(t) - n + 1):
            chunk = t[i:i + n]
            positions[chunk].append(i)
        # patterns with multiple occurrences
        for chunk, pos_list in positions.items():
            if len(pos_list) > 1:
                distances = []
                for i in range(1, len(pos_list)):
                    distances.append(pos_list[i] - pos_list[i - 1])
                results.setdefault(chunk, []).extend(distances)
    return results


def factorize(n: int) -> list:
    """Kembalikan faktor-faktor >1 dari n (kecil ke besar)."""
    n = abs(n)
    if n <= 1:
        return []
    facts = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            facts.append(i)
            n //= i
        i += 1
    if n > 1:
        facts.append(n)
    return sorted(list(set(facts)))


def kasiski_suggestions(kasiski_result: dict) -> dict:
    """
    Ambil jarak dari kasiski_result dan kembalikan faktor yang mungkin menjadi panjang kunci.
    Mengembalikan mapping faktor -> count (berapa kali muncul).
    """
    all_distances = []
    for distances in kasiski_result.values():
        all_distances.extend(distances)
    factor_counts = Counter()
    for d in all_distances:
        for f in factorize(d):
            factor_counts[f] += 1
    return dict(factor_counts.most_common())


def pretty_print_freq(freq: dict):
    total = sum(freq.values())
    print("Letter | Count | Frequency")
    print("--------------------------")
    for ch, cnt in freq.items():
        pct = (cnt / total * 100) if total else 0
        print(f"  {ch}   |  {cnt:3d}  |  {pct:6.2f}%")
    print(f"\nTotal letters: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Vigenère cipher + analysis (encrypt, decrypt, analyze).")
    parser.add_argument("--mode", choices=["encrypt", "decrypt", "analyze"], required=True,
                        help="Mode operasi.")
    parser.add_argument("--text", "-t", type=str, default="",
                        help="Teks input (plaintext atau ciphertext). Jika kosong, baca dari stdin.")
    parser.add_argument("--key", "-k", type=str, default="",
                        help="Kunci (key) untuk Vigenère (string).")
    parser.add_argument("--plot", action="store_true", help="Tampilkan plot frekuensi (matplotlib).")
    parser.add_argument("--kasiski-min", type=int, default=3, help="Minimum n-gram untuk Kasiski (default 3).")
    parser.add_argument("--kasiski-max", type=int, default=5, help="Maximum n-gram untuk Kasiski (default 5).")
    args = parser.parse_args()

    if not args.text:
        # baca dari stdin
        print("Masukkan teks, lalu tekan CTRL+D (Unix) / CTRL+Z (Windows) pada baris baru:")
        args.text = sys.stdin.read().strip()

    if args.mode in ["encrypt", "decrypt"] and not args.key:
        print("Error: mode encrypt/decrypt membutuhkan --key. Gunakan -k atau --key.")
        sys.exit(1)

    if args.mode == "encrypt":
        cipher = vigenere_encrypt(args.text, args.key)
        print("=== ENCRYPT ===")
        print(f"Plaintext:  {clean_text(args.text)}")
        print(f"Key:        {clean_text(args.key)}")
        print(f"Ciphertext: {cipher}")
        # show frequency if requested
        freq = frequency_analysis(cipher)
        print()
        pretty_print_freq(freq)
        if args.plot:
            plot_frequency(freq, title="Frequency of Ciphertext (Encrypted)")

    elif args.mode == "decrypt":
        plain = vigenere_decrypt(args.text, args.key)
        print("=== DECRYPT ===")
        print(f"Ciphertext: {clean_text(args.text)}")
        print(f"Key:        {clean_text(args.key)}")
        print(f"Plaintext:  {plain}")
        # frequency of ciphertext available too
        freq = frequency_analysis(args.text)
        print()
        pretty_print_freq(freq)
        if args.plot:
            plot_frequency(freq, title="Frequency of Ciphertext (Before Decryption)")

    elif args.mode == "analyze":
        t = clean_text(args.text)
        print("=== ANALYSIS ===")
        print(f"Input text (cleaned): {t}")
        print("\n--- Frequency Analysis ---")
        freq = frequency_analysis(t)
        pretty_print_freq(freq)
        if args.plot:
            plot_frequency(freq, title="Frequency of Input Text")

        print("\n--- Index of Coincidence (IC) & Friedman ---")
        ic = index_of_coincidence(t)
        est_len = friedman_estimate_key_length(t)
        print(f"Index of Coincidence (IC): {ic:.6f}")
        print(f"Friedman estimated key length (approx): {est_len:.2f}")

        print("\n--- Kasiski Examination ---")
        kas = kasiski_examination(t, min_len=args.kasiski_min, max_len=args.kasiski_max)
        if not kas:
            print("Tidak ditemukan pola n-gram berulang (dengan panjang yang dipilih).")
        else:
            for chunk, distances in kas.items():
                print(f"Pattern '{chunk}' found; distances: {distances}")
            suggestions = kasiski_suggestions(kas)
            if suggestions:
                print("\nSuggested key length factors (from Kasiski distances):")
                for factor, count in suggestions.items():
                    print(f"  Factor {factor} -> {count} occurrence(s)")
            else:
                print("No factor suggestions could be produced from distances.")

        # Jika user juga memberikan key, coba dekripsi sebagai validasi
        if args.key:
            try:
                dec = vigenere_decrypt(t, args.key)
                print("\n--- Validation (decrypt with provided key) ---")
                print(f"Using key: {clean_text(args.key)}")
                print(f"Decrypted text: {dec}")
            except Exception as e:
                print(f"Validation (decrypt) failed: {e}")


if __name__ == "__main__":
    main()