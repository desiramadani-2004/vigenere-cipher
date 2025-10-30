from collections import Counter

def vigenere_encrypt(plaintext, key):
    plaintext = plaintext.upper().replace(" ", "")
    key = key.upper()
    ciphertext = ""
    for i in range(len(plaintext)):
        p = ord(plaintext[i]) - 65
        k = ord(key[i % len(key)]) - 65
        c = (p + k) % 26
        ciphertext += chr(c + 65)
    return ciphertext

def vigenere_decrypt(ciphertext, key):
    ciphertext = ciphertext.upper().replace(" ", "")
    key = key.upper()
    plaintext = ""
    for i in range(len(ciphertext)):
        c = ord(ciphertext[i]) - 65
        k = ord(key[i % len(key)]) - 65
        p = (c - k) % 26
        plaintext += chr(p + 65)
    return plaintext

def frequency_analysis(text):
    text = text.upper().replace(" ", "")
    freq = Counter(text)
    total = sum(freq.values())
    for letter in sorted(freq):
        print(f"{letter}: {freq[letter]} ({freq[letter]/total:.2%})")
    return freq

if __name__ == "__main__":
    plaintext = "ATTACKATDAWN"
    key = "LEMON"

    ciphertext = vigenere_encrypt(plaintext, key)
    print("Plaintext :", plaintext)
    print("Key       :", key)
    print("Ciphertext:", ciphertext)

    print("\n=== Frequency Analysis ===")
    frequency_analysis(ciphertext)

    decrypted = vigenere_decrypt(ciphertext, key)
    print("\nDecrypted :", decrypted)