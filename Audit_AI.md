# Audit AI

### Pentru codul din laborator, am folosit Antigravity, care este un AI de la Google. Ca model de limbaj, am folosit Claude 4.6 Sonnet. Am mai folosit si Gemini 3 Pro Preview.


Urmatoarele sectiuni au fost generate de AI, si au trebuit corectate de noi.
```py
def compute_ssim(img1, img2):
    y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    score, _ = ssim(y1, y2, full=True)
    return score
```

### Aici am vrut sa fac o functie care sa calculeze SSIM pentru a vedea diferenta intre diferite formate de compresie, dar AI-ul a generat-o initial prin a calcula SSIM pe toata imaginea in versiunea grayscale, ceea ce nu era tocmai ideal pentru a vedea diferenta intre diferite formate de compresie, ci trebuia sa le comparam in versiunea YCrCb. Also Ai-ul a crezut ca scorul SSIM este cuprins intre 0 si 1, dar de fapt este cuprins intre -1 si 1. [[1]](https://www.freeimagecompression.com/guides/image-compression-quality-standards-evaluation) [[2]](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)
```py
def compute_ssim(img1, img2):
    y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(y1, y2, full=True)
    return score
```
Versiunea initiala ^


### La sectiunea FFT, am vrut sa fac o functie care sa calculeze FFT pentru a vedea diferenta intre diferite formate de compresie, doar ca AI-ul a gresit axele X si Y la plotarea FFT-ului. In loc sa puna frecventa la care se schimba pixelii pe orizontala pe axa X si cea verticala pe axa Y, a pus doua intervale de la 0 la lungimea maxima a imaginii pe lungime respectiv pe latime. [[1]](https://www.mathworks.com/help/images/fourier-transform.html)

```py
for idx, (mag, label) in enumerate(
    [(mag_orig, "Original PNG"), (mag_webp, f"WebP q={fft_quality}")]
):
    rows, cols = mag.shape
    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    X, Y = np.meshgrid(fx, fy) # Aici a trebuit sa modific eu, initial AI-ul a crezut ca X si Y sunt egale cu cols si rows, ceea ce nu era corect

    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    surf = ax.plot_surface(X, Y, mag, cmap="inferno",
                           vmin=vmin, vmax=vmax,
                           linewidth=0, antialiased=False)
    ax.set_title(f"FFT Magnitude — {label}")
    ax.set_xlabel("Frequency X (cycles/px)") # Aici a trebuit sa modific eu label-ul
    ax.set_ylabel("Frequency Y (cycles/px)") # Same here
    ax.set_zlabel("Magnitude (dB)")
    ax.set_zlim(vmin, vmax)
```
Versiunea initiala ^

