# # WebP Compression Lab
# Exploring the effect of WebP quality settings on file size and image fidelity.

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

# Optional AVIF support
try:
    import pillow_avif  # noqa: F401
    AVIF_AVAILABLE = True
except ImportError:
    AVIF_AVAILABLE = False
    print("Warning: pillow-avif-plugin not found — AVIF will be skipped.")

IMG = "img/dice.png"
os.makedirs("output", exist_ok=True)
qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# ## 1. Save image at various WebP quality levels and plot file size

img = Image.open(IMG)

for q in qualities:
    path = f"output/dice_q{q}.webp"
    img.save(path, "WEBP", quality=q)
    print(f"Quality {q}: {os.path.getsize(path)} bytes")

sizes = [os.path.getsize(f"output/dice_q{q}.webp") for q in qualities]

plt.figure()
plt.plot(qualities, sizes, marker="o")
plt.xlabel("Quality")
plt.ylabel("File Size (bytes)")
plt.title("Quality vs File Size (WEBP)")
plt.savefig("output/quality_vs_size.png")
plt.show()
print("Saved output/quality_vs_size.png")

# Now do the same for AVIF
for q in qualities:
    path = f"output/dice_q{q}.avif"
    img.save(path, "AVIF", quality=q)
    print(f"Quality {q}: {os.path.getsize(path)} bytes")

plt.figure()
plt.plot(qualities, sizes, marker="o")
plt.xlabel("Quality")
plt.ylabel("File Size (bytes)")
plt.title("Quality vs File Size (AVIF)")
plt.savefig("output/quality_vs_size_avif.png")
plt.show()
print("Saved output/quality_vs_size_avif.png")

# ## 2. Visual comparison: original vs compressed (q=10) vs diff

original   = np.array(Image.open(IMG).convert("RGB"))
compressed = np.array(Image.open("output/dice_q10.webp").convert("RGB"))
diff = cv2.absdiff(original, compressed)

cv2.imshow("Original", original)
cv2.imshow("WEBP q=10", compressed)
cv2.imshow("Diff", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ## 3. Compare compression algorithms on a noise image

noise = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
noise_img = Image.fromarray(noise)

# Document this formats variable on how its used
# It is a list of tuples, where each tuple is a tuple of (name, format, kwargs)
# name is the name of the format, format is the format of the image, kwargs is a dictionary of keyword arguments

formats = [
    ("png",          "PNG",  {}),
    ("webp",         "WEBP", {}),
    ("jpg",          "JPEG", {}),
    ("tif",          "TIFF", {}),
    ("tif_lzw",      "TIFF", {"compression": "tiff_lzw"}),
    ("tif_deflate",  "TIFF", {"compression": "tiff_deflate"}),
    ("bmp",          "BMP",  {}),
]
if AVIF_AVAILABLE:
    formats.append(("avif", "AVIF", {}))

for name, fmt, kwargs in formats:
    path = f"output/noise_{name}.{name.split('_')[0]}"
    noise_img.save(path, format=fmt, **kwargs)
    size = os.path.getsize(path)
    print(f"{name:<14} size: {size / 1024:.2f} KB")

# ## 4. Image quality metrics: PSNR & SSIM

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)

def compute_ssim(img1, img2):
    y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    score, _ = ssim(y1, y2, full=True)
    return score

results = []
for q in qualities:
    comp = np.array(Image.open(f"output/dice_q{q}.webp").convert("RGB"))
    results.append((q, compute_psnr(original, comp), compute_ssim(original, comp)))
    print(f"Quality {q:3d} | PSNR: {results[-1][1]:.2f} dB | SSIM: {results[-1][2]:.4f}")

qs, psnrs, ssims_vals = zip(*results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(qs, psnrs,      marker="o", color="steelblue");  ax1.set_xlabel("Quality"); ax1.set_ylabel("PSNR (dB)"); ax1.set_title("Quality vs PSNR")
ax2.plot(qs, ssims_vals, marker="o", color="darkorange"); ax2.set_xlabel("Quality"); ax2.set_ylabel("SSIM");      ax2.set_title("Quality vs SSIM")
plt.tight_layout()
plt.savefig("output/quality_metrics.png")
plt.show()
print("Saved output/quality_metrics.png")

# Now do the same for AVIF
results = []
for q in qualities:
    comp = np.array(Image.open(f"output/dice_q{q}.avif").convert("RGB"))
    results.append((q, compute_psnr(original, comp), compute_ssim(original, comp)))
    print(f"Quality {q:3d} | PSNR: {results[-1][1]:.2f} dB | SSIM: {results[-1][2]:.4f}")

qs, psnrs, ssims_vals = zip(*results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(qs, psnrs,      marker="o", color="steelblue");  ax1.set_xlabel("Quality"); ax1.set_ylabel("PSNR (dB)"); ax1.set_title("Quality vs PSNR")
ax2.plot(qs, ssims_vals, marker="o", color="darkorange"); ax2.set_xlabel("Quality"); ax2.set_ylabel("SSIM");      ax2.set_title("Quality vs SSIM")
plt.tight_layout()
plt.savefig("output/quality_metrics_avif.png")
plt.show()
print("Saved output/quality_metrics_avif.png")

# ## 5. Fourier Transform (3-D surface): original PNG vs WebP-compressed

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

def fft_magnitude(img_array):
    """Return the log-scaled 2-D FFT magnitude spectrum of a grayscale image."""
    gray   = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    return 20 * np.log10(np.abs(fshift) + 1)   # +1 avoids log(0)

def downsample(arr, target=200):
    """Downsample a 2-D array to at most target x target for fast 3-D rendering."""
    sy = max(1, arr.shape[0] // target)
    sx = max(1, arr.shape[1] // target)
    return arr[::sy, ::sx]

# Re-use the already-saved WebP; change quality level here if desired
fft_webp_quality = 30
fft_webp_path    = f"output/dice_q{fft_webp_quality}.webp"

original_rgb = np.array(Image.open(IMG).convert("RGB"))
webp_rgb     = np.array(Image.open(fft_webp_path).convert("RGB"))

mag_orig = downsample(fft_magnitude(original_rgb))
mag_webp = downsample(fft_magnitude(webp_rgb))

vmin = min(mag_orig.min(), mag_webp.min())
vmax = max(mag_orig.max(), mag_webp.max())

fig = plt.figure(figsize=(14, 6))

for idx, (mag, label) in enumerate(
    [(mag_orig, "Original PNG"), (mag_webp, f"WebP q={fft_webp_quality}")]
):
    rows, cols = mag.shape
    # Normalized frequency axes: -0.5 … +0.5 cycles/pixel (DC centred by fftshift)
    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    X, Y = np.meshgrid(fx, fy)

    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    surf = ax.plot_surface(X, Y, mag, cmap="inferno",
                           vmin=vmin, vmax=vmax,
                           linewidth=0, antialiased=False)
    ax.set_title(f"FFT Magnitude — {label}")
    ax.set_xlabel("Frequency X (cycles/px)")
    ax.set_ylabel("Frequency Y (cycles/px)")
    ax.set_zlabel("Magnitude (dB)")
    ax.set_zlim(vmin, vmax)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])   # [left, bottom, width, height]
fig.colorbar(surf, cax=cbar_ax, label="Magnitude (dB)")
plt.suptitle("2-D Fourier Transform — 3-D Surface Comparison", fontsize=14)
plt.subplots_adjust(right=0.90)
plt.savefig("output/fft_comparison.png", bbox_inches="tight")
plt.show()
print("Saved output/fft_comparison.png")
