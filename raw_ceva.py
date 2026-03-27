import rawpy
import matplotlib.pyplot as plt
import numpy as np
import os

# Înlocuiește cu calea către un fișier RAW real (ex: .CR2, .NEF, .DNG)
# Recomand o imagine cu texturi fine (haine, frunze) sau margini ascuțite
raw_path = 'img/DSC08381.ARW' # asta nu exista inca ca e uriasa

if not os.path.exists(raw_path):
    print(f"Eroare: Fisierul {raw_path} nu a fost gasit. Adaugati o imagine RAW in folder.")
else:
    with rawpy.imread(raw_path) as raw:
        # 1. Interpolare Biliniară (Rapidă, dar produce artefacte la margini - 'zippering')
        # user_wb aplică balansul de alb din cameră pentru ca imaginea să arate natural
        rgb_bilinear = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, 
            use_camera_wb=True, 
            half_size=False
        )

        # 2. Interpolare AHD (Adaptive Homogeneity-Directed)
        # Un algoritm mult mai avansat care adaptează interpolarea pe direcția marginilor
        rgb_ahd = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, 
            use_camera_wb=True, 
            half_size=False
        )

    crop_bilinear = rgb_bilinear
    crop_ahd = rgb_ahd

    # 3. Calculăm Harta Diferențelor (Eroarea)
    # Scădem cele două matrice pentru a vedea EXACT unde diferă algoritmii
    # Folosim np.abs și convertim la int pentru a evita overflow la uint8
    diff = np.abs(crop_ahd - crop_bilinear)

    # --- Secțiunea de Afișare (Plotting) ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(crop_bilinear)
    axs[0].set_title("1. Interpolare Biliniară\n(Căutați artefacte 'zipper' pe margini)", fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(crop_ahd)
    axs[1].set_title("2. Interpolare Adaptivă (AHD)\n(Margini reconstruite corect)", fontsize=12)
    axs[1].axis('off')

    # Afișăm diferența. Zonele negre = algoritmii au dat același rezultat. 
    # Zonele colorate/albe = diferențe majore de reconstrucție.
    axs[2].imshow(diff)
    axs[2].set_title("3. Harta Diferențelor (Eroarea Biliniară)\n(Zonele luminate indică diferențele)", fontsize=12)
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()