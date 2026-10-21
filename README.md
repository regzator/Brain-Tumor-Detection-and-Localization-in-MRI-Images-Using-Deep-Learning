<div align="center">

# 🧠 Brain Tumor Detection & Localization in MRI Scans
*A Deep Learning pipeline leveraging U-Net encoder-decoder architecture for precise brain tumor segmentation from MRI image slices.*

</div>

---

## 📌 About The Project
Accurate and early detection of brain tumors is vital in neuro-oncology. This project was developed as my **B.Sc. Final Year Project at the University of Tabriz**. It implements an end-to-end semantic segmentation pipeline using TensorFlow/Keras to process MRI scans, detect tumor regions, and generate high-precision localization masks.

---

## 🏗️ Model Architecture (U-Net)
The model is built on a modified U-Net architecture featuring:
- **Encoder (Downsampling):** Feature extraction via stacked convolutional blocks, batch normalization, ReLU activations, max-pooling, and spatial dropout ($p=0.3$).
- **Bottleneck:** High-capacity feature representation layer ($1024$ channels).
- **Decoder (Upsampling):** Transpose convolutions coupled with skip connections (`concatenate`) from the encoder to recover spatial resolution and precise tumor boundaries.
- **Output:** Pixel-wise classification layer with a `sigmoid` activation.

---

## 📊 Dataset & Preprocessing
- **Dataset Source:** Kaggle brain MRI segmentation dataset containing image slices and corresponding binary masks.
- **Preprocessing Pipeline:**
  - Resized uniformly to $256 \times 256$ pixels.
  - Converted to single-channel grayscale.
  - Normalized pixel intensities to the range $[0, 1]$.
  - Thresholding masks into binary labels ($\ge 0.5 \to 1$, else $0$).
- **Data Augmentation (Albumentations):**
  - Horizontal & Vertical Flips (`p=0.5`)
  - Affine transforms (scaling, translation, rotation) (`p=0.5`)
  - Random Brightness & Contrast adjustment (`p=0.2`)
  - *Stochastic bypass mechanism included for robust generalization.*

---

## ⚙️ Training Setup & Performance
- **Loss Function:** Composite Loss function combining Binary Cross-Entropy and Dice Loss ($\mathcal{L} = 0.5 \cdot \text{BCE} + 0.5 \cdot \text{Dice Loss}$).
- **Optimizer:** Adam ($\text{learning rate} = 1e-4$).
- **Callbacks:** `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping`.
- **Results:** Achieved a peak validation **Dice Coefficient of ~0.76** and validation loss of **0.1305**.
