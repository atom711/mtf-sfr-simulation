import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_synthetic_slanted_edge(width=512, height=512, angle_deg=5.0):
    """
    Generates a pristine, high-contrast slanted edge target.
    Slanted edges are used in ISO 12233 standards to oversample the 
    Spatial Frequency Response (SFR) and eliminate pixel-phase aliasing.
    """
    y, x = np.indices((height, width))
    angle_rad = np.radians(angle_deg)
    
    # Mathematical boundary equation for a rotated slanted edge line
    edge_line = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    center = (width / 2) * np.cos(angle_rad) + (height / 2) * np.sin(angle_rad)
    
    # Create absolute step transition: Dark side (40 counts) to Bright side (200 counts)
    frame = np.where(edge_line < center, 40.0, 200.0)
    return frame.astype(np.float64)

def simulate_optical_lens_blur(ideal_frame, blur_sigma=3.5):
    """
    Simulates physical optical degradation (aberrations, diffraction limits, 
    or vehicle jitter) by convolving the target scene with a Gaussian 
    Point Spread Function (PSF).
    """
    return gaussian_filter(ideal_frame, sigma=blur_sigma)

def calculate_mtf_via_edge_profile(blurred_frame):
    """
    Computes the Modulation Transfer Function (MTF) using the Slanted Edge method:
    1. Differentiate the Edge Profile to get the Line Spread Function (LSF).
    2. Execute a Fast Fourier Transform (FFT) on the LSF.
    3. Normalize the magnitude spectrum to 1.0 at DC (0 frequency).
    """
    # Grab a horizontal profile row across the center transition zone
    center_row = blurred_frame[int(blurred_frame.shape[0]/2), :]
    
    # Step 1: Differentiate the Edge Spread Function (ESF) to isolate the LSF
    # This acts as an impulse response tracking the true blur width
    lsf = np.abs(np.diff(center_row))
    
    # Pad the LSF to enhance frequency domain resolution interpolation steps
    padded_lsf = np.pad(lsf, (128, 128), 'constant')
    
    # Step 2: Compute the Forward Fast Fourier Transform (FFT)
    fft_complex = np.fft.rfft(padded_lsf)
    fft_magnitude = np.abs(fft_complex)
    
    # Step 3: Normalize the curve so contrast at zero spatial frequency equals 1.0
    mtf_curve = fft_magnitude / fft_magnitude[0]
    
    # Generate the normalized spatial frequency scale (cycles per pixel)
    freqs = np.fft.rfftfreq(len(padded_lsf), d=1.0)
    
    return center_row, lsf, freqs, mtf_curve

# =====================================================================
# =====================================================================
# 1. Initialize the scene and apply physical lens blur loop
ideal_edge = generate_synthetic_slanted_edge()
blurred_edge = simulate_optical_lens_blur(ideal_edge, blur_sigma=2.5)

# 2. Extract edge profiles and compute FFT contrast metrics
esf, lsf_profile, spatial_freqs, mtf_output = calculate_mtf_via_edge_profile(blurred_edge)

# 3. Initialize the 4-Panel Laboratory Dashboard Plot
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
fig.suptitle("Electro-Optical (EO) Lens System MTF & SFR Performance Analysis", fontsize=14, fontweight='bold')

# Panel 1: Ideal input target scene contrast
im1 = axes[0, 0].imshow(ideal_edge, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("1. Ideal ISO-12233 Slanted Edge Target")
fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# Panel 2: Blurred Sensor Ingest Frame Layer
im2 = axes[0, 1].imshow(blurred_edge, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title("2. Degraded Sensor Frame (Lens PSF Applied)")
fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Panel 3: Edge Spread Function (ESF) vs. Line Spread Function (LSF)
axes[1, 0].plot(esf, label='ESF (Edge Profile)', color='blue', linewidth=2)
ax_twin = axes[1, 0].twinx()
ax_twin.plot(lsf_profile, label='LSF (Impulse Blur Width)', color='red', linewidth=2, linestyle='--')
axes[1, 0].set_title("3. Spatial Derivative (ESF to LSF Transformation)")
axes[1, 0].set_xlabel("Pixel Index Across Transition Row")
axes[1, 0].set_ylabel("Pixel Counts (ESF Scale)")
ax_twin.set_ylabel("Delta Intensity (LSF Scale)")
# Combine legends safely across dual axes
lines, labels = axes[1, 0].get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
axes[1, 0].legend(lines + lines2, labels + labels2, loc='upper left')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: The Modulation Transfer Function (MTF) Contrast Curve
# Filter display up to the Nyquist limit limit (0.5 cycles/pixel)
nyquist_idx = np.where(spatial_freqs <= 0.5)[0][-1]
axes[1, 1].plot(spatial_freqs[:nyquist_idx], mtf_output[:nyquist_idx], color='green', linewidth=2.5)
# Draw the critical MTF50 metric lines (the standard target resolution threshold score)
axes[1, 1].axhline(y=0.5, color='black', linestyle=':', alpha=0.7)
mtf50_freq = spatial_freqs[np.where(mtf_output <= 0.5)[0][0]]
axes[1, 1].axvline(x=mtf50_freq, color='black', linestyle=':', alpha=0.7)
axes[1, 1].text(mtf50_freq + 0.02, 0.55, f"MTF50 = {mtf50_freq:.3f} cyc/px", fontsize=10, fontweight='bold')

axes[1, 1].set_title("4. System MTF Modulation Contrast Curve")
axes[1, 1].set_xlabel("Spatial Frequency (Cycles Per Pixel)")
axes[1, 1].set_ylabel("Modulation Contrast Factor (Normalized)")
axes[1, 1].set_xlim(0, 0.5)
axes[1, 1].set_ylim(0, 1.05)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("projects/mtf_sfr_sim/mtf_sfr_sim_plot.png", dpi=300)
plt.show()
