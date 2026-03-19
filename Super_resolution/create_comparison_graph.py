"""
Create comparison graph: CycleGAN pretraining epochs vs SSIM/PSNR
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the held-out split benchmark
cyclegan_epochs = [20, 30, 40, 60]
cyclegan_ssim = [0.8286, 0.8432, 0.8524, 0.8293]
cyclegan_psnr = [27.46, 28.28, 28.43, 26.38]

ot_ssim = 0.8575
ot_psnr = 28.40
bicubic_ssim = 0.8698
bicubic_psnr = 28.24

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: SSIM vs CycleGAN Pretraining Epochs
ax1.plot(cyclegan_epochs, cyclegan_ssim, 'o-', linewidth=2.5, markersize=8, 
         label='CycleGAN', color='#2E86AB')
ax1.axhline(y=ot_ssim, color='#A23B72', linestyle='--', linewidth=2.5, label='OT coupling')
ax1.axhline(y=bicubic_ssim, color='#F18F01', linestyle=':', linewidth=2.5, label='Bicubic baseline')

# Highlight best CycleGAN point
best_idx = np.argmax(cyclegan_ssim)
ax1.plot(cyclegan_epochs[best_idx], cyclegan_ssim[best_idx], 'D', markersize=12, 
         color='#06A77D', zorder=5)

ax1.set_xlabel('CycleGAN Pretraining Epochs', fontsize=12, fontweight='bold')
ax1.set_ylabel('SSIM', fontsize=12, fontweight='bold')
ax1.set_title('SSIM Comparison: CycleGAN Pretraining Epochs vs OT & Bicubic', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xticks(cyclegan_epochs)
ax1.set_ylim([0.80, 0.88])

# Add value annotations
for i, (ep, ssim) in enumerate(zip(cyclegan_epochs, cyclegan_ssim)):
    delta = ssim - ot_ssim
    ax1.annotate(f'{ssim:.4f}\n({delta:+.4f})', xy=(ep, ssim), 
                xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)

# Plot 2: PSNR vs CycleGAN Pretraining Epochs
ax2.plot(cyclegan_epochs, cyclegan_psnr, 's-', linewidth=2.5, markersize=8, 
         label='CycleGAN', color='#2E86AB')
ax2.axhline(y=ot_psnr, color='#A23B72', linestyle='--', linewidth=2.5, label='OT coupling')
ax2.axhline(y=bicubic_psnr, color='#F18F01', linestyle=':', linewidth=2.5, label='Bicubic baseline')

# Highlight best CycleGAN point
best_idx_psnr = np.argmax(cyclegan_psnr)
ax2.plot(cyclegan_epochs[best_idx_psnr], cyclegan_psnr[best_idx_psnr], 'D', markersize=12, 
         color='#06A77D', zorder=5)

ax2.set_xlabel('CycleGAN Pretraining Epochs', fontsize=12, fontweight='bold')
ax2.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
ax2.set_title('PSNR Comparison: CycleGAN Pretraining Epochs vs OT & Bicubic', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='lower right')
ax2.set_xticks(cyclegan_epochs)

# Add value annotations
for i, (ep, psnr) in enumerate(zip(cyclegan_epochs, cyclegan_psnr)):
    delta = psnr - ot_psnr
    ax2.annotate(f'{psnr:.2f}\n({delta:+.2f})', xy=(ep, psnr), 
                xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('CycleGAN_vs_OT_Comparison.png', dpi=300, bbox_inches='tight')
print("Graph saved: CycleGAN_vs_OT_Comparison.png")
plt.show()

# Create summary table
print("\n" + "="*80)
print("HELD-OUT SPLIT COMPARISON: CycleGAN PRETRAINING EPOCHS VS OT & BICUBIC")
print("="*80)
print(f"{'Method':<30} {'SSIM':<10} {'ΔSSIM vs OT':<15} {'PSNR (dB)':<10} {'ΔPSNR vs OT':<15}")
print("-"*80)
print(f"{'Bicubic baseline':<30} {bicubic_ssim:<10.4f} {bicubic_ssim-ot_ssim:<15.4f} {bicubic_psnr:<10.2f} {bicubic_psnr-ot_psnr:<15.2f}")
print(f"{'OT coupling':<30} {ot_ssim:<10.4f} {0.0:<15.4f} {ot_psnr:<10.2f} {0.0:<15.2f}")
print("-"*80)
for ep, ssim, psnr in zip(cyclegan_epochs, cyclegan_ssim, cyclegan_psnr):
    method_name = f"CycleGAN ({ep}e pretrain)"
    print(f"{method_name:<30} {ssim:<10.4f} {ssim-ot_ssim:<15.4f} {psnr:<10.2f} {psnr-ot_psnr:<15.2f}")

print("="*80)
print("\nKEY FINDINGS:")
print("="*80)
best_cg_idx = np.argmax(cyclegan_ssim)
best_cg_ssim = cyclegan_ssim[best_cg_idx]
best_cg_epochs = cyclegan_epochs[best_cg_idx]
print(f"✓ Best CycleGAN: {best_cg_epochs} epochs → SSIM={best_cg_ssim:.4f}")
print(f"  - Gap to OT: {best_cg_ssim-ot_ssim:+.4f} SSIM")
print(f"  - This is {abs(best_cg_ssim-ot_ssim)/ot_ssim*100:.2f}% difference")
print(f"\n✓ CycleGAN-40e achieves PSNR {cyclegan_psnr[2]:.2f}, BETTER than OT {ot_psnr:.2f}")
print(f"\n⚠️  CycleGAN-60e shows overfitting: SSIM drops to {cyclegan_ssim[-1]:.4f}")
print(f"    Optimal appears to be around 40 epochs for this 150-sample dataset")
