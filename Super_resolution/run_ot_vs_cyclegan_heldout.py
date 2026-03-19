"""
Compare OT vs CycleGAN coupling under proper held-out split.

Protocol:
- Generate truly unpaired data (A: HR domain, B: LR domain)
- Split first 50% for GAN pretraining, second 50% for FM training
- Evaluate on separate paired test set
- Run two modes on identical FM setup:
    1) OT coupling (unpaired)
    2) CycleGAN coupling (cyclegan)
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
import torch
import torch.nn.functional as F

from flow_matching import (
    Config,
    set_seed,
    get_device,
    VelocityUNet,
    compute_ssim_scores,
    compute_psnr,
    flow_matching_inference,
)
from test_truly_unpaired import (
    create_truly_unpaired_dataset,
    train_truly_unpaired,
)


def evaluate_model(model, lr_test, hr_test, config, device):
    model.eval()
    with torch.no_grad():
        sr = flow_matching_inference(model, lr_test, config, device)
    ssim = float(np.mean(compute_ssim_scores(sr, hr_test)))
    psnr = float(compute_psnr(sr, hr_test))
    return ssim, psnr


def run_experiment(name: str, coupling_mode: str, cyclegan_epochs: int, data, base_config: Config, device):
    hr_pretrain, lr_up_pretrain, hr_fm_train, lr_up_fm_train, hr_test, lr_test = data

    config = Config(**base_config.__dict__)
    config.coupling_mode = coupling_mode
    config.cyclegan_pretrain_epochs = cyclegan_epochs

    model = VelocityUNet(base_channels=config.base_channels).to(device)

    start = time.time()
    model = train_truly_unpaired(
        model,
        lr_up_pretrain,
        hr_pretrain,
        lr_up_fm_train,
        hr_fm_train,
        config,
        device,
    )
    train_time = time.time() - start

    torch.cuda.empty_cache()
    ssim, psnr = evaluate_model(model, lr_test, hr_test, config, device)

    return {
        'name': name,
        'coupling_mode': coupling_mode,
        'cyclegan_pretrain_epochs': cyclegan_epochs,
        'ssim': ssim,
        'psnr': psnr,
        'train_time_sec': train_time,
    }


def main():
    print('=' * 78)
    print('HELD-OUT SPLIT COMPARISON: OT vs CYCLEGAN')
    print('=' * 78)

    base_config = Config(
        n_images=300,
        epochs=40,
        batch_size=8,
        base_channels=32,
        fm_type='stochastic',
        sigma_max=0.02,
        ot_reg=0.01,
        inference_mode='ode',
        inference_steps=50,
        seed=42,
    )

    set_seed(base_config.seed)
    device = get_device()

    data = create_truly_unpaired_dataset(base_config, device)

    hr_pretrain, lr_up_pretrain, hr_fm_train, lr_up_fm_train, hr_test, lr_test = data
    bicubic_up = F.interpolate(lr_test, scale_factor=4, mode='bicubic', align_corners=False)
    bicubic_ssim = float(np.mean(compute_ssim_scores(bicubic_up, hr_test)))
    bicubic_psnr = float(compute_psnr(bicubic_up, hr_test))

    results = []

    print('\nRunning OT (unpaired) coupling...')
    results.append(
        run_experiment(
            name='OT coupling',
            coupling_mode='unpaired',
            cyclegan_epochs=0,
            data=data,
            base_config=base_config,
            device=device,
        )
    )

    print('\nRunning CycleGAN coupling (20 pretrain epochs)...')
    results.append(
        run_experiment(
            name='CycleGAN coupling',
            coupling_mode='cyclegan',
            cyclegan_epochs=20,
            data=data,
            base_config=base_config,
            device=device,
        )
    )

    print('\nRunning CycleGAN coupling (30 pretrain epochs)...')
    results.append(
        run_experiment(
            name='CycleGAN coupling (30e)',
            coupling_mode='cyclegan',
            cyclegan_epochs=30,
            data=data,
            base_config=base_config,
            device=device,
        )
    )

    print('\nRunning CycleGAN coupling (40 pretrain epochs)...')
    results.append(
        run_experiment(
            name='CycleGAN coupling (40e)',
            coupling_mode='cyclegan',
            cyclegan_epochs=40,
            data=data,
            base_config=base_config,
            device=device,
        )
    )

    print('\nRunning CycleGAN coupling (60 pretrain epochs)...')
    results.append(
        run_experiment(
            name='CycleGAN coupling (60e)',
            coupling_mode='cyclegan',
            cyclegan_epochs=60,
            data=data,
            base_config=base_config,
            device=device,
        )
    )

    print('\n' + '=' * 78)
    print('RESULTS')
    print('=' * 78)
    print(f"{'Method':<22} {'SSIM':<10} {'PSNR(dB)':<10} {'Time(s)':<10} {'ΔSSIM vs Bicubic':<18}")
    print('-' * 78)
    print(f"{'Bicubic baseline':<22} {bicubic_ssim:<10.4f} {bicubic_psnr:<10.2f} {'-':<10} {0.0:<18.4f}")
    for r in results:
        delta_ssim = r['ssim'] - bicubic_ssim
        print(f"{r['name']:<22} {r['ssim']:<10.4f} {r['psnr']:<10.2f} {r['train_time_sec']:<10.1f} {delta_ssim:<18.4f}")

    best = max(results, key=lambda x: x['ssim'])
    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)
    print(f"Best learned method: {best['name']} (SSIM={best['ssim']:.4f})")
    print(f"Bicubic baseline: SSIM={bicubic_ssim:.4f}")

    if best['ssim'] > bicubic_ssim:
        print('At least one learned coupling beats bicubic on held-out split.')
    else:
        print('Neither learned coupling beats bicubic on held-out split.')


if __name__ == '__main__':
    main()
