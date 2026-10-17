import gc
import numpy as np
import torch
from tqdm import tqdm


def compute_quick_batch_dsc(preds, masks, num_classes=2):
    """Kalkulasi cepat Mean DSC khusus 2 kanal biner untuk progress bar tqdm per batch."""
    dscs = []
    for c in range(num_classes):
        pred_c = preds[:, c] == 1 if preds.ndim == 4 else preds == c
        gt_c = masks[:, c] == 1 if masks.ndim == 4 else masks == c

        intersection = (pred_c & gt_c).sum().item()
        cardinality = pred_c.sum().item() + gt_c.sum().item()
        if cardinality > 0:
            dscs.append((2.0 * intersection) / (cardinality + 1e-6))

    return float(np.mean(dscs)) if dscs else 1.0


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    gpu_transform,
    amp_dtype,
    use_bf16,
    scaler,
    dora_magnitude_params,
    accumulation_steps=1,
):
    model.train()
    running_loss = 0.0
    running_dsc = 0.0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training", leave=False)

    trainable_params = [
        p
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    ]

    for i, batch in enumerate(pbar):
        # 1. ROBUST BATCH UNPACKING
        images, masks, dtms = None, None, None

        if isinstance(batch, (tuple, list)):
            images, masks = batch[0], batch[1]
            for elem in batch[2:]:
                if isinstance(elem, torch.Tensor) and elem.ndim >= 3:
                    dtms = elem
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
            masks = batch.get("mask", batch.get("masks"))
            dtms = batch.get("dtm", batch.get("dtms"), None)  # type: ignore
        else:
            raise TypeError(f"🚨 Tipe batch tidak dikenali: {type(batch)}")

        images = images.to(device, non_blocking=True)  # type: ignore
        masks = masks.to(device, non_blocking=True).float()  # type: ignore
        if dtms is not None:
            dtms = dtms.to(device, non_blocking=True).float()

        if masks.ndim == 4 and masks.shape[-1] == 2:
            masks = masks.permute(0, 3, 1, 2)
        if dtms is not None and dtms.ndim == 4 and dtms.shape[-1] == 2:
            dtms = dtms.permute(0, 3, 1, 2)

        # 2. SYNCHRONOUS GPU AUGMENTATION
        with torch.no_grad():
            if gpu_transform is not None:
                trans_out = (
                    gpu_transform(images, masks, dtms)
                    if dtms is not None
                    else gpu_transform(images, masks)
                )
                if isinstance(trans_out, (tuple, list)):
                    images, masks = trans_out[0], trans_out[1]
                    if len(trans_out) >= 3:
                        dtms = trans_out[2]
                else:
                    images = trans_out

        # 3. FORWARD PASS & LOSS CALCULATION
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(images)
            loss = (
                criterion(outputs, masks, dtm_targets=dtms)
                if dtms is not None
                else criterion(outputs, masks)
            )
            loss_scaled = loss / accumulation_steps

        # 🔍 SENSOR 1: INSPEKSI SANGAT DETAIL JIKA LOSS MENJADI NaN / Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n🚨 [NaN/Inf DETECTED AT BATCH {i}/{len(dataloader)}]")
            print(
                f"  ├─ Output Logits : min={outputs.min().item():.4f},"
                f" max={outputs.max().item():.4f},"
                f" has_nan={torch.isnan(outputs).any().item()}"
            )
            print(
                f"  ├─ Mask Target   : min={masks.min().item():.4f},"
                f" max={masks.max().item():.4f},"
                f" sum={masks.sum().item():.1f}"
            )
            if dtms is not None:
                print(
                    f"  ├─ DTM Target    : min={dtms.min().item():.4f},"
                    f" max={dtms.max().item():.4f},"
                    f" has_nan={torch.isnan(dtms).any().item()}"
                )
            print(
                f"  └─ Raw Loss Val  : {loss.item()} (Loss Scaled:"
                f" {loss_scaled.item()})"
            )

            # Lempar exception agar training langsung STOP sebelum optimizer merusak bobot
            raise ValueError(
                f"⛔ Pelatihan dihentikan otomatis: NaN Loss pada Batch {i}"
            )

        # 4. BACKWARD PASS
        if use_bf16 or scaler is None:
            loss_scaled.backward()
        else:
            scaler.scale(loss_scaled).backward()

        # 5. GRADIENT ACCUMULATION & STEP
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_bf16 or scaler is None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0
                )

                # 🔍 SENSOR 2: INSPEKSI GRADIENT NORM
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(
                        f"\n🚨 [NaN Gradient Norm Detected at Batch {i}] Grad Norm:"
                        f" {grad_norm}"
                    )
                    raise ValueError(
                        f"⛔ Pelatihan dihentikan: Gradient Norm NaN pada Batch {i}"
                    )

                optimizer.step()

                if dora_magnitude_params:
                    for p in dora_magnitude_params:
                        p.data.clamp_(min=1e-6)
            else:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0
                )

                # 🔍 SENSOR 2 (FP16): INSPEKSI GRADIENT NORM
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(
                        f"\n🚨 [NaN Gradient Norm Detected at Batch {i}] Grad Norm:"
                        f" {grad_norm}"
                    )
                    raise ValueError(
                        f"⛔ Pelatihan dihentikan: Gradient Norm NaN pada Batch {i}"
                    )

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                if scale_before <= scale_after and dora_magnitude_params:
                    for p in dora_magnitude_params:
                        p.data.clamp_(min=1e-6)

            optimizer.zero_grad(set_to_none=True)

        # 6. BATCH LOGGING
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        with torch.no_grad():
            probs = torch.sigmoid(outputs.detach().float())
            preds = (probs > 0.5).to(torch.uint8)
            batch_dsc = compute_quick_batch_dsc(preds, masks)

        running_dsc += batch_dsc * batch_size
        total_samples += batch_size

        if i % 10 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{(running_loss / total_samples):.4f}",
                    "dsc": f"{(running_dsc / total_samples):.4f}",
                }
            )

    return running_loss / total_samples, running_dsc / total_samples


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    gpu_transform,
    amp_dtype=torch.bfloat16,
    desc="Validation",
):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_true_masks = []
    all_pred_masks = []
    all_pred_probs = []

    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            images, masks, dtms = None, None, None

            if isinstance(batch, (tuple, list)):
                images, masks = batch[0], batch[1]
                for elem in batch[2:]:
                    if isinstance(elem, torch.Tensor) and elem.ndim >= 3:
                        dtms = elem
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
                masks = batch.get("mask", batch.get("masks"))
                dtms = batch.get("dtm", batch.get("dtms"), None)  # type: ignore
            else:
                raise TypeError(f"🚨 Tipe batch tidak dikenali: {type(batch)}")

            images = images.to(device, non_blocking=True)  # type: ignore
            masks = masks.to(device, non_blocking=True).float()  # type: ignore
            if dtms is not None:
                dtms = dtms.to(device, non_blocking=True).float()

            if masks.ndim == 4 and masks.shape[-1] == 2:
                masks = masks.permute(0, 3, 1, 2)
            if dtms is not None and dtms.ndim == 4 and dtms.shape[-1] == 2:
                dtms = dtms.permute(0, 3, 1, 2)

            if gpu_transform is not None:
                trans_out = (
                    gpu_transform(images, masks, dtms)
                    if dtms is not None
                    else gpu_transform(images, masks)
                )
                if isinstance(trans_out, (tuple, list)):
                    images, masks = trans_out[0], trans_out[1]
                    if len(trans_out) >= 3:
                        dtms = trans_out[2]
                else:
                    images = trans_out

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = (
                    criterion(outputs, masks, dtm_targets=dtms)
                    if dtms is not None
                    else criterion(outputs, masks)
                )

            probs = torch.sigmoid(outputs.float())
            preds = (probs > 0.5).to(torch.uint8)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_true_masks.append(masks.cpu().numpy().astype(np.uint8))
            all_pred_masks.append(preds.cpu().numpy().astype(np.uint8))
            all_pred_probs.append(probs.cpu().numpy().astype(np.float16))

            if i % 5 == 0 or (i + 1) == len(dataloader):
                pbar.set_postfix({"loss": f"{(running_loss / total_samples):.4f}"})

    epoch_loss = running_loss / total_samples

    y_true_masks = np.concatenate(all_true_masks, axis=0)
    y_pred_masks = np.concatenate(all_pred_masks, axis=0)
    y_pred_probs = np.concatenate(all_pred_probs, axis=0)

    del all_true_masks, all_pred_masks, all_pred_probs
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_loss, (y_true_masks, y_pred_masks, y_pred_probs)
