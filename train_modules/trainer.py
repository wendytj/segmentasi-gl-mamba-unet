import numpy as np
import gc
import torch
from tqdm import tqdm


def compute_quick_batch_dsc(preds, masks, num_classes=2):
    """Kalkulasi cepat Mean DSC khusus 2 kanal biner (Kanal 0: Ischemia, Kanal 1: Bleeding)

    untuk monitoring progress bar tqdm per batch.
    """
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
    total_batches = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training", leave=False)

    trainable_params = [p for group in optimizer.param_groups for p in group["params"]]

    for i, batch in enumerate(pbar):
        # ======================================================================
        # 1. ROBUST ITERATIVE BATCH UNPACKING (Auto-detect DTM & Labels)
        # ======================================================================
        images, masks, dtms, labels = None, None, None, None

        if isinstance(batch, (tuple, list)):
            images = batch[0]
            masks = batch[1]

            # Pindai seluruh elemen sisa (batch[2:]) untuk mencari DTM & Labels
            for elem in batch[2:]:
                if isinstance(elem, torch.Tensor):
                    if elem.ndim >= 3:  # DTM Map berdimensi 3D/4D [B, C, H, W]
                        dtms = elem
                    elif elem.ndim <= 2:  # Classification Label [B] atau [B, 1]
                        labels = elem

        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
            masks = batch.get("mask", batch.get("masks"))
            dtms = batch.get("dtm", batch.get("dtms"), None)  # type: ignore
            labels = batch.get("label", batch.get("labels"), None)  # type: ignore
        else:
            raise TypeError(f"🚨 Tipe batch tidak dikenali: {type(batch)}")

        images = images.to(device, non_blocking=True)  # type: ignore
        masks = masks.to(device, non_blocking=True).float()  # type: ignore # [B, 2, H, W]

        if dtms is not None:
            dtms = dtms.to(device, non_blocking=True).float()  # [B, 2, H, W]

        # Ensure correct channel-first format [B, C, H, W]
        if masks.ndim == 4 and masks.shape[-1] == 2:
            masks = masks.permute(0, 3, 1, 2)
        if dtms is not None and dtms.ndim == 4 and dtms.shape[-1] == 2:
            dtms = dtms.permute(0, 3, 1, 2)

        # ======================================================================
        # 2. SYNCHRONOUS GPU AUGMENTATION
        # ======================================================================
        with torch.no_grad():
            if gpu_transform is not None:
                trans_out = (
                    gpu_transform(images, masks, dtms)
                    if dtms is not None
                    else gpu_transform(images, masks)
                )
                if isinstance(trans_out, (tuple, list)):
                    if len(trans_out) >= 3:
                        images, masks, dtms = trans_out[:3]
                    elif len(trans_out) == 2:
                        images, masks = trans_out
                else:
                    images = trans_out

        # ======================================================================
        # 3. FORWARD PASS & LOSS CALCULATION WITH EXPLICIT DTM PASSING
        # ======================================================================
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(images)  # Shape: [B, 2, H, W]

            if dtms is not None:
                try:
                    loss = criterion(outputs, masks, dtm_targets=dtms)
                except TypeError:
                    try:
                        loss = criterion(outputs, masks, dtms)
                    except TypeError:
                        loss = criterion(outputs, masks)
            else:
                loss = criterion(outputs, masks)

            loss_scaled = loss / accumulation_steps

        # 4. Backward Pass
        if use_bf16:
            loss_scaled.backward()
        else:
            scaler.scale(loss_scaled).backward()

        # 5. Gradient Accumulation & Step
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_bf16:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                if dora_magnitude_params:
                    for p in dora_magnitude_params:
                        p.data.clamp_(min=1e-6)
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                if scale_before <= scale_after and dora_magnitude_params:
                    for p in dora_magnitude_params:
                        p.data.clamp_(min=1e-6)

            optimizer.zero_grad(set_to_none=True)

        # 6. Batch Statistics Logging
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        with torch.no_grad():
            probs = torch.sigmoid(outputs.detach())
            preds = (probs > 0.5).to(torch.uint8)  # Shape: [B, 2, H, W]
            batch_dsc = compute_quick_batch_dsc(preds, masks)

        running_dsc += batch_dsc * batch_size
        total_batches += batch_size

        if i % 10 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{(running_loss / total_batches):.4f}",
                    "dsc": f"{(running_dsc / total_batches):.4f}",
                }
            )

    epoch_loss = running_loss / total_batches
    epoch_dsc = running_dsc / total_batches
    return epoch_loss, epoch_dsc


def evaluate(model, dataloader, criterion, device, gpu_transform, desc="Validation"):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_true_masks = []
    all_pred_masks = []
    all_pred_probs = []

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            images, masks, dtms, labels = None, None, None, None

            if isinstance(batch, (tuple, list)):
                images = batch[0]
                masks = batch[1]

                for elem in batch[2:]:
                    if isinstance(elem, torch.Tensor):
                        if elem.ndim >= 3:
                            dtms = elem
                        elif elem.ndim <= 2:
                            labels = elem

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
                    if len(trans_out) >= 3:
                        images, masks, dtms = trans_out[:3]
                    elif len(trans_out) == 2:
                        images, masks = trans_out
                else:
                    images = trans_out

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)  # Shape: [B, 2, H, W]
                if dtms is not None:
                    try:
                        loss = criterion(outputs, masks, dtm_targets=dtms)
                    except TypeError:
                        try:
                            loss = criterion(outputs, masks, dtms)
                        except TypeError:
                            loss = criterion(outputs, masks)
                else:
                    loss = criterion(outputs, masks)

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
