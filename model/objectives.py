import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)


    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2


def _lmc_normalize(x, mask, mode="minmax", eps=1e-6):
    r"""In-batch normalisation of the local calibration state.

    Normalisation statistics are computed over ``mask`` only, so the positive
    branch is not contaminated by negative-pair statistics and vice versa.
    Entries outside ``mask`` are returned as 0 (they end up with lambda = 1 and
    are masked out of the log-sum-exp anyway).

    Args:
        x    (B, B): raw local calibration state r.
        mask (B, B): bool mask selecting the pairs this branch applies to.
        mode (str):  'minmax'  -> [0, 1], keeps lambda >= 1 (paper semantics);
                     'zscore'  -> zero mean / unit std, allows lambda < 1;
                     'none'    -> passthrough (reproduces the original behaviour).
    """
    if mode == "none":
        return x * mask.to(x.dtype)
    if mask.sum() < 2:
        return torch.zeros_like(x)

    if mode == "minmax":
        lo = x[mask].min()
        hi = x[mask].max()
        out = (x - lo) / (hi - lo + eps)
    elif mode == "zscore":
        v = x[mask]
        out = (x - v.mean()) / (v.std() + eps)
    else:
        raise ValueError(f"unknown lmc_norm: {mode}")
    return out * mask.to(x.dtype)


def compute_wbcmcir(image_features, text_features, pid, s_opt, m, logit_scale,
                    delta_p=0.60, delta_n=0.20, circle_m=0.15, gamma=None,
                    lmc_eta=2.0, lmc_norm="minmax", pos_state="m",
                    epsilon=1e-8, return_stats=False):
    r"""Local-Matching-Calibrated bidirectional cross-modal Circle loss.

    Args:
        image_features (B, D): global image features g^v.
        text_features  (B, D): global text  features g^t.
        pid            (B,):   identity labels shared by image / text.
        s_opt          (B, B): fine-grained transport score S^{OPT}_{i,j} (Eq. 14).
        m              (B, B): unmatched degree m_{i,j}                   (Eq. 14).
        logit_scale:           1/temperature; used as gamma unless ``gamma`` is given.
        delta_p (float):       positive similarity target. MUST be reachable —
                               set it to (measured positive cosine + 0.05~0.10).
        delta_n (float):       negative similarity target.
        circle_m (float):      relaxation margin, O_p = delta_p + m, O_n = delta_n - m.
        gamma (float | None):  Circle scale factor; None -> logit_scale.
        lmc_eta (float):       dynamic range of the calibration weight,
                               lambda in [1, 1+eta] under 'minmax'.
        lmc_norm (str):        'minmax' | 'zscore' | 'none'.
        pos_state (str):       how r^+ is formed from (m, S^OPT):
                               'm'      -> r^+ = m * [S^OPT]_+   (Eq. 15, default)
                               'm_only' -> r^+ = m
                               's_only' -> r^+ = [S^OPT]_+
        return_stats (bool):   also return a dict of diagnostics (see below).

    Returns:
        loss, or (loss, stats) when ``return_stats`` is True.
    """
    B = image_features.shape[0]
    pid = pid.reshape((B, 1))
    pos_mask = (pid - pid.t()) == 0                 # (B, B) bool
    neg_mask = ~pos_mask

    # ---- global cosine similarity s^g_{i,j}, Eq. (4) --------------------- #
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)
    s_g = image_norm @ text_norm.t()                # s_g[i, j] = <image_i, text_j>

    if gamma is None:
        gamma = logit_scale

    # ---- Step 1: decoupled Circle targets, Eqs. (18)-(19) ---------------- #
    # O_p = delta_p + circle_m ,  O_n = delta_n - circle_m
    alpha_p = torch.clamp_min(delta_p + circle_m - s_g, min=0.0).detach()
    alpha_n = torch.clamp_min(s_g - delta_n + circle_m, min=0.0).detach()
    logit_p = -gamma * alpha_p * (s_g - delta_p)    # positive pairs
    logit_n = gamma * alpha_n * (s_g - delta_n)     # negative pairs

    # ---- Step 3: normalised calibration weights, Eqs. (15)-(17) ---------- #
    # No gradient flows through the weight branch, so the model cannot cheat by
    # shrinking S^OPT / m to reduce the weighted loss.
    with torch.no_grad():
        s_opt_pos = s_opt.detach().float().clamp(min=0.0)
        m_det = m.detach().float().clamp(min=0.0, max=1.0)

        if pos_state == "m":
            r_pos = m_det * s_opt_pos               # Eq. (15)
        elif pos_state == "m_only":
            r_pos = m_det
        elif pos_state == "s_only":
            r_pos = s_opt_pos
        else:
            raise ValueError(f"unknown pos_state: {pos_state}")
        r_neg = s_opt_pos                           # Eq. (16)

        r_pos = _lmc_normalize(r_pos, pos_mask, lmc_norm)
        r_neg = _lmc_normalize(r_neg, neg_mask, lmc_norm)

        # lambda = 1 + eta * r_hat ,  Eq. (17); clamp guards zscore mode where
        # r_hat can be strongly negative.
        lam_pos = (1.0 + lmc_eta * r_pos).clamp(min=0.05)
        lam_neg = (1.0 + lmc_eta * r_neg).clamp(min=0.05)
        log_lam_pos = torch.log(lam_pos + epsilon).to(logit_p.dtype)
        log_lam_neg = torch.log(lam_neg + epsilon).to(logit_n.dtype)

    # ---- calibrated bidirectional aggregation, Eqs. (18)-(20) ------------ #
    neg_inf = torch.finfo(logit_p.dtype).min
    lp = (logit_p + log_lam_pos).masked_fill(neg_mask, neg_inf)
    ln = (logit_n + log_lam_neg).masked_fill(pos_mask, neg_inf)

    # t2i: for each text query j aggregate over images i  -> reduce dim 0
    # i2t: for each image query i aggregate over texts j  -> reduce dim 1
    loss_t2i = F.softplus(torch.logsumexp(ln, dim=0) + torch.logsumexp(lp, dim=0)).mean()
    loss_i2t = F.softplus(torch.logsumexp(ln, dim=1) + torch.logsumexp(lp, dim=1)).mean()
    loss = 0.5 * (loss_t2i + loss_i2t)              # Eq. (20)

    if not return_stats:
        return loss

    # ---- diagnostics ------------------------------------------------------ #
    # calib_ratio_* is the key number: in-batch std of the calibration term
    # divided by in-batch std of the Circle logit it modulates.  Below ~0.1 the
    # calibration is doing nothing and lmc_eta must go up.
    with torch.no_grad():
        lse_p = torch.logsumexp(lp, dim=1).mean()
        lse_n = torch.logsumexp(ln, dim=1).mean()
        sp_p, sp_n = logit_p[pos_mask].std(), logit_n[neg_mask].std()
        sl_p, sl_n = log_lam_pos[pos_mask].std(), log_lam_neg[neg_mask].std()
        stats = {
            "pos_cos": s_g[pos_mask].mean(),
            "neg_cos": s_g[neg_mask].mean(),
            "lse_pos": lse_p,                       # want lse_pos ~ lse_neg
            "lse_neg": lse_n,
            "calib_ratio_pos": sl_p / (sp_p + epsilon),
            "calib_ratio_neg": sl_n / (sp_n + epsilon),
            "s_opt_pos": s_opt.detach().float()[pos_mask].mean(),
            "s_opt_neg": s_opt.detach().float()[neg_mask].mean(),
            "m_mean": m.detach().float().mean(),
            "m_std": m.detach().float().std(),
        }
    return loss, stats


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss
