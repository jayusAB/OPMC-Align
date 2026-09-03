import torch
import torch.nn as nn
import torch.nn.functional as F


class OPTFA(nn.Module):
    def __init__(self, sinkhorn_iters=20, lamb=0.1, rho=0.5, eps=1e-6, split=4,
                 dust_mode='global', dust_tau=0.85, dust_offset=-0.05,
                 final_row_norm=True):
        super().__init__()
        self.iters = sinkhorn_iters
        self.lamb = lamb
        self.rho = rho
        self.eps = eps
        self.split = split
        self.dust_mode = dust_mode
        self.dust_tau = dust_tau
        self.dust_offset = dust_offset
        self.final_row_norm = final_row_norm
        # diagnostics, filled in by the last forward (plain floats/tensors)
        self.last_row_err = None
        self.last_col_err = None

    # ------------------------------------------------------------------ #
    #  Sinkhorn on the extended (dust-augmented) plan                     #
    # ------------------------------------------------------------------ #
    def _sinkhorn(self, cost, r, c, valid):
        r"""Entropic Sinkhorn with per-pair extended marginals.

        Returns:
            plan    (b, Bt, N+1, M+1): approximate optimal extended plan.
            row_err (b, Bt): L1 violation of the image-side marginal.
            col_err (b, Bt): L1 violation of the text-side  marginal.
        """
        P = torch.exp(-cost / self.lamb)
        P = P * valid
        P = P / (P.sum(dim=(-2, -1), keepdim=True) + self.eps)

        for _ in range(self.iters):
            u = P.sum(dim=-1) + self.eps                     # (b,Bt,N+1)
            P = P * (r / u).unsqueeze(-1)                    # row  norm, Eq. (10)
            v = P.sum(dim=-2) + self.eps                     # (b,Bt,M+1)
            P = P * (c / v).unsqueeze(-2)                    # col  norm, Eq. (11)

        if self.final_row_norm:
            # End on a row pass so that sum_m Omega[n,m] + Omega[n,dust] is
            # exactly alpha_n.  Without this, ||Omega||_1 (and therefore m)
            # depends on where the iteration happened to stop.
            u = P.sum(dim=-1) + self.eps
            P = P * (r / u).unsqueeze(-1)

        row_err = (P.sum(dim=-1) - r).abs().sum(dim=-1)      # (b,Bt)
        col_err = (P.sum(dim=-2) - c).abs().sum(dim=-1)      # (b,Bt)
        return P, row_err, col_err

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, img_local, txt_local, img_global, txt_global,
                img_mask=None, txt_mask=None):
        """Compute fine-grained transport score and unmatched degree.

        Returns:
            s_opt (Bi, Bt): fine-grained transport score S^{OPT}_{i,j}.
            m     (Bi, Bt): unmatched degree m_{i,j}.
        """
        Bi, N, D = img_local.shape
        Bt, M, _ = txt_local.shape

        img_local = F.normalize(img_local, dim=-1)
        txt_local = F.normalize(txt_local, dim=-1)
        img_global = F.normalize(img_global, dim=-1)
        txt_global = F.normalize(txt_global, dim=-1)

        if img_mask is None:
            img_mask = torch.ones(Bi, N, device=img_local.device, dtype=img_local.dtype)
        if txt_mask is None:
            txt_mask = torch.ones(Bt, M, device=txt_local.device, dtype=txt_local.dtype)
        img_mask = img_mask.to(img_local.dtype)
        txt_mask = txt_mask.to(txt_local.dtype)

        n_txt = txt_mask.sum(dim=-1).clamp(min=1.0)                    # (Bt,)

        s_opt = img_local.new_zeros(Bi, Bt, dtype=torch.float32)
        m = img_local.new_zeros(Bi, Bt, dtype=torch.float32)
        row_err = img_local.new_zeros(Bi, Bt, dtype=torch.float32)
        col_err = img_local.new_zeros(Bi, Bt, dtype=torch.float32)

        split = max(1, min(self.split, Bi))
        step = (Bi + split - 1) // split
        for beg in range(0, Bi, step):
            ed = min(beg + step, Bi)
            s_c, m_c, re_c, ce_c = self._pair_block(
                img_local[beg:ed], txt_local,
                img_global[beg:ed], txt_global,
                img_mask[beg:ed], txt_mask, n_txt)
            s_opt[beg:ed] = s_c
            m[beg:ed] = m_c
            row_err[beg:ed] = re_c
            col_err[beg:ed] = ce_c

        self.last_row_err = row_err.mean().detach()
        self.last_col_err = col_err.mean().detach()
        return s_opt, m

    def _pair_block(self, img_local, txt_local, img_global, txt_global,
                    img_mask, txt_mask, n_txt):
        """Extended OT for one image chunk against all texts."""
        b, N, D = img_local.shape
        Bt, M, _ = txt_local.shape

        # ---- local similarity Q and base cost C, Eqs. (5)-(6) ---------- #
        Q = torch.einsum('ind,jmd->ijnm', img_local, txt_local)        # (b,Bt,N,M)
        C = 1.0 - Q

        # ---- validity mask over the extended plan --------------------- #
        row_valid = torch.cat(
            [img_mask.unsqueeze(1).expand(b, Bt, N),
             img_mask.new_ones(b, Bt, 1)], dim=-1)                     # (b,Bt,N+1)
        col_valid = torch.cat(
            [txt_mask.unsqueeze(0).expand(b, Bt, M),
             txt_mask.new_ones(b, Bt, 1)], dim=-1)                     # (b,Bt,M+1)
        valid = row_valid.unsqueeze(-1) * col_valid.unsqueeze(-2)      # (b,Bt,N+1,M+1)

        # ---- assemble extended cost \hat C, Eq. (7) ------------------- #
        cost = C.new_zeros(b, Bt, N + 1, M + 1)
        cost[:, :, :N, :M] = C

        if self.dust_mode == 'global':
            # original formulation
            c_v2gt = 1.0 - torch.einsum('ind,jd->ijn', img_local, txt_global)
            c_gv2t = 1.0 - torch.einsum('id,jmd->ijm', img_global, txt_local)
            c_gg = 1.0 - torch.einsum('id,jd->ij', img_global, txt_global)
            cost[:, :, :N, M] = c_v2gt
            cost[:, :, N, :M] = c_gv2t
            cost[:, :, N, M] = c_gg
        else:
            if self.dust_mode == 'const':
                tau = C.new_full((b, Bt), float(self.dust_tau))
            elif self.dust_mode == 'adaptive':
                w = valid[:, :, :N, :M]
                tau = (C * w).sum(dim=(-2, -1)) / w.sum(dim=(-2, -1)).clamp(min=1.0)
                tau = tau + self.dust_offset
            else:
                raise ValueError(f"unknown dust_mode: {self.dust_mode}")
            cost[:, :, :N, M] = tau.unsqueeze(-1)
            cost[:, :, N, :M] = tau.unsqueeze(-1)
            # Dust-to-dust must be MORE expensive than dust-to-local, otherwise
            # the two dust nodes settle with each other, the text-side dust
            # column saturates at rho, and image local units have nowhere to
            # drain -- which is exactly the failure that pins m at its ceiling.
            cost[:, :, N, M] = 2.0 * tau

        n_img = img_mask.sum(dim=-1).clamp(min=1.0)                    # (b,)
        r = row_valid.clone()
        r[:, :, :N] = r[:, :, :N] * (1.0 / n_img).view(b, 1, 1)
        r[:, :, N] = self.rho
        c = col_valid.clone()
        c[:, :, :M] = c[:, :, :M] * (1.0 / n_txt).view(1, Bt, 1)
        c[:, :, M] = self.rho

    
        plan, row_err, col_err = self._sinkhorn(
            cost.float(), r.float(), c.float(), valid.float())

        Omega = plan[:, :, :N, :M]                                     # (b,Bt,N,M)
        s_opt = (Omega * Q.float()).sum(dim=(-2, -1))
        m = (1.0 - Omega.abs().sum(dim=(-2, -1))).clamp(min=0.0, max=1.0)
        return s_opt, m, row_err, col_err
