"""Attention-driven cross-modal network (adaptation of Att-CMIR-Net to EuroSAT RGB <-> SAR).

Followed from the paper (Chaudhuri et al., IGARSS 2021):
  * two ImageNet-pretrained VGG-16 convolutional encoders (one per modality);
  * attention = global-average-pool(last conv map of the *attention-source* encoder)
    -> fully connected layer -> sigmoid, multiplied onto the other modality's FC features;
  * a shared 128-d embedding space; classifier / cross-decoders on that space.

Adaptations / assumptions (NOT stated in the paper) are marked ASSUMPTION below and are
listed in the README and report.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGEncoder(nn.Module):
    """VGG-16 convolutional trunk (incl. final max-pool).

    With 128x128 inputs the output map is 512 x 4 x 4 (the paper's 7x7x512 corresponds to
    224x224 inputs). The first conv layer is rebuilt for ``in_channels != 3``: ImageNet
    weights are averaged over RGB and tiled across the new channels (rescaled by 3/C so the
    expected activation magnitude is preserved).
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import vgg16

        if pretrained:
            try:
                from torchvision.models import VGG16_Weights

                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except ImportError:  # very old torchvision
                vgg = vgg16(pretrained=True)
        else:
            vgg = vgg16(weights=None)
        feats = vgg.features
        if in_channels != 3:
            old = feats[0]
            new = nn.Conv2d(in_channels, old.out_channels, kernel_size=3, padding=1)
            if pretrained:
                with torch.no_grad():
                    w = old.weight.mean(dim=1, keepdim=True)  # (64,1,3,3)
                    new.weight.copy_(w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
                    new.bias.copy_(old.bias)
            feats[0] = new
        self.features = feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class FCHead(nn.Module):
    """Global average pooling -> FC -> ReLU -> dropout (the 'FCN' blocks of Fig. 1)."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        g = fmap.mean(dim=(2, 3))
        return self.drop(F.relu(self.fc(g), inplace=False))


class AttCMIRNet(nn.Module):
    """RGB <-> SAR cross-modal retrieval network.

    Training forward (``forward``): needs a paired (rgb, sar) batch because the SAR encoder
    produces the attention that modulates the RGB features.

    Inference (``embed_rgb`` / ``embed_sar``): each modality is embedded independently, so a
    query and gallery items never require a counterpart.  # ASSUMPTION / deviation: the paper
    does not say how a photo is attended at inference when no sketch is paired with it.
    """

    def __init__(self, num_classes: int, emb_dim: int = 128, hidden_dim: int = 512,
                 dropout: float = 0.3, pretrained: bool = True, sar_channels: int = 2,
                 input_size: int = 128, attention: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sar_channels = sar_channels
        self.input_size = input_size
        self.use_attention = attention

        # --- input normalisation (buffers => stored in the checkpoint) -------------------
        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("sar_lo", torch.full((1, sar_channels, 1, 1), -1e30))
        self.register_buffer("sar_hi", torch.full((1, sar_channels, 1, 1), 1e30))
        self.register_buffer("sar_mean", torch.zeros(1, sar_channels, 1, 1))
        self.register_buffer("sar_std", torch.ones(1, sar_channels, 1, 1))

        # --- encoders + FC heads ----------------------------------------------------------
        self.enc_rgb = VGGEncoder(3, pretrained)
        self.enc_sar = VGGEncoder(sar_channels, pretrained)
        self.head_rgb = FCHead(512, hidden_dim, dropout)
        self.head_sar = FCHead(512, hidden_dim, dropout)

        # --- sketch-style attention: GAP -> FC -> sigmoid (SAR is the attention source) ----
        self.att_fc = nn.Linear(512, hidden_dim)

        # --- shared "unified representation" network -> 128-d space ----------------------
        # ASSUMPTION: a 2-layer MLP with weights shared by both modalities.
        self.unified = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(inplace=False), nn.Linear(256, emb_dim)
        )
        # Classifier for L2 (shared across modalities) -- ASSUMPTION.
        self.classifier = nn.Linear(emb_dim, num_classes)
        # Cross-decoders for L4 (w_ab, w_ba in Eq. 4) -- ASSUMPTION: single linear maps.
        self.dec_ab = nn.Linear(emb_dim, emb_dim)
        self.dec_ba = nn.Linear(emb_dim, emb_dim)

    # ------------------------------------------------------------------ preprocessing --
    def set_sar_stats(self, stats: dict) -> None:
        def _t(v):
            return torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)

        with torch.no_grad():
            self.sar_lo.copy_(_t(stats["lo"]))
            self.sar_hi.copy_(_t(stats["hi"]))
            self.sar_mean.copy_(_t(stats["mean"]))
            self.sar_std.copy_(_t(stats["std"]))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                              align_corners=False)
        return x

    def prep_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """uint8 or float in [0, 255], shape (B,3,H,W)."""
        x = rgb.float() / 255.0
        x = (x - self.rgb_mean) / self.rgb_std
        return self._resize(x)

    def prep_sar(self, sar: torch.Tensor) -> torch.Tensor:
        """Raw float SAR values, shape (B,C,H,W)."""
        x = torch.nan_to_num(sar.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.minimum(torch.maximum(x, self.sar_lo), self.sar_hi)
        x = (x - self.sar_mean) / self.sar_std
        return self._resize(x)

    # ---------------------------------------------------------------------- inference --
    def _z_rgb(self, rgb):
        fmap = self.enc_rgb(self.prep_rgb(rgb))
        return self.head_rgb(fmap)

    def _sar_maps(self, sar):
        fmap = self.enc_sar(self.prep_sar(sar))
        return fmap, self.head_sar(fmap)

    def embed_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Pair-free RGB embedding (B, emb_dim)."""
        return self.unified(self._z_rgb(rgb))

    def embed_sar(self, sar: torch.Tensor) -> torch.Tensor:
        """Pair-free SAR embedding (B, emb_dim)."""
        return self.unified(self._sar_maps(sar)[1])

    # ---------------------------------------------------------------------- training ---
    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> dict:
        z_r = self._z_rgb(rgb)
        fmap_s, z_s = self._sar_maps(sar)

        v_sar = self.unified(z_s)
        v_rgb_plain = self.unified(z_r)
        if self.use_attention:
            g_s = fmap_s.mean(dim=(2, 3))                    # GAP of last conv map
            att = torch.sigmoid(self.att_fc(g_s))            # (B, hidden_dim) in (0,1)
            v_rgb = self.unified(z_r * att)                  # attended RGB features
        else:
            att = None
            v_rgb = v_rgb_plain

        out = {
            "v_rgb": v_rgb,                 # V_a  (attended when attention is on)
            "v_sar": v_sar,                 # V_b
            "v_rgb_plain": v_rgb_plain,     # pair-free path used at inference
            "att": att,
            "logits_rgb": self.classifier(v_rgb),
            "logits_sar": self.classifier(v_sar),
            "logits_rgb_plain": self.classifier(v_rgb_plain),
            "dec_a2b": self.dec_ab(v_rgb),  # w_ab V_a  (should reconstruct V_b)
            "dec_b2a": self.dec_ba(v_sar),  # w_ba V_b  (should reconstruct V_a)
        }
        return out


def build_model(cfg: dict, num_classes: int, pretrained: bool | None = None) -> AttCMIRNet:
    m = cfg["model"]
    if pretrained is None:
        pretrained = bool(m.get("pretrained", True))
    return AttCMIRNet(
        num_classes=num_classes,
        emb_dim=int(m["emb_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        dropout=float(m["dropout"]),
        pretrained=pretrained,
        sar_channels=int(cfg["data"]["sar_channels"]),
        input_size=int(m["input_size"]),
        attention=bool(m["attention"]),
    )
