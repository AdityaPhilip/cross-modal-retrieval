#!/usr/bin/env python
"""Build reports/report.pdf from the artefacts produced by train.py / test.py.

    python make_report.py --group "Group X: Name1, Name2"

Every number in the report is read from JSON files written by real runs
(reports/train_summary.json, reports/test_results.json, plots/history.json and, optionally, the
no-attention ablation in reports/no_attention/). Nothing is hard-coded except the paper's
*published* values, which are labelled as such. If a results file is missing, the corresponding
section says so instead of inventing content.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils import load_json, resolve_path

PAPER = {  # published in Chaudhuri et al., IGARSS 2021 (Earth on Canvas, photo-sketch). NOT our results.
    "Sketch to Photo": (0.753, 0.784), "Photo to Sketch": (0.723, 0.745),
    "Sketch to Sketch": (0.775, 0.788), "Photo to Photo": (0.804, 0.823),
}


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def build(args):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rd = resolve_path("reports")

    def maybe(path):
        p = resolve_path(path)
        return load_json(p) if p.exists() else None

    summary = maybe(args.train_summary)
    test = maybe(args.results)
    abl_test = maybe(args.ablation_results)
    abl_sum = maybe(args.ablation_summary)
    plot_dir = resolve_path(args.plot_dir)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
    h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
    title = ParagraphStyle("title", parent=ss["Title"], fontSize=17, leading=21)

    def P(t, st=body):
        return Paragraph(t, st)

    def table(rows, widths, header=True, wrap=False):
        data = [[Paragraph(str(c), cell) if wrap else str(c) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                 ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7"))]
            if not wrap:
                style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    def clean(name):
        return name.replace("->", " to ")

    bs_txt = (f"{summary['config']['train']['batch_size']} (6 GB GPU), mixed precision"
              if summary else "see train_summary.json")
    ep_txt = (f"{summary['epochs_run']} (best epoch {summary['best']['epoch']}, chosen on validation mAP)"
              if summary else "see train_summary.json")
    s = []
    s += [P("Attention-Driven Cross-Modal Remote Sensing Image Retrieval", title),
          P("Re-implementation and adaptation to EuroSAT RGB and SAR imagery", ParagraphStyle(
              "sub", parent=body, alignment=1, fontSize=11)),
          P(f"ECPE49 Foundations of Artificial Intelligence, NIT Tiruchirappalli (July 2026 session). "
            f"{args.group}", ParagraphStyle("who", parent=body, alignment=1, fontSize=9)), Spacer(1, 6)]

    s += [P("1. Introduction and objective", h1),
          P("Satellites observe the same scene with different sensors (optical RGB, multispectral, SAR, ...). "
            "Cross-modal retrieval takes a query from one modality (M1) and returns semantically relevant images "
            "(same land-use class) from a gallery of another modality (M2). The assignment asks us to implement an "
            "assigned research paper and reproduce its results. This report documents our implementation of "
            "<i>Attention-Driven Cross-Modal Remote Sensing Image Retrieval</i> (Chaudhuri, Banerjee, Bhattacharya, "
            "Datcu, IGARSS 2021), adapted to the datasets named in the assignment sheet: EuroSAT RGB and EuroSAT-SAR. "
            "The retrieval directions evaluated are RGB to SAR and SAR to RGB, measured by mean average precision (mAP).")]

    s += [P("2. Original paper and methodology", h1),
          P("The paper (Att-CMIR-Net) extends CMIR-Net. Two ImageNet-pretrained VGG-16 encoders process the two "
            "modalities (photo and hand-drawn sketch). The last convolutional map of the <i>sketch</i> encoder is "
            "globally average-pooled, passed through a fully connected layer and a sigmoid; the resulting vector "
            "multiplies the fully connected features of the <i>photo</i> encoder (attention). Both streams are mapped "
            "to a shared 128-d space trained with four losses: L<sub>1</sub> = ||V<sub>a</sub> - V<sub>b</sub>||<super>2</super> "
            "(alignment), L<sub>2</sub> = CE(V<sub>ab</sub>) (classification), L<sub>3</sub> = ||V<sub>a</sub>||<super>2</super> + "
            "||V<sub>b</sub>||<super>2</super> (norm) and L<sub>4</sub> = ||w<sub>ab</sub>V<sub>a</sub> - V<sub>b</sub>||<super>2</super> + "
            "||w<sub>ba</sub>V<sub>b</sub> - V<sub>a</sub>||<super>2</super> (cross-decoders); L = L<sub>1</sub>+L<sub>2</sub>+L<sub>3</sub>+L<sub>4</sub>, "
            "optimised with SGD (lr 0.001, batch 128, about 40-50 epochs). Retrieval ranks the gallery by Euclidean distance. "
            "The paper evaluates on Earth on Canvas (14 classes, 100 photo and 100 sketch images per class, 70:30 split)."),
          P("The values below are the paper's <b>published</b> results on a different dataset and task (photo/sketch). "
            "They are quoted for context only and are not results of this project.", body),
          table([["Direction (published, Earth on Canvas)", "mAP", "P@10"]] +
                [[k, f"{v[0]:.3f}", f"{v[1]:.3f}"] for k, v in PAPER.items()], [8 * cm, 2.5 * cm, 2.5 * cm])]

    s += [P("3. Dataset and preprocessing", h1),
          P("<b>Data.</b> EuroSAT (Sentinel-2 RGB, 64x64 px, 10 land-use classes, 27,000 patches) and EuroSAT-SAR "
            "(27,000 dual-polarisation Sentinel-1 patches, GeoTIFF, 2 channels). <b>Pairing.</b> The EuroSAT-SAR "
            "description states that each Sentinel-2 patch was matched with one Sentinel-1 patch using geospatial "
            "coordinates and that the folder structure mirrors EuroSAT. We pair samples by (class folder, filename stem); "
            "a check on our copy found all 27,000 stems present in both modalities. The documentation does not state "
            "pixel-level alignment or same-day acquisition, so we treat the pairs as geographically matched, not as "
            "pixel-registered image pairs (inspect_data.py provides a weak empirical check)."),
          P("<b>Split.</b> Stratified by class, 70% / 15% / 15% (train / validation / test), seed 42, performed over "
            "<i>pairs</i> so both modalities of a patch always fall in the same split (splits/*.csv). Test pairs are never used "
            "for training, SAR normalisation statistics or checkpoint selection. EuroSAT patches come from many locations "
            "but neighbouring patches can be similar, so residual leakage between splits cannot be fully excluded from metadata alone. "
            "<b>Preprocessing.</b> RGB: scaled to [0,1], ImageNet mean/std. SAR: non-finite values replaced by the channel median; "
            "per-channel 1st-99th percentile clipping and standardisation, statistics fitted on the training split only and stored in "
            "the checkpoint. Both inputs are bilinearly upsampled from 64 to 128 px. Training augmentation: random flips and transpose.")]

    s += [P("4. Model architecture", h1),
          P("Two VGG-16 convolutional encoders (RGB: 3 input channels; SAR: first convolution rebuilt for 2 channels by averaging the "
            "ImageNet filters over RGB and tiling them). Each is followed by global average pooling and a 512-unit FC+ReLU+dropout block. "
            "SAR is the attention source (the structural modality, analogous to the sketch). The attention vector sigmoid(FC(GAP(SAR conv map))) "
            "multiplies the RGB FC features. A shared MLP (512-256-128) produces the 128-d embeddings V<sub>a</sub> (RGB) and V<sub>b</sub> (SAR); "
            "a shared linear classifier, and two linear cross-decoders implement L<sub>2</sub> and L<sub>4</sub>."),
          table([["Aspect", "Paper", "This implementation"],
                 ["Modalities", "Photo + sketch (Earth on Canvas)", "RGB + dual-pol SAR (EuroSAT), as listed in the assignment"],
                 ["Attention source", "Sketch encoder", "SAR encoder (an analogy; SAR is not a sketch)"],
                 ["Input / feature map", "224 px assumed (7x7x512)", "64 px upsampled to 128 px (4x4x512)"],
                 ["Batch size", "128", bs_txt],
                 ["Epochs", "about 40-50", ep_txt],
                 ["Loss reduction", "Frobenius norm (sum)", "Mean over batch and dimensions by default (configurable)"],
                 ["Inference", "Not specified for unpaired queries", "Each modality embedded independently (pair-free RGB path)"],
                 ["Extra training terms", "-", "Auxiliary CE + distillation of the pair-free RGB path towards the attended embedding"],
                 ["Unspecified details", "FC sizes, V<sub>ab</sub>, decoders, pairing", "Our assumptions: MLP sizes, shared classifier, linear decoders"],
                 ["Training stage", "Single stage (this paper)", "Single stage, end-to-end"]],
                [3.2 * cm, 5.2 * cm, 8.2 * cm], wrap=True)]

    s += [P("5. Training configuration", h1)]
    if summary:
        c = summary["config"]
        s += [table([["Setting", "Value"],
                     ["Seed (split, init, shuffling)", c["seed"]], ["Optimiser", f"SGD, lr {c['train']['lr']}, momentum {c['train']['momentum']}, weight decay {c['train']['weight_decay']}"],
                     ["Epochs / batch size", f"{summary['epochs_run']} run (configured {c['train']['epochs']}) / {c['train']['batch_size']}"],
                     ["Input size / embedding", f"{c['model']['input_size']} px / {c['model']['emb_dim']}-d"],
                     ["Attention", c["model"]["attention"]], ["Pairing mode", c["data"]["pairing"]],
                     ["Mixed precision", c["train"]["amp"]], ["Loss weights (w1..w4, aux)",
                      f"{c['loss']['w1']}, {c['loss']['w2']}, {c['loss']['w3']}, {c['loss']['w4']}, {c['loss']['aux']} ({c['loss']['reduction']})"],
                     ["Split sizes (train/val/test)", " / ".join(str(summary['split_sizes'][k]) for k in ("train", "val", "test"))],
                     ["Best epoch (by validation mAP)", f"{summary['best']['epoch']}"],
                     ["Training time", f"{summary['train_seconds'] / 60:.1f} min"],
                     ["Hardware / software", f"{summary['env'].get('gpu', 'CPU')} | torch {summary['env'].get('torch', '?')} | python {summary['env'].get('python', '?')}"]],
                    [5.5 * cm, 11 * cm], wrap=True)]
        if summary["best"]["epoch"] >= 0.8 * summary["epochs_run"]:
            s += [Spacer(1, 4), P(f"The best validation epoch ({summary['best']['epoch']}) is close to the last epoch "
                                  f"({summary['epochs_run']}), so training may not have fully converged; longer training was not tried.")]
    else:
        s += [P("<b>reports/train_summary.json not found: run train.py first.</b>")]

    s += [P("6. Experimental protocol", h1),
          P("Relevance is class-level: a gallery item is relevant to a query iff it has the same land-use label. Gallery and queries are the "
            "held-out test pairs; RGB to SAR uses RGB queries and the SAR gallery, and vice versa. Items are ranked by Euclidean distance between "
            "128-d embeddings. AP is computed over the full ranking, mAP is the mean over queries; P@K is precision in the top K. In the "
            "<i>standard</i> cross-modal variant the query's own paired counterpart remains in the gallery; the <i>strict</i> variant removes it. "
            "Same-modality retrieval excludes the query itself. Queries with no relevant item would be skipped (none occur with balanced classes). "
            "A random-embedding row gives the chance level (about the class prior, 0.1 for 10 balanced classes).")]

    s += [P("7. Results (measured in this project)", h1)]
    if test:
        r = test["results"]
        rows = [["Direction", "mAP", "P@10", "P@20", "P@50"]]
        for k, v in r.items():
            rows.append([clean(k), _fmt(v.get("mAP")), _fmt(v.get("P@10")),
                         _fmt(v.get("P@20")), _fmt(v.get("P@50"))])

        # Some training runs write test_results_after_training.json, whose schema
        # records the checkpoint path but not checkpoint_epoch or l2_normalize.
        checkpoint_epoch = test.get("checkpoint_epoch")
        if checkpoint_epoch is None and summary:
            checkpoint_epoch = summary.get("best", {}).get("epoch")
        checkpoint_label = checkpoint_epoch if checkpoint_epoch is not None else "not recorded"
        embedding_type = "L2-normalised" if test.get("l2_normalize", False) else "raw"

        s += [
            P(f"Test split: {test.get('n_test_pairs', 'unknown')} pairs; checkpoint epoch {checkpoint_label}; "
              f"seed {test.get('seed', 'not recorded')}; {embedding_type} embeddings. Values are fractions in [0, 1]."),
            table(rows, [7.5 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm])
        ]

        if abl_test:
            ra = abl_test["results"]
            rows = [["Model", "RGB to SAR mAP", "SAR to RGB mAP", "RGB to SAR P@10", "SAR to RGB P@10"]]
            rows.append(["With attention (main)", _fmt(r["RGB->SAR"]["mAP"]), _fmt(r["SAR->RGB"]["mAP"]),
                         _fmt(r["RGB->SAR"]["P@10"]), _fmt(r["SAR->RGB"]["P@10"])])
            rows.append(["Without attention (ablation)", _fmt(ra["RGB->SAR"]["mAP"]), _fmt(ra["SAR->RGB"]["mAP"]),
                         _fmt(ra["RGB->SAR"]["P@10"]), _fmt(ra["SAR->RGB"]["P@10"])])
            d1 = r["RGB->SAR"]["mAP"] - ra["RGB->SAR"]["mAP"]
            d2 = r["SAR->RGB"]["mAP"] - ra["SAR->RGB"]["mAP"]
            s += [
                Spacer(1, 6),
                P("<b>Attention ablation</b> (identical settings except <i>model.attention</i>)."),
                table(rows, [5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]),
                P(f"Difference (with minus without): RGB to SAR {d1:+.4f}, SAR to RGB {d2:+.4f} mAP. Each configuration was "
                  f"trained once with one seed, so differences of this size cannot be separated from run-to-run variance "
                  f"without repeated runs.")
            ]
        else:
            s += [P("No attention ablation results were found (reports/no_attention/test_results.json).", small)]
    else:
        s += [P("<b>Test results JSON not found: run test.py after training.</b> No results are reported here because none "
                "have been measured.")]

    s += [P("8. Training and validation loss", h1)]
    for f, cap in (("loss_curves.png", "Total loss per epoch (train and validation)."),
                   ("loss_components.png", "Individual loss terms and classifier accuracy.")):
        if (plot_dir / f).exists():
            s += [KeepTogether([Image(str(plot_dir / f), width=14 * cm if f == "loss_curves.png" else 16.5 * cm,
                                      height=9 * cm if f == "loss_curves.png" else 8.8 * cm, kind="proportional"),
                                P(cap, small)])]
        else:
            s += [P(f"{f} not found: run train.py first.", small)]

    s += [P("9. Comparison with the published results", h1),
          P("The published numbers (Section 2) are for photo-sketch retrieval on Earth on Canvas with 14 classes (chance level about 0.07). "
            "Our experiments use RGB-SAR retrieval on EuroSAT with 10 classes (chance level about 0.10), a different task, dataset, class count and "
            "split. The absolute mAP values are therefore <b>not comparable</b>, and we do not claim to have reproduced the paper's numbers. "
            "What is reproduced is the method (sketch-style cross-attention between two VGG-16 encoders with the four-term loss in a shared "
            "128-d space) and the evaluation measure; the paper's central claim (attention helps relative to a no-attention baseline) is examined "
            "by the ablation above, where available.")]

    s += [P("10. Limitations and conclusion", h1),
          P("<b>Limitations.</b> (i) The dataset differs from the paper's; SAR is an analogy for the sketch, not an equivalent. (ii) Several "
            "architectural details are not specified in the paper and are our assumptions (Section 4). (iii) At inference, attention cannot use a paired "
            "SAR image, so the attention benefit reaches the pair-free path only through training (auxiliary distillation); this is our design, "
            "not the paper's. (iv) Loss terms L<sub>1</sub> and L<sub>3</sub> admit a trivial collapse, which only L<sub>2</sub> counteracts; the weights were "
            "not tuned. (v) Results come from a single seed per configuration. (vi) The EuroSAT RGB-SAR pairing is geographic according to the dataset "
            "documentation; pixel alignment and acquisition date were not verified. (vii) Mixed precision and GPU kernels leave small sources of "
            "non-determinism, so re-training gives close but not bit-identical metrics. (viii) The input size and batch size differ from the paper "
            "because of the 6 GB GPU and the 64 px EuroSAT patches."),
          P("<b>Conclusion.</b> We implemented an attention-driven RGB-SAR cross-modal retrieval network following Att-CMIR-Net, with train/test "
            "separation, fixed seeds, saved checkpoints and loss plots, and evaluated both retrieval directions with mAP and P@K on a held-out test "
            "split. The measured values are listed in Section 7; interpretation should respect the single-seed and dataset-mismatch caveats above.")]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Attention-Driven Cross-Modal RS Image Retrieval - report")
    doc.build(s)
    print(f"[saved] {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--group", default="Group: <names>")
    p.add_argument("--train_summary", default="reports/train_summary.json")
    p.add_argument("--results", default="reports/test_results.json")
    p.add_argument("--ablation_results", default="reports/no_attention/test_results.json")
    p.add_argument("--ablation_summary", default="reports/no_attention/train_summary.json")
    p.add_argument("--plot_dir", default="plots")
    p.add_argument("--out", default="reports/report.pdf")
    build(p.parse_args(argv))


if __name__ == "__main__":
    main()
