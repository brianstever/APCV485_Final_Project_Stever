"""Check the public submission repo structure."""

from __future__ import annotations

from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "data/README.md",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "src/__init__.py",
    "src/preprocessing.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/utils.py",
    "src/data/dataset.py",
    "src/data/preprocess.py",
    "src/models/cnn_se.py",
    "src/models/baseline.py",
    "notebooks/Stever_notebook.ipynb",
    "models/README.md",
    "models/best_model.pt",
    "results/README.md",
    "results/figures/sample_spectrogram.png",
    "results/figures/augmentation_comparison.png",
    "results/figures/cnn_se_training_history.png",
    "results/figures/cnn_se_roc_curve.png",
    "results/figures/cnn_se_confusion_matrix.png",
    "results/figures/cnn_se_score_distribution.png",
    "results/figures/eer_comparison.png",
    "results/figures/robustness_checks.png",
    "results/metrics/metrics_summary_full_run.csv",
    "results/metrics/metrics_summary_full_run.json",
    "results/metrics/cnn_se_metrics.json",
    "results/metrics/resnet18_metrics.json",
    "results/metrics/lfcc_gmm_metrics.json",
    "report/Stever_Report.pdf",
    "report/Stever_Report.tex",
    "report/figures/system_pipeline.png",
    "report/figures/model_architecture_table.png",
    "presentation/Stever_Presentation.pdf",
]


def main() -> None:
    root = Path.cwd()
    missing = []
    for rel_path in REQUIRED_PATHS:
        path = root / rel_path
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:8} {rel_path}")
        if not path.exists():
            missing.append(rel_path)

    if missing:
        print("\nNot ready yet: fix the missing items above.")
        raise SystemExit(1)

    print("\nReady: public submission repo has the required files.")


if __name__ == "__main__":
    main()
