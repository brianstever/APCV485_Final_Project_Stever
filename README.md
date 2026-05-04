# Spectrogram-Based CNN With Attention for Deepfake Audio Detection

**Author:** Brian Stever  
**Email:** brianstever@arizona.edu  
**Course:** APCV 485 - Deep Learning in Cyber  

## Description

This project builds a deepfake audio detector for podcast production workflows. Audio clips are converted into 128-bin log-Mel spectrograms, then classified as bonafide or spoofed using a CNN with Squeeze-and-Excitation channel attention.

The main idea is to treat voice authenticity as a time-frequency detection problem. I trained on ASVspoof 2019 LA train/dev, evaluated on the full ASVspoof 2021 LA evaluation set, and compared the CNN + SE model against ResNet-18 and LFCC-GMM baselines.

Main result: **18.28% EER**, **0.8979 AUC-ROC**, **0.9479 F1**, and **1.66 ms MacBook MPS latency** for CNN + SE on ASVspoof 2021 LA.

## Repository Structure

```text
APCV485_Final_Project_Stever/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── data/
│   └── models/
├── notebooks/
│   └── Stever_notebook.ipynb
├── models/
│   ├── README.md
│   └── best_model.pt
├── results/
│   ├── README.md
│   ├── figures/
│   └── metrics/
├── report/
│   ├── Stever_Report.pdf
│   ├── Stever_Report.tex
│   └── figures/
└── presentation/
    └── Stever_Presentation.pdf
```

## Requirements

Recommended Python version: **Python 3.10+**

Key libraries:

- PyTorch / torchaudio
- NumPy, SciPy, pandas
- scikit-learn
- matplotlib, seaborn
- soundfile, pydub
- joblib, tqdm

All Python dependencies are listed in `requirements.txt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The podcast-style MP3 augmentation uses `pydub`, which works best when `ffmpeg` is installed. If `ffmpeg` is not available, the code falls back without MP3 compression.

## Data

The audio data is too large to include in GitHub. The expected local layout is:

```text
data/
├── raw/
├── processed/
├── LA2019_full_train_dev/
│   ├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019_LA_train/flac/
│   └── ASVspoof2019_LA_dev/flac/
├── ASVspoof2021_LA_eval/flac/
└── asvspoof2021_keys/extracted/keys/LA/CM/trial_metadata.txt
```

Dataset counts used in the completed experiment:

| Split | Bonafide | Spoof | Total |
|---|---:|---:|---:|
| ASVspoof 2019 LA train | 2,580 | 22,800 | 25,380 |
| ASVspoof 2019 LA dev | 2,548 | 22,296 | 24,844 |
| ASVspoof 2021 LA eval | 18,452 | 163,114 | 181,566 |

ASVspoof 2019 LA train/dev was exported into the official ASVspoof folder/protocol layout. ASVspoof 2021 LA evaluation audio and trial metadata were downloaded from the ASVspoof/Zenodo release.

Full training and evaluation outputs are written under `results/runs/` when the scripts are run locally. That folder is intentionally not packaged because it contains generated checkpoints and large run artifacts. The final figures, metrics, and selected model checkpoint are included in `results/` and `models/`.

## Training

Train the main CNN + SE model:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py \
  --data_root data/LA2019_full_train_dev \
  --model cnn_se \
  --epochs 30 \
  --batch_size 64 \
  --num_workers 4 \
  --augment \
  --save_dir results/runs/cnn_se \
  --no_progress
```

Train the ResNet-18 baseline:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py \
  --data_root data/LA2019_full_train_dev \
  --model resnet18 \
  --epochs 30 \
  --batch_size 64 \
  --num_workers 4 \
  --augment \
  --save_dir results/runs/resnet18 \
  --no_progress
```

Run the LFCC-GMM baseline:

```bash
python src/lfcc_gmm.py \
  --train_protocol data/LA2019_full_train_dev/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --train_audio_dir data/LA2019_full_train_dev/ASVspoof2019_LA_train/flac \
  --eval_protocol data/asvspoof2021_keys/extracted/keys/LA/CM/trial_metadata.txt \
  --eval_audio_dir data/ASVspoof2021_LA_eval/flac \
  --eval_protocol_format asvspoof2021 \
  --save_dir results/runs/lfcc_gmm \
  --num_workers 8 \
  --no_progress
```

## Evaluation

Evaluate the saved CNN + SE checkpoint:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/evaluate.py \
  --data_root data/LA2019_full_train_dev \
  --model cnn_se \
  --checkpoint models/best_model.pt \
  --batch_size 64 \
  --num_workers 4 \
  --protocol_path data/asvspoof2021_keys/extracted/keys/LA/CM/trial_metadata.txt \
  --audio_dir data/ASVspoof2021_LA_eval/flac \
  --protocol_format asvspoof2021 \
  --save_dir results/runs/cnn_se \
  --no_progress
```

Run podcast-stress evaluation:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/evaluate.py \
  --data_root data/LA2019_full_train_dev \
  --model cnn_se \
  --checkpoint models/best_model.pt \
  --batch_size 64 \
  --num_workers 4 \
  --protocol_path data/asvspoof2021_keys/extracted/keys/LA/CM/trial_metadata.txt \
  --audio_dir data/ASVspoof2021_LA_eval/flac \
  --protocol_format asvspoof2021 \
  --augment_eval \
  --augment_prob 1.0 \
  --save_dir results/runs/cnn_se_podcast_eval \
  --seed 142 \
  --no_progress
```

## Results

| Experiment | EER | AUC-ROC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| CNN + SE clean 2021 LA | 18.28% | 0.8979 | 0.9072 | 0.9479 |
| CNN + SE podcast stress 2021 LA | 18.48% | 0.8852 | 0.8991 | 0.9430 |
| ResNet-18 clean 2021 LA | 17.21% | 0.8957 | 0.8549 | 0.9145 |
| LFCC-GMM clean 2021 LA | 30.89% | 0.7400 | 0.7063 | 0.8128 |
| CNN + SE cross-vocoder A05/A06 dev | 5.07% | 0.9881 | 0.4131 | 0.3506 |

The cross-vocoder split had strong EER/AUC but weak fixed-threshold F1, which showed me that threshold calibration matters.

Key packaged outputs:

- `models/best_model.pt`
- `results/figures/`
- `results/metrics/`
- `report/Stever_Report.pdf`
- `presentation/Stever_Presentation.pdf`
- `notebooks/Stever_notebook.ipynb`

## Reproducibility Check

Run:

```bash
python scripts/check_project_ready.py
```

Expected result:

```text
Ready: public submission repo has the required files.
```

## Notes

- Latency is reported as a local MacBook measurement using Apple Silicon MPS.
- The code supports CUDA, MPS, and CPU device selection.
- The full ASVspoof audio data is not meant to be committed to Git because it is large.
- The executed notebook is included because it matches the class lab style and shows the project walkthrough with outputs.
