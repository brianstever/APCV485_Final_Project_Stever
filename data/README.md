# Data

The audio datasets are too large to include in Git.

Expected local layout:

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

For this project, ASVspoof 2019 LA train/dev was exported into the official ASVspoof folder layout, and ASVspoof 2021 LA eval audio plus keys were downloaded from the ASVspoof/Zenodo release.
