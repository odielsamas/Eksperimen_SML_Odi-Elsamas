# Eksperimen SML — Developer Burnout Prediction

Submission Kriteria 1: Membangun Sistem Machine Learning (Dicoding)

## Struktur Folder

```
Eksperimen_SML_Odi-Elsamas/
├── .github/
│   └── workflows/
│       └── preprocessing.yml        # GitHub Actions (Advanced)
├── developer_burnout_raw/
│   └── developer_burnout.csv        # Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Odi-Elsamas.ipynb  # Notebook eksperimen (Basic)
│   └── automate_Odi-Elsamas.py       # Script otomatisasi (Skilled)
├── requirements.txt
└── README.md
```

## Dataset

- **Sumber:** [Kaggle - Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)
- **Ukuran:** 7.000 baris × 12 kolom
- **Target:** `burnout_level` (Low / Medium / High)

## Cara Menjalankan

### Manual (Notebook)

Buka `preprocessing/Eksperimen_Odi-Elsamas.ipynb` di Jupyter atau Google Colab.

### Otomatis (Script)

```bash
pip install -r requirements.txt

python preprocessing/automate_Odi-Elsamas.py \
    --input developer_burnout_raw/developer_burnout.csv \
    --output developer_burnout_preprocessing
```

### GitHub Actions

Workflow berjalan otomatis saat ada push ke branch `main` yang mengubah file dataset atau script preprocessing. Hasil preprocessing tersimpan sebagai **artifact** yang dapat didownload dari tab Actions.

## Output

Folder `developer_burnout_preprocessing/` berisi:

- `train.csv` — data latih (80%)
- `test.csv` — data uji (20%)
