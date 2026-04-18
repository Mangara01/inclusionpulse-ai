# InclusionPulse AI

InclusionPulse AI adalah prototype berbasis AI untuk menajamkan prioritas intervensi nasional Indonesia pada tema ekonomi digital dan inklusi keuangan menggunakan data real dari API publik.

## Sumber data
Semua indikator diambil dari World Bank Indicators API tanpa token.

## File utama
- `scripts/01_fetch_and_build_worldbank_dataset.py` → ambil data & bentuk dataset final
- `app.py` → dashboard Streamlit
- `azure_language_helper.py` → Azure AI Language helper
- `.github/workflows/refresh-open-api-data.yml` → refresh data otomatis

## Jalankan lokal
```bash
pip install -r requirements.txt
pip install -r requirements_pipeline.txt
python scripts/01_fetch_and_build_worldbank_dataset.py
streamlit run app.py
