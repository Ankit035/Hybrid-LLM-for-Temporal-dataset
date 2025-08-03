#  Deep Semantics for Structured Data: Hybrid LLM-Based Models for Temporal Forecasting

This repository presents multiple approaches to **time series forecasting** using both **traditional machine learning models** (LightGBM) and **hybrid models enhanced by embeddings from a Large Language Model (LLM)** such as BERT. We apply these techniques on two datasets: an energy consumption dataset and the Jena Climate Dataset.

---

## 📊 Datasets Used

### ⚡ Energy Dataset
- **Source**: [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- **Focus**: Daily total consumption from a single meter (`MT_001`)

### 🌤️ Jena Climate Dataset
- **Source**: [Jena Climate Dataset (TensorFlow)](https://www.tensorflow.org/tutorials/weather_forecasting)
- **Features**: Meteorological parameters (temperature, humidity, pressure, wind, etc.) recorded every 10 minutes from 2009–2016 at the Max Planck Institute for Biogeochemistry in Jena, Germany.

---

## 🛠 Technologies

- Python 3.x
- Pandas, NumPy
- LightGBM
- Scikit-learn
- Hugging Face Transformers (BERT)
- PyTorch
- TensorFlow/Keras (for LSTM)
- Matplotlib / Seaborn
- PCA (for embedding reduction)

---

## 🔁 Project Pipeline (Applicable Across Datasets)

1. **Data Loading & Resampling**
   - Convert to daily frequency using average or sum.
   - Parse timestamps, index by date.

2. **Feature Engineering**
   - Time-based features: day of week, month, weekend.
   - Lag features, rolling means, and Fourier terms.

3. **Modeling Approaches**
   - LightGBM with traditional tabular features.
   - LightGBM + LLM (BERT) embeddings.
   - Hybrid LightGBM + LSTM architecture (Jena dataset only).

---

## 🤖 BERT Embedding Integration

- Convert structured time series features to text prompts (e.g., `"Wednesday, March, Consumption: 123"`).
- Use pre-trained BERT to obtain contextual embeddings.
- Apply PCA to reduce embeddings to 10 dimensions.
- Append to traditional features for model input.

---

## 🧪 Results

### ⚡ Energy Consumption Forecasting (UCI Dataset)

| Model                                 | MAE    | RMSE   | R² Score | MAPE   |
|--------------------------------------|--------|--------|----------|--------|
| LightGBM (Traditional Features Only) | 71.27  | 105.68 | 0.9550   | 80.09% |
| LightGBM + BERT Embeddings           | 72.22  | 106.31 | 0.9545   | 77.21% |

📌 *While the BERT-enhanced model had a marginally higher MAE and RMSE, it offered better relative accuracy (lower MAPE).*

---

### 🌤️ Weather Forecasting (Jena Climate Dataset)

| Model                              | MAE  | RMSE | R² Score | MAPE   |
|-----------------------------------|------|------|----------|--------|
| Hybrid LightGBM + LSTM Model      | 1.31 | 1.70 | 0.9500   | 54.99% |

📌 *The hybrid architecture combining LightGBM and LSTM achieved strong performance, especially in capturing temporal and sequential patterns in weather data.*

---

## 📁 Folder Structure

```
├── data/
│   ├── LD2011_2014.txt
│   └── jena_climate_2009_2016.csv
├── notebooks/
│   ├── energy_forecasting_lightgbm.ipynb
│   └── jena_hybrid_model.ipynb
├── scripts/
│   └── energy_forecast_lightgbm.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/time-series-hybrid-forecasting.git
cd time-series-hybrid-forecasting

# Install dependencies
pip install -r requirements.txt

# Run Energy Forecasting
python scripts/energy_forecast_lightgbm.py

# Or run Jupyter Notebooks for interactive exploration
jupyter notebook notebooks/energy_forecasting_lightgbm.ipynb
jupyter notebook notebooks/jena_hybrid_model.ipynb
```

---

## 📎 Notes

- PCA is used for dimensionality reduction of BERT embeddings.
- BERT embeddings are simulated from tabular time series by generating pseudo-text.
- The hybrid model leverages LightGBM for structured features and LSTM for temporal dependency.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [UCI ML Repository](https://archive.ics.uci.edu/)
- [TensorFlow Jena Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
