# echoes-of-xingu
AI-powered pipeline for the discovery of archaeological sites in the Upper Xingu River Basin, leveraging satellite imagery, LIDAR data, historical texts, and more.

**Discovering Lost Civilizations in the Amazon Through AI, Remote Sensing, and Indigenous Knowledge**

---

## 🌍 Project Overview

_Echoes of Xingu_ is an interdisciplinary geospatial AI project designed for the **OpenAI to Z Challenge**. It aims to uncover previously undocumented archaeological sites in the Brazilian Amazon — specifically the historically rich yet underexplored **Xingu River Basin** — by fusing state-of-the-art machine learning, satellite imagery, LIDAR, historical documents, and Indigenous oral traditions.

This project is not just about finding ruins. It’s about reconstructing lost histories, city networks, ceremonial routes, and challenging outdated notions about pre-Columbian Amazonian societies.

---

## 🧠 Core Objectives

- **Predict** the locations of undiscovered archaeological sites in the Xingu region.
- **Analyze** and integrate multiple data modalities, including satellite data, LIDAR, historical records, and ethnographic sources.
- **Leverage GPT-4.1** for interpreting colonial and Indigenous texts into geospatial hypotheses.
- **Visualize** and validate predictions through open-source geospatial tools and external archaeological evidence.

---

## 🗺️ Region of Focus: The Upper Xingu Basin

This project focuses on the Upper Xingu, a dense, biodiverse region in Mato Grosso, Brazil, known for:
- Pre-Columbian earthworks (geoglyphs, raised fields, ditches)
- Historical accounts of advanced settlements
- Long-standing Indigenous cultural continuity (e.g., Kuikuro, Kalapalo tribes)
- Relatively sparse LIDAR coverage (opportunity for discovery)

---

## 🔬 Methodology

1. **Data Sources**
   - **Satellite Imagery**: Sentinel-2, Landsat-8, MODIS
   - **LIDAR**: NASA GEDI, Earth Archive surveys
   - **Historical Texts**: Diaries from Portuguese bandeirantes, Jesuit missions
   - **Ethnographic Data**: Kuikuro oral maps, linguistic clustering
   - **Hydrological Models**: Ancient riverbeds, oxbow lakes, and seasonal floodplains

2. **Preprocessing**
   - Vegetation masking via NDVI & EVI
   - Topographic anomaly extraction from LIDAR
   - OCR & NLP conversion of colonial records to structured data
   - Named entity and location resolution from oral traditions

3. **Model Pipeline**
   - CNN or Swin Transformer for spatial anomaly detection (imagery)
   - GNN (Graph Neural Network) for spatial context across river networks
   - GPT-4.1 for:
     - Parsing textual descriptions into GIS overlays
     - Identifying place candidates mentioned in colonial/Indigenous sources
     - Summarizing findings and suggesting new site hypotheses

---

## 🧠 Use of GPT-4.1 / o4-mini

- **Colonial Diaries Parsing**: GPT translates historical descriptions (e.g., "2 days journey past the river fork near a tall hill") into spatial constraints.
- **Oral Tradition Structuring**: Converts Indigenous stories and spatial metaphors into coordinate-based overlays.
- **Hypothesis Generation**: GPT proposes new candidate locations based on learned patterns from text + image analysis.

---

## ✅ Validation Strategy

1. **Cross-Modal Confirmation**
   - Ensure predicted anomaly overlaps across LIDAR + satellite + textual heatmaps.
2. **Ethno-Historical Cross-Checks**
   - Validate predictions against known Indigenous maps and oral histories.
3. **Archaeological Ground Truthing**
   - Compare predictions with known earthwork sites via published academic data.
4. **Crowdsourced Review**
   - Publish candidate maps for Indigenous and academic communities to review.

---

## 🏛️ Potential Historical Discoveries

- **Urban Planning**: Rediscovery of large, gridded settlements similar to Kuhikugu.
- **Agricultural Networks**: Raised fields and fish weirs indicating advanced agronomy.
- **Trade Routes**: Interconnected waterways and road systems forming regional economies.
- **Ceremonial Sites**: Geoglyphs aligned with solstice or cardinal directions.

---

## 💡 Innovation Highlight

- **"Text-to-Geo" Layering**: Novel use of GPT-4.1 to convert colonial and Indigenous descriptions into map-ready overlays.
- **Multi-Fusion Validation**: Combines CNN-LIDAR anomalies, historical texts, and tribal oral geography in a joint confidence scoring system.
- **Hydro-Temporal Modeling**: Uses ancient river paths to weight predictions, assuming ancient people settled near former water routes.

---

## 🧰 Tech Stack

- Python, PyTorch, Hugging Face Transformers
- QGIS / Leaflet for geospatial visualization
- LangChain + OpenAI API (GPT-4.1, o4-mini)
- GDAL, Rasterio, Geopandas
- Jupyter + Colab for development

---

## 📁 Repository Structure

```

X-(RP)/
│
├── configs/              # Model and data configs
├── data/                 # Raw and processed geospatial data
├── models/               # Saved model checkpoints
├── notebooks/            # Exploratory analysis and demos
├── src/                  # Core ML, NLP, and pipeline code
├── tests/                # Unit and integration tests
├── run\_pipeline.sh       # Full training/inference pipeline
├── README.md             # This file

````

---

## 🤝 Ethical Collaboration

This project is committed to working with Indigenous communities respectfully and with consent. Predictive data is shared for review and feedback before any public dissemination.

---

## 🧪 How to Run

```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
bash run_pipeline.sh
````

---

## 👥 Team & Acknowledgements

Developed by an interdisciplinary team of archaeologists, geographers, Indigenous scholars, and machine learning engineers. Special thanks to the Kuikuro community and open-access geospatial providers.

---

## 📜 License

MIT License — For research and educational use only. Any archaeological fieldwork should be conducted with permission and collaboration of Indigenous communities.
