# 🛢️ AI SpillGuard Pro
### Enterprise Oil Spill Detection System
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI SpillGuard Pro** is a state-of-the-art oil spill detection system that leverages deep learning (U-Net + ResNet34) to identify and segment oil spills from Synthetic Aperture Radar (SAR) and satellite imagery in real-time.

---

## 🚀 Features

-   **Deep Learning Engine**: Powered by a U-Net architecture with a ResNet34 encoder for high-precision segmentation.
-   **Real-Time Inference**: Processes satellite imagery in seconds to provide immediate actionable insights.
-   **Enterprise Dashboard**: Built with Streamlit, offering a polished, responsive, and intuitive interface.
-   **Smart Alert System**:
    -   **✅ SAFE**: Indicates no significant oil spill detected.
    -   **🚨 CRITICAL**: Automatically triggers high-priority alerts when oil coverage exceeds safe thresholds.
-   **Visual Trinity**: View the original image, segmentation mask, and an alpha-blended overlay side-by-side.
-   **Analytics & Reporting**:
    -   Donut charts for coverage distribution.
    -   Detailed pixel-level statistics.
    -   Exportable JSON reports and CSV history.
-   **API Integration**: Ready-to-deploy FastAPI endpoints for programmatic access.

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Premchand006/AI_SpillGuard_Oil_Spill_Detection.git
    cd AI_SpillGuard_Oil_Spill_Detection
    ```

2.  **Create a virtual environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Weights**
    Ensure `best_model.pth` is placed in the root directory.

---

## 💻 Usage

### Running the Web App
Launch the interactive dashboard:
```bash
streamlit run app.py
```
The application will open automatically in your browser (usually at `http://localhost:8501`).

### How to Use
1.  **Upload**: Drag and drop a SAR or satellite image (PNG, JPG, TIFF) into the upload zone.
2.  **Analyze**: The AI automatically processes the image.
3.  **Inspect**:
    -   Check the **Status Card** for immediate "SAFE" or "CRITICAL" feedback.
    -   Use the **Overlay Transparency** slider to compare the detection with the original image.
    -   View the **Coverage Distribution** chart.
4.  **Export**: Download the result images or the full analysis report.

---

## 🏗️ Tech Stack

-   **Frontend**: Streamlit
-   **Deep Learning**: PyTorch, Segmentation Models PyTorch (SMP)
-   **Image Processing**: OpenCV, Pillow, Numpy
-   **Visualization**: Plotly, Matplotlib
-   **Backend (API)**: FastAPI (documented in app)

---

## 📊 Model Performance

The model is trained to segment 4 classes:
| Class ID | Name | Color | Description |
| :---: | :--- | :--- | :--- |
| 0 | **Background** | ⬛ Black | Ocean surface, land, etc. |
| 1 | **Oil Spill** | 🟪 Magenta | Confirmed oil slicks |
| 2 | **Look-alike** | 🟨 Yellow | Natural biogenic films, wind slicks |
| 3 | **Ship/Wake** | 🟦 Cyan | Vessels and their wake patterns |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

<p align="center">
  Developed by <strong>Prem Chand</strong>
</p>
