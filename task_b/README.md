# Layout Aware Text Extraction (Computer Vision)

A Python program to convert presentation slide images into interactive and editable HTML pages. This program automatically extracts text, detects font sizes and colors, and removes the original text from the background image.

## 📂 Project Structure

```text
task_b/
├─ data/           # Place your slide images here (.jpg/.png)
├─ output/         # The generated .html files will be saved here
├─ main.py         # Main program script
├─ requirements.txt# List of required Python libraries
└─ README.md       # Project documentation

```

## ✨ Key Features

* **Dual-Pipeline OCR**: Uses EasyOCR to accurately read text while precisely calculating font sizes.
* **Smart Background Eraser**: Automatically removes the original text from the image using OpenCV *Inpainting* techniques without ruining the background design.
* **Editable HTML**: Generates an HTML file where the text is perfectly overlaid and can be directly typed/edited right in the browser.

## 🚀 How to Run

### 1. Navigate to the Directory

```bash
cd task_b
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment. Install the required packages using:

```bash
pip install -r requirements.txt
```

### 3. Prepare Images

Place the presentation slide images you want to convert into the `task_b/data/` folder.

### 4. Run the Program

To execute the full end-to-end extraction pipeline, simply run the main Python script:

```bash
python main.py
```

### 5. View the Results

Open the `task_b/output/` folder and double-click on the generated `.html` files to open them in your browser (Chrome, Edge, etc.). You can hover your cursor over any text and edit it directly!
