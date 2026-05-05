# Hybrid Quantum-Classical Diabetes Prediction System

## Overview

This project implements a **Hybrid Quantum Neural Network** to predict diabetes risk using both classical deep learning and quantum computing. Leveraging PyTorch for classical feature extraction and Qiskit for quantum circuits, the system aims to push accuracy and innovation in medical AI.


## Features

- **Hybrid Architecture:**  
  Classical Encoder → Quantum Circuit (`ZZFeatureMap` + `RealAmplitudes`) → Classical Classifier.
- **Professional Dashboard:**  
  Glassmorphism UI with medical-grade aesthetics.
- **Real-time Inference:**  
  Flask backend for instant predictions.
- **Visualizations:**  
  Quantum processing animations and probability outputs.

## Example

**Input:**
```json
{
  "Glucose": 120,
  "BMI": 33.6,
  "Age": 45,
  "...": "other features"
}
```

**Output:**
```
Diabetes risk: 92%
Prediction: Positive
```

## Prerequisites

- Python 3.8+
- Qiskit (`pip install qiskit`)
- PyTorch (`pip install torch`)
- Flask (`pip install flask`)
- Node.js *(optional, for advanced frontend, basic HTML works)*
- See `requirements.txt` for all dependencies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shahakash2028/qcardio-heart-disease.git
   cd hybrid-quantum-diabetes-prediction
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (only needed on first run):**
   ```bash
   python backend/train_model.py
   ```
   *This trains the hybrid PyTorch-Qiskit model and saves it in `models/`.*

4. **Run the Backend Server:**
   ```bash
   python backend/app.py
   ```
   *Server runs at* [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage

1. Open `frontend/index.html` in your web browser.
2. Enter patient data (Glucose, BMI, Age, etc.).
3. Click **Analyze Data**.
4. View the quantum-enhanced prediction and risk probability.

## Project Structure

```
backend/
  ├── quantum_model.py
  ├── train_model.py
  └── app.py
frontend/
  ├── index.html
  ├── style.css
  └── ... (scripts and assets)
models/
  └── ... (trained `.pth` and `.pkl` files)
requirements.txt
README.md
LICENSE
```

## Architecture

The pipeline integrates classical and quantum computing as follows:

1. **Classical Encoder:** Extracts relevant medical features using a neural network (PyTorch).
2. **Quantum Circuit:** Applies quantum feature mapping (Qiskit: `ZZFeatureMap`, `RealAmplitudes`).
3. **Classical Classifier:** Combines quantum output with classical layers for final risk prediction.

*For more details, see the code in `backend/quantum_model.py` and the [Qiskit documentation](https://qiskit.org/documentation/).*

## Contributing

Pull requests are welcome!  
If you’d like to propose changes or report bugs, please open an issue first to discuss your ideas.

## Citing & References

- [Qiskit Tutorials – Machine Learning](https://qiskit.org/documentation/)
- Diabetes dataset: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## License

Distributed under the MIT License.  
See [`LICENSE`](LICENSE) for more information.

## Contact

Developed by [AKASH SHAH](mailto:as.shah.2060@gmail.com).

---

*This project is for educational and research purposes and should not be used as a substitute for professional medical advice.*
