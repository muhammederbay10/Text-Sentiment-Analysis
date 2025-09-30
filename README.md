# Text Sentiment Analysis with BERT

This project provides a complete workflow for fine-tuning a BERT model to classify text sentiment. The model is trained on the IMDB Movie Reviews dataset to distinguish between positive and negative reviews. A user-friendly web interface is included, built with Streamlit, to allow for easy testing and inference.

---

## 🚀 Live Demo

Here are a couple of examples from the interactive Streamlit application, showcasing its ability to correctly identify both positive and negative sentiment.

**Negative Review Example:**

![Negative Review Demo](https://github.com/user-attachments/assets/40595cf3-459c-4062-bf3d-30b73f00a73f)

**Positive Review Example:**

![Positive Review Demo](https://github.com/user-attachments/assets/918aa729-9673-4762-adc1-c9fd14f476e9)

---

## 📈 Model Performance

The fine-tuned model was evaluated on a held-out validation set and achieved the following strong performance metrics:

| Metric    | Score   |
| :-------- | :------ |
| **Accuracy**  | **83.6%**   |
| **F1 Score**  | **83.5%**   |
| Precision | 84.0%   |
| Recall    | 82.9%   |

---

## ✨ Features

-   **Data Processing:** Efficiently loads, cleans, and balances the dataset.
-   **Hugging Face Integration:** Uses `AutoTokenizer` for tokenization and the `Trainer` API for a streamlined training process.
-   **Interactive UI:** A Streamlit application is provided to easily test the model with your own text.
-   **Pre-trained Model:** The final trained model checkpoint is included in the repository.

---

## 🛠️ Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.8+
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/muhammederbay10/Text-Sentiment-Analysis.git
    cd Text-Sentiment-Analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## ▶️ Running the Application

This project includes a Streamlit application for easy inference.

To run the web app, use the following command in your terminal:

```sh
streamlit run app.py
```

This will launch a local web server where you can input text and see the model's sentiment prediction in real-time.

---

Created by Muhammed (2025)
