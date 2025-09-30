import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model():
    """
    Loads the fined-tuned model and tokenizer.
    """
    model_path = "models/checkpoint-1491"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}. please ensure the path {model_path} is correct.")
        return None, None
    
# --- Streamlit App ---

st.title("Text Sentiment Analysis")
st.write("Enter text to analyze its sentiment.")

# Load model and tokenizer
model, tokenizer = load_model()

if model and tokenizer:
    # setup the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a text area for user input
    user_input = st.text_area("Enter your review here:", height = 150)

    # Create a button to trigger the analysis
    if st.button("Analyze sentiment"):
        if user_input.strip():
            # Tokenize the user's input text
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

            # Move tensors to correct device
            inputs = {key: val.to(device) for key, val in inputs.items()}
                
            # Perform inference
            with torch.no_grad():
                logits = model(**inputs).logits

            # Get the predicted class ID (0 for N, 1 for P)
            predicted_class_id = torch.argmax(logits, dim=1).item()

            # Map the prediction to a human-readable label
            sentiment = "Positive" if predicted_class_id == 1 else "Negative"    

            # Display the result
            st.subheader("Prediction:")
            if sentiment == "Positive":
                st.success(f"The text' sentiment is {sentiment}")
            else:
                st.error(f"The text' sentiment is {sentiment}")

        else:
            st.warning("Please enter a review to analyze.")