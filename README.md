# Llama-Mental-Health-Chatbot-with-Sentiment-Analysis-Integration

**[Google Colab Link](https://colab.research.google.com/drive/1_cl-kCMFcJctDmwYNouQM7DaO5_q95HD)**



---

# 🌸 Wellness Bot: A Sentiment-Aware Chatbot 🌸

Welcome to the **Wellness Bot** project! This chatbot is designed to offer empathetic responses based on sentiment analysis. Whether you're feeling upbeat or need a pick-me-up, the Wellness Bot is here to respond thoughtfully to your mood.

---

## 👩‍💻 **Project Overview**
This project enhances a chatbot by integrating **sentiment analysis** with **natural language generation** to create meaningful, dynamic conversations. It uses state-of-the-art NLP models and provides a simple user interface built using Gradio.

### Features:
- **Sentiment Detection:** Analyzes user input to detect positive or negative sentiment.
- **Adaptive Responses:** Tailors its reply based on the detected sentiment.
- **Empathy:** Offers supportive responses if negative sentiment is detected.
- **Real-time Text Generation:** Generates conversational responses using a state-of-the-art language model.
- **Clear Chat History:** Easily clear the conversation history by typing "clear".

---

## 🛠️ **Tech Stack**
- **Programming Language:** Python
- **Libraries:**
  - [Transformers](https://huggingface.co/docs/transformers): For using pre-trained models like DistilBERT and Llama-2.
  - [Torch](https://pytorch.org/): Backend for handling tensor computations.
  - [Gradio](https://gradio.app/): For creating a simple, web-based UI.

---

## 🚀 **Getting Started**

### 1. **Set Up Environment**
Install the required dependencies by running:

```bash
pip install torch torchvision torchaudio transformers sentencepiece gradio sentence-transformers accelerate
```

### 2. **Run the Code**
Once the libraries are installed, run the script (either locally or in a Colab environment) to launch the chatbot.

### 3. **Interact with the Bot**
After launching the script, a Gradio interface will open in your browser. Type your message, and the chatbot will respond with sentiment-aware replies. You can clear the chat history by typing `clear`.

---

## 🧠 **How It Works**

### Sentiment Analysis
We use the **DistilBERT** model fine-tuned on the SST-2 dataset to analyze the sentiment of the user's input, classifying it as either **positive** or **negative**.

### Text Generation
The **Llama-2** model (7B version) generates text-based responses for the user. It combines human-like language generation with an understanding of the user’s sentiment to provide relevant and emotionally sensitive responses.

### Chat History Management
The bot tracks chat history and allows users to clear it by typing "clear". This ensures a fresh conversation when needed.

---

## 📋 **Example Usage**

### Input:
```text
I'm feeling really down today.
```

### Bot Reply:
```text
Sorry you're feeling down. I'm here for you. Let me know if there's anything I can do to help.
```

---

## 📄 **Code Breakdown**

```python
# Step 1: Install Necessary Libraries
# Install all dependencies, including Hugging Face's Transformers, Torch, and Gradio.
!pip install torch torchvision torchaudio transformers sentencepiece gradio sentence-transformers accelerate

# Step 2: Import Required Libraries
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Step 3: Sentiment Analysis Pipeline
# Using DistilBERT model fine-tuned for sentiment analysis.
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Step 4: Load Llama-2 Model and Tokenizer
# Loading Llama-2 model and tokenizer from Hugging Face.
llama_model_id = "NousResearch/Llama-2-7b-chat-hf"
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, torch_dtype=torch.float16, device_map="auto")
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)

# Step 5: Text Generation Pipeline
response_generator = pipeline(
    "text-generation", model=llama_model, tokenizer=llama_tokenizer,
    torch_dtype=torch.float16, device_map="auto",
    max_length=4096, do_sample=True, temperature=0.6, top_p=0.95
)

# Step 6: Generate Bot's Response
def generate_response(user_input):
    input_len = len(user_input.split())
    max_len = min(4096, input_len * 10)
    return response_generator(user_input, max_length=max_len, do_sample=True)[0]['generated_text'].strip()

# Step 7: Analyze Sentiment of User Input
def analyze_sentiment(user_input):
    sentiment = sentiment_analyzer(user_input)[0]
    label, score = sentiment['label'], sentiment['score']

    if label == "NEGATIVE":
        return "Sorry you're feeling down. I'm here for you." if score > 0.9 else "It seems rough. I'm listening."
    elif label == "POSITIVE":
        return "Great to hear! Keep up the good vibes!" if score > 0.9 else "Glad you're feeling better."
    else:
        return "I'm here for you no matter what."

# Step 8: Clean Redundant Phrases from Generated Text
def clean_redundant(user_input, generated_text):
    phrases_to_remove = ["It seems rough.", "Take a deep breath."]
    for phrase in phrases_to_remove:
        generated_text = generated_text.replace(phrase, "").strip()
    return generated_text

# Step 9: Main Chatbot Reply Logic
def chatbot_reply(user_input):
    sentiment_msg = analyze_sentiment(user_input)
    generated_text = generate_response(user_input)
    cleaned_response = clean_redundant(user_input, generated_text)
    return f"{sentiment_msg} {cleaned_response}".strip()

# Step 10: Track Chat History and Handle 'Clear' Command
chat_history = []

def interface_function(user_input):
    global chat_history
    if user_input.lower() == "clear":
        chat_history = []
        return "Chat history cleared."

    reply = chatbot_reply(user_input)
    chat_history.append(f"USER: {user_input}\nBOT: {reply}")
    return "\n\n".join(chat_history)

# Step 11: Build Gradio Interface
interface = gr.Interface(
    fn=interface_function,
    inputs="text",
    outputs="text",
    title="Wellness Bot",
    description="Chat about your feelings, and the bot will respond based on sentiment. Type 'clear' to clear chat.",
    allow_flagging="never"
)

# Step 12: Launch Gradio Interface
interface.launch(debug=True)
```

---

## 🧩 **Code Explanation**
- **Sentiment Analysis:** This part uses **DistilBERT** to analyze the input and classify it as positive or negative.
- **Text Generation:** **Llama-2** generates responses based on user input, with max length and sampling parameters to make the conversation more dynamic.
- **Interface:** Gradio is used to create a simple web interface where users can interact with the bot.

---

## 🔍 **Future Improvements**
- **Multilingual Support:** Expand the bot’s capabilities to support more languages.
- **Contextual Memory:** Allow the bot to maintain long-term context over multiple interactions.
- **Advanced Sentiment Features:** Include more nuanced sentiment analysis, such as detecting joy, anger, or confusion.

---

## 📄 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 **Acknowledgements**
Big thanks to Hugging Face, Gradio, and the open-source community for providing the tools and resources to bring this project to life!

---
