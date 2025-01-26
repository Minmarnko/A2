# Importing libraries
import streamlit as st
import pickle
import torch
from utils import LSTMLanguageModel, generate

# Load training data and model parameters
Data = pickle.load(open('./models/Data.pkl', 'rb'))
vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Instantiate the model
vocab_size = 7026
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('./models/best-val-lstm_lm.pt', map_location=torch.device('cpu')))
model.eval()

# Streamlit app setup
st.set_page_config(page_title="Story Generation", layout="centered")
st.title("Gone with the Wind Story Generator")
st.markdown("Enter a prompt to generate a custom Gone with the Wind story.")

# User input section
prompt = st.text_input("Enter a story prompt:", placeholder="e.g., Scarlett was ...")
seq_len = st.selectbox("Select the maximum word limit:", [10, 20, 30])

# Generate story button
if st.button("Generate Story"):
    if prompt.strip() == "":
        st.error("Please enter a valid prompt.")
    else:
        temperature = 0.8
        seed = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate text
        generation = generate(prompt, seq_len, temperature, model, tokenizer, vocab, device, seed)
        sentence = ' '.join(generation)

        # Display result
        st.subheader("Generated Story:")
        st.write(sentence)
