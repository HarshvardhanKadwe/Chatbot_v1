import random
import json
import torch
import time
import streamlit as st

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#to check if we can use cuda for faster or better performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()



st.title("Chatbot")

bot_name = "ChatGPT-1"
# print("Let's chat! (type 'quit' to exit)")

count=0
sentence = st.text_input("You: ")
if st.button('Send'): 
    # while True:
    #sentence from the user
    # print(sentence)
    # if sentence == "quit":
    #     break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    #so we try and find the output with max probability
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    #the threshhold is defined by the developer
    if prob.item() > 0.55:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                st.write(f"{bot_name}: {random.choice(intent['responses'])}")
                count+=1

    elif prob.item() < 0.75:
        st.write(f"{bot_name}: I do not understand...")
    
    #Chatbot asks the user "Do you like apples or oranges?" within the first 5 interactions
    # if count==5:
    #     time.sleep(0.01)
    #     print(f"{bot_name}: Do you like apples or oranges?")
    #     count+=1