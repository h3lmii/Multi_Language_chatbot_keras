
import random
from pickle import load
import tensorflow as tf
from keras.utils import pad_sequences
import streamlit as st
from tensorflow.keras.models import load_model
import random
import string
import numpy as np
from streamlit_chat import message



with open('chatbot_tokenizer', 'rb') as f:
    loaded_tokenizer = load(f)

loaded_model=load_model('my_chat_model.h5',compile=False)

with open('label_encoder', 'rb') as l:
    le = load(l)

responses={'greeting': ["Hey :-) I'm Helmi and I'm happy to respond you to get to know me better",
  "Hello, thanks for visiting ,I'm Helmi and I'm happy to respond you to get to know me better",
  "Hi there, I'm Helmi and I'm happy to respond you to get to know me better"],
 'goodbye': ['See you later, thanks for visiting',
  'Have a nice day',
  'Bye! Come back again soon.'],
 'thanks': ['Happy to help!', 'Any time!', 'My pleasure'],
 'introduction': ["I’m Helmi Massoussi a Computer Science Engineering student at INSAT Tunisia. I'm very passionate about Data Science and Artificial Intelligence: NLP and Computer Vision in particular",
  'I’m Helmi Massoussi a Computer Science Engineering student at INSAT Tunisia. I like turning machine learning papers into code & I enjoy doing research to pursue my interests in hot AI topics.'],
 'experience': ["Sure, I've worked 3 months as a summer Computer Vision intern to build a real time facial recognition for Startup Members.\n I've also worked as technical data Science Content Writer where I've made research on OCR and Intelligent Document Processing \n and I'm Machine Learning Developer at CrunchDAO among 2000 data scientists and 400 PHDs "],
 'projects': ["Sure,I've worked Smart Protecting System for Online Examination using Opencv and Human Activity Recogntion as Academic Pojects. \n My personal projects are this MultiLanguage Chatbot,Resume Parser WebApp with Streamlilt, Tunisian Dialect Sentiment Analysis deployed with flask and django and more on my github: github.com/h3lmii "],
 'skills': ["In short my Technical skills are: Deep Learning with Keras's Tensorflow and Pytorch.Computer Vision with Opencv Python. NLP with Spacy,Nltk and Transformers. Django,Flask,Streamlit for model deployment and Rest Apis"],
 'education': ["I'm a Computer Science engineering Student at INSAT Tunisia from 09/2019 where I studied Programming,Statistics, Probability , Linear Algebra"],
 'Languages': ["I speak fluent Arabic, English & French and I'm preparing to pass the B1 goethe Deutsch Exam."],
 'Contact': ['Feel free to contact me on Linkedin: https://www.linkedin.com/in/massoussihelmi/  or by mail: helmi.messousi@insat.ucar.tn or by Phone: +21653006861']}


def predict(prediction_input,model,tokenizer,label_encoder):
      texts=[]
      prediction_input=[letters.lower() for letters in prediction_input if letters not in string.punctuation]

      prediction_input=''.join(prediction_input)

      texts.append(prediction_input)
      prediction_input=tokenizer.texts_to_sequences(texts)
      
      prediction_input=np.array(prediction_input).reshape(-1)
      prediction_input=pad_sequences([prediction_input],13)
      
      output=model.predict(prediction_input,verbose=0)
      output=output.argmax()

      
      response_tag=label_encoder.inverse_transform([output])[0]

      return response_tag


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response_tag=predict(userText,loaded_model,loaded_tokenizer,le)
    response = random.choice(responses[response_tag])
    return response


app.run(debug=True)
