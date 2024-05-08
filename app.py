from flask import Flask, request, render_template
from transformers import BertTokenizer
import tensorflow as tf

app = Flask(__name__)

model_directory = '/Users/shivani/Desktop/Movie reviews_bert/Code/model_files'

# Load the TensorFlow SavedModel
loaded_model = tf.saved_model.load(model_directory)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_directory)

# Define the model architecture and weights
model = loaded_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['text']
    except KeyError:
        return 'Error: Text input is missing. Please provide text input in the form.'
    
    # Preprocess the input text
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors='tf')
    # Get the model prediction
    outputs = model(inputs)
    logits = outputs['logits']  # Accessing the logits from the model outputs
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    # Map predicted class index to labels
    labels = ['negative', 'positive']
    result = {'text': data, 'sentiment': labels[predicted_class]}
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
