import urllib.request
import tarfile
import ssl
import certifi
import os
from transformers import BertTokenizer
import tensorflow as tf
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# URL for the dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"
# Create an SSL context using the certifi certificate bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Use the SSL context to retrieve the data
with urllib.request.urlopen(url, context=ssl_context) as u, open(filename, 'wb') as f:
    f.write(u.read())

with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall()
    
def clean_data(text):
    # Converting text to lowercase
    text = text.lower()
    
    # Removing punctuation from text
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Removing stop words from data
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = [i for i in tokens if not i in stop_words]
    
    # Performing Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(token) for token in text]
    
    return " ".join(text)

# Reading the data from the files
def read_data(directory):
    data = []
    uncleaned_data = []
    labels = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type) # Construct the directory path for the label
         # Iterate through each file in the directory
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname)) as f:
                    text = f.read()   # Read the content of the file
                    uncleaned_text = text   # Storing the uncleaned text
                    cleaned_text = clean_data(text)  # Storing Clean the text
                    data.append(cleaned_text)  # Append cleaned text to 'data'
                    uncleaned_data.append(uncleaned_text)
                labels.append(1 if label_type == 'pos' else 0)
    return uncleaned_data, data, labels
# Reading and preprocessing data from the training directory
train_uncleaned, train_data, train_labels = read_data('aclImdb/train')
# Reading and preprocessing data from the test directory
test_uncleaned, test_data, test_labels = read_data('aclImdb/test')
print("Uncleaned data", train_uncleaned[0])   # Print the first piece of uncleaned text
print("Cleaned data",train_data[0])     # Print the first piece of cleaned text
print("Lable",train_labels[0])         # Print the label of the first piece of text

# Loading the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing the data to match the input format of the model
def preprocess_data(data, labels):
     # Initialize lists to store tokenized input ids and attention masks
    input_ids = []
    attention_masks = []
# Loop through each piece of text data
    for text in data:
        encoded_dict = tokenizer.encode_plus(
                            text,           # Text to encode           
                            add_special_tokens = True,   # Add special tokens for BERT
                            max_length = 64,    # Maximum sequence length       
                            padding='max_length',   # Pad sequences to max length
                            truncation=True,      # Truncate sequences if longer than max length
                            return_attention_mask = True,   # Generate attention mask
                            return_tensors = 'tf',     # Return tensors in TensorFlow format
                      )
        # Append the input ids and attention mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    # Concatenate the input ids and attention masks along axis 0 to create tensors
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    # Convert the labels to a TensorFlow tensor
    labels = tf.convert_to_tensor(labels)

# Return a dictionary containing the input ids and attention mask tensors, along with the labels tensor
    return {'input_ids': input_ids, 'attention_mask': attention_masks}, labels

preprocessed_data = preprocess_data(train_data, train_labels)


from transformers import TFBertForSequenceClassification

# Load pre-trained BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Printing the model summary
model.summary()


# Printing the shapes of the preprocessed data
print("Shape of 'input_ids': ", preprocessed_data[0]['input_ids'].shape)
print("Shape of 'attention_mask': ", preprocessed_data[0]['attention_mask'].shape)
print("Shape of 'labels': ", preprocessed_data[1].shape)

# Printing the first 5 'input_ids'
print("\nFirst 5 'input_ids':")
print(preprocessed_data[0]['input_ids'][:5])

# Printing the first 5 'attention_mask'
print("\nFirst 5 'attention_mask':")
print(preprocessed_data[0]['attention_mask'][:5])

# Printing the first 5 'labels'
print("\nFirst 5 'labels':")
print(preprocessed_data[1][:5])

num_labels = 2  # Assuming binary sentiment analysis
# This layer will map the output of the previous layer to a probability distribution over the two classes (0 and 1).
classification_layer = tf.keras.layers.Dense(num_labels, activation='softmax')
# This is done because the BERT model typically outputs logits (raw predictions) without applying any activation function
model.layers[-1].activation = tf.keras.activations.linear  
# This adds a new layer on top of the existing BERT model to perform the final classification
model.layers.append(classification_layer)

from sklearn.model_selection import train_test_split
#split data into train and test using train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Prepare the training and testing datasets
preprocessed_train_data = preprocess_data(train_data, train_labels)
preprocessed_test_data = preprocess_data(test_data, test_labels)

# Encode the training data using the tokenizer
# The tokenizer tokenizes the text, adds special tokens, pads sequences to a maximum length,
# truncates sequences if they exceed the maximum length, and returns tensors in TensorFlow format
X_train_encoded = tokenizer(train_data, padding=True, truncation=True, return_tensors='tf')
X_test_encoded = tokenizer(test_data, padding=True, truncation=True, return_tensors='tf')

# Compiling the model
#Adam optimizer is used with a specific learning rate and epsilon value
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
# Sparse categorical crossentropy loss is used since the labels are integers
# The logits are used as the input to the loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Sparse categorical accuracy metric is used to compute the accuracy of the model predictions
# 'accuracy' is the name given to this metric
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')


model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Training the model
history = model.fit(
    preprocessed_data[0],
    preprocessed_data[1],
    epochs=3,  
    batch_size=32,  
    validation_split=0.2 # Fraction of training data to use as validation data
)

preprocessed_test_data = preprocess_data(test_data, test_labels)

# Evaluating the model performance on test data
test_loss, test_accuracy = model.evaluate(
    preprocessed_test_data[0],
    preprocessed_test_data[1],
    verbose=1
)

# Printing the test loss and accuracy
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
# tokenizer and model are saved into a directory so that  it can be loaded later for making predictions
tokenizer.save_pretrained('/Users/shivani/Desktop/Movie reviews_bert/')
model.save("/Users/shivani/Desktop/Movie reviews_bert/")

from sklearn.metrics import classification_report, confusion_matrix

# Making predictions on test data
predictions = model.predict(preprocessed_test_data[0])
predictions = np.argmax(predictions.logits, axis=1)

# Printing the classification report
print(classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))

# Printing the confusion matrix
print(confusion_matrix(test_labels, predictions))
