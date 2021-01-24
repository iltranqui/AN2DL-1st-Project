#!/usr/bin/env python
# coding: utf-8

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# Set the seed for random operations. 
# This let our experiments to be reproducible. 
SEED = 1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Get current working directory
cwd = os.getcwd()

# Make the gpu work in the current enviroment
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


# In[74]:


cwd = os.getcwd()
cwd


# In[75]:


dataset_dir = os.path.join(cwd, 'VQA_Dataset')

# if os.path.exists(dataset_dir):
   # shutil.rmtree(dataset_dir)


# # Hyperparameters

# In[76]:


img_w = 256
img_h = 256
batch_size = 16
lr = 1e-4

MAX_NUM_WORDS = 5000 # max number of unique words in dictionary

FEATURES = 512 # size of feature vector for images and questions

UNITS = 32  


# In[77]:


labels_dict = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        'apple': 6,
        'baseball': 7,
        'bench': 8,
        'bike': 9,
        'bird': 10,
        'black': 11,
        'blanket': 12,
        'blue': 13,
        'bone': 14,
        'book': 15,
        'boy': 16,
        'brown': 17,
        'cat': 18,
        'chair': 19,
        'couch': 20,
        'dog': 21,
        'floor': 22,
        'food': 23,
        'football': 24,
        'girl': 25,
        'grass': 26,
        'gray': 27,
        'green': 28,
        'left': 29,
        'log': 30,
        'man': 31,
        'monkey bars': 32,
        'no': 33,
        'nothing': 34,
        'orange': 35,
        'pie': 36,
        'plant': 37,
        'playing': 38,
        'red': 39,
        'right': 40,
        'rug': 41,
        'sandbox': 42,
        'sitting': 43,
        'sleeping': 44,
        'soccer': 45,
        'squirrel': 46,
        'standing': 47,
        'stool': 48,
        'sunny': 49,
        'table': 50,
        'tree': 51,
        'watermelon': 52,
        'white': 53,
        'wine': 54,
        'woman': 55,
        'yellow': 56,
        'yes': 57
}

num_answers = len(labels_dict)


# # Functions

# In[78]:


import json
def unwrap_weighted(path, split = 0.2):
    
    dataset_dir = os.path.join(path, 'train_questions_annotations.json')
    training_dir = os.path.join(path, 'training.json')
    validation_dir = os.path.join(path, 'validation.json')
        
    dic_images = None
    
    with open(dataset_dir) as f:
       dic_images = json.load(f)
        
    dict_keys = list(dic_images.keys())
    np.random.shuffle(dict_keys)
    questions = int(round(split*len(dict_keys)))
        
    dic_validations = { dict_keys[i]:dic_images[dict_keys[i]] for i in range(questions)}
    dic_training = {dict_keys[i]:dic_images[dict_keys[i]] for i in range(questions, len(dict_keys))}
        
    with open(training_dir, 'w') as fp:
       json.dump(dic_training, fp)
    with open(validation_dir, 'w') as fp:
       json.dump(dic_validations, fp)

path = os.getcwd()


# In[79]:


def get_token_dic_quest(path, max_num_words = 5000):
    from tensorflow.keras.preprocessing.text import Tokenizer
    dataset_dir = os.path.join(path, 'train_questions_annotations.json')
    
    # Load dataset
    with open(dataset_dir) as f:
        dic_images = json.load(f)

    # Get all questions as strings in a list
    questions = [dic['question'] for dic in dic_images.values()]

    # Strip '?' from questions
    questions = [s.translate(str.maketrans('', '', '?')).lower() for s in questions if not s == '']
    questions_tokenizer = Tokenizer(num_words=max_num_words)
    questions_tokenizer.fit_on_texts(questions)

    questions_wtoi = questions_tokenizer.word_index # index 0 reserved for padding
    
    questions_tokenized = questions_tokenizer.texts_to_sequences(questions)
    max_question_length = max(len(sentence) for sentence in questions_tokenized)
    
    return questions_tokenizer, questions_wtoi, max_question_length


def from_questions_to_dict(path, dict_req, max_num_words = 5000):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Return dictionary q_wtoi
    tokenizer, wtoi, max_len = get_token_dic_quest(path, max_num_words = 5000)
    
    translated_dics = []
    
    for dic in dict_req:
        
        question = dic['question'].translate(str.maketrans('', '', '?')).lower()
        question = tokenizer.texts_to_sequences([question])
        question = pad_sequences(question, maxlen=max_len)
        dic['question'] = question[0]
        dic['answer'] = labels_dict[dic['answer']]
        translated_dics.append(dic)
    
    return translated_dics


# In[80]:


from PIL import Image
    
# Patches Generator
class dataset_generator(tf.keras.utils.Sequence):

  def __init__(self, path, preprocessing, subset = "training", image_generator = None, batch_size = 5, max_num_words=5000):
    json_file = subset + ".json"
    dat_dir = os.path.join(path, 'VQA_Dataset')
    subset_file = os.path.join(dat_dir, json_file)
    
    with open(subset_file) as f:
       dictionaries = json.load(f)
       dictionaries = dictionaries.values()
       self.dictionary = from_questions_to_dict(dat_dir, dictionaries, max_num_words)
    
    self.batch_size = batch_size
    self.image_generator = image_generator
    self.preprocessing = preprocessing
    self.dat_dir = dat_dir
    self.gen = image_generator
    self.batch_size = batch_size
    self.max_num_words = max_num_words
    self.n = 0
    
  def __len__(self):
    return len(self.dictionary)//self.batch_size

  def __getitem__(self, index):
    lower_bound = index*self.batch_size
    upper_bound = (index+1)*self.batch_size
    
    batch_img = []
    batch_que = []
    batch_ans = []
    
    for idx in range(lower_bound, upper_bound):
        img, que, ans = self.__data_generation__(idx)
        batch_img.append(img)
        batch_que.append(que)
        batch_ans.append(ans)
        
    batch_img = np.stack(batch_img, axis=0)
    batch_que = np.stack(batch_que, axis=0)
    batch_ans = np.stack(batch_ans, axis=0)
    
    x = [batch_img, batch_que]
    y = batch_ans
    
    return x, y
    
    
  def __data_generation__(self, idx):
    actual_dict = self.dictionary[idx]
    
    img_name = actual_dict['image_id']
    answer = actual_dict['answer']
    question = actual_dict['question']
    
    actual_img = Image.open(os.path.join(self.dat_dir, "Images", img_name + ".png"))
    actual_img = actual_img.convert('RGB')
    img_arr = np.array(actual_img)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    if self.image_generator is not None:
        img_arr = self.gen.random_transform(img_arr)
    
    if self.preprocessing is not None:
        img_arr = self.preprocessing(img_arr)
        
    img_arr = np.squeeze(img_arr, axis=0)
    
    return img_arr, question, answer


# Datasets generation

# In[81]:


unwrap_weighted(os.path.join(path, 'VQA_Dataset'))


# In[82]:


preprocessing_function = tf.keras.applications.vgg16.preprocess_input

gen = dataset_generator(path = os.getcwd(), preprocessing = preprocessing_function, 
                  subset = "training", image_generator = None, max_num_words=5000, batch_size = batch_size)

gen_val = dataset_generator(path = os.getcwd(), preprocessing = preprocessing_function, 
                  subset = "validation", image_generator = None, max_num_words=5000, batch_size = batch_size)

'''
dataset = tf.data.Dataset.from_generator(lambda: gen, output_types=([tf.float32, tf.uint8], tf.uint8), 
                                         output_shapes=([2,], ()))

dataset_val = tf.data.Dataset.from_generator(lambda: gen_val, output_types=([tf.float32, tf.uint8], tf.uint8), 
                                         output_shapes=([2,], ()))

dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

dataset_val = dataset_val.batch(batch_size)
dataset_val = dataset_val.repeat()

iterator = iter(dataset)
giggino = next(iterator)
print(giggino)
'''

for f in gen:
    print(f)
    break
    


# # Image Encoder

# In[83]:


image_encoder = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(img_h, img_w, 3),
    pooling='avg'
)

for layer in image_encoder.layers:
    layer.trainable = False

image_encoder.summary()


# # Question Encoder

# Load questions into a List

# In[84]:


import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset_dir = os.path.join(cwd, "VQA_Dataset", "train_questions_annotations.json")

# Load dataset
with open(dataset_dir) as f:
    dic_images = json.load(f)
            
# Get all questions as strings in a list
questions = [dic['question'] for dic in dic_images.values()]

# Strip '?' from questions
questions = [s.translate(str.maketrans('', '', '?')).lower() for s in questions if not s == '']
print(questions[12])

# max_words_in_sentence = max(len(question.split(' ')) for question in questions)
# print(max_words_in_sentence)


# Tokenize questions

# In[85]:


# Create Tokenizer to convert words to integers
# num_words: Top No. of words to be tokenized. Rest will be marked as unknown or ignored.
questions_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# tokenizing based on "texts". This step generates the word_index and map each            word to an integer other than 0.
questions_tokenizer.fit_on_texts(questions)

# generating sequence based on tokenizer's word_index. Each sentence will now be          represented by combination of numericals
        # Example: "Good movie" may be represented by [22, 37]
questions_tokenized = questions_tokenizer.texts_to_sequences(questions)
# each sentence into a sequence of tokens (in this case, only the 20000 most frequent)

# "hello raffaele" -> [9, 78] 

questions_wtoi = questions_tokenizer.word_index # index 0 reserved for padding
print('Total number of words:', len(questions_wtoi))

print(questions_tokenized[0])


# In[86]:


# simplt measure the max length in all the questions
max_question_length = max(len(sentence) for sentence in questions_tokenized)
print('Max question length:', max_question_length)

# Pad to max question sentence length
padded_questions = pad_sequences(questions_tokenized, maxlen=max_question_length)
# Padding: [[1], [2, 3], [4, 5, 6]]   --> [[0,0,1], [0, 2, 3], [4, 5, 6]]
print("Padded questions shape:", padded_questions.shape)


# # Stardrand Model

# ### Load pre-trained GloVe embedding

# In[87]:


path_to_glove_file = os.path.join(cwd,'glove.6B\glove.6B.100d.txt')

embeddings_index = {}
with open(path_to_glove_file, encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found", len(embeddings_index), "word vectors.")


# In[88]:


num_tokens = len(questions_wtoi) + 1
embedding_dim = 100

hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in questions_wtoi.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# Create question encoder

# In[89]:


embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=False,
    input_length=max_question_length
)

question_encoder = tf.keras.models.Sequential()
question_encoder.add(tf.keras.layers.Input(shape=(max_question_length), dtype="int32"))
question_encoder.add(embedding_layer)
question_encoder.add(tf.keras.layers.LSTM(units=FEATURES))

question_encoder.summary()


# # Create complete model

# Load indexes for answers

# In[90]:


multiplied_features = tf.keras.layers.Multiply()([image_encoder.layers[-1].output, question_encoder.layers[-1].output])
dense_1 = tf.keras.layers.Dense(UNITS, activation='tanh')(multiplied_features)
out = tf.keras.layers.Dense(num_answers, activation='softmax')(dense_1)

network = tf.keras.models.Model(inputs=[image_encoder.layers[0].input, question_encoder.layers[0].input], outputs=out)

network.summary()


# In[91]:


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
metrics = ['accuracy']
loss = tf.keras.losses.SparseCategoricalCrossentropy()


# In[92]:


network.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[93]:


import os
from datetime import datetime

cwd = os.getcwd()

exps_dir = os.path.join('C')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

exp_name = 'exp'

exp_dir = os.path.join(exps_dir, exp_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    
callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'c')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True, save_best_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

# Early Stopping
# --------------
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks.append(es_callback)

decay = 0.1
min_lr = 1e-5


# Decay
decay_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=decay, patience=15, verbose=1,
    mode='auto', min_lr=min_lr)
callbacks.append(decay_callback)

# ---------------------------------


# # Training

# In[95]:


network.fit(x=gen,
            epochs=100,
            steps_per_epoch=len(gen),
            validation_data=gen_val,
            validation_steps=len(gen_val),
            callbacks=callbacks)


# In[26]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

path = os.path.join(cwd, 'VQA_Dataset','test_questions.json')
max_num_words = 50000
max_question_length = 21
# preprocessing_function = tf.keras.applications.vgg16.preprocess_input
dat_dir = os.path.join(cwd, 'VQA_Dataset','Images')
# network = None

with open(path) as f:
       dic_test = json.load(f)

dic_test_values = dic_test.values()
test_questions = [q['question'].lower().translate(str.maketrans('', '', '?')) for q in dic_test_values]

test_tokenizer = Tokenizer(num_words=max_num_words)
test_tokenizer.fit_on_texts(test_questions)

test_wtoi = test_tokenizer.word_index

test_tokenized = test_tokenizer.texts_to_sequences(test_questions)

# print(test_tokenized[45])

# max_question_length = max(len(sentence) for sentence in test_tokenized)        

results = dict()

for question_id in dic_test.keys():
    
    temp_dic = dic_test[question_id]
    # print(temp)
    question = temp_dic['question'].lower().translate(str.maketrans('', '', '?'))
    # print(question)
    image_id = dic_test[question_id]['image_id']
    
    img = Image.open(os.path.join(dat_dir, image_id + ".png"))
    img = img.convert('RGB')
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocessing_function(img_arr)
    
    question_tokenized = test_tokenizer.texts_to_sequences([question])
    
    padded_question = pad_sequences(question_tokenized, maxlen=max_question_length)
    # print(padded_question)
    
    result = network.predict([img_arr, padded_question], verbose=0, batch_size=1)
    
    result = tf.argmax(result[0])
    result = int(result)
    itoa = {v: k for k, v in labels_dict.items()}
    
    answer = itoa[result]
    
    print(answer)
    
    results[question_id] = answer
    
    # print(question_id, question, image_id)


# In[ ]:


import os
from datetime import datetime

def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


# In[28]:


create_csv(results, results_dir='/content/drive/MyDrive/')

