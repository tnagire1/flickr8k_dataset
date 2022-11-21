from os import listdir
from pickle import dump
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Input, Reshape, Concatenate, Dense
from tensorflow.keras import backend as k
import numpy as np
import string
from progressbar import progressbar
from tqdm import tqdm
from tensorflow.keras.models import Model


# load an image from filepath
def load_image(path):
    img = load_img(path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return np.asarray(img)


# extract features from each photo in the directory
def extract_features(directory,is_attention=False):
  if is_attention:
    model = VGG19()
    model.layers.pop()
    final_conv = Reshape([49,512])(model.layers[-4].output)
    model = Model(inputs=model.inputs, outputs=final_conv)
    print(model.summary())
    features = dict()
  else:
    model = VGG19()
    hidden_layer = model.layers[-3].output
    new_output = Dense(4096)(hidden_layer)
    model = Model(inputs=model.inputs, outputs=new_output)
    print(model.summary())
    features = dict()

  for name in tqdm(listdir(directory)):
    if name == 'README.md':
      continue
    filename = directory + '/' + name
    image = load_image(filename)
    feature = model.predict(image, verbose=0)
    image_id = name.split('.')[0]
    features[image_id] = feature
    print('>%s' % name)
  return features


# load doc into memory
def load_doc(filename):
  file = open(filename, 'r')
  text = file.read()
  file.close()
  return text


# extract descriptions for images
def load_descriptions(doc):
  mapping = dict()
  for line in doc.split('\n'):
    tokens = line.split()
    if len(line) < 2:
      continue
    image_id, image_desc = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    image_desc = ' '.join(image_desc)
    if image_id not in mapping:
      mapping[image_id] = list()
    mapping[image_id].append(image_desc)
  return mapping


def clean_descriptions(descriptions):
  table = str.maketrans('', '', string.punctuation)
  for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
      desc = desc_list[i]
      desc = desc.split()
      desc = [word.lower() for word in desc]
      desc = [w.translate(table) for w in desc]
      desc = [word for word in desc if len(word)>1]
      desc = [word for word in desc if word.isalpha()]
      desc_list[i] =  ' '.join(desc)


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
  all_desc = set()
  for key in descriptions.keys():
    [all_desc.update(d.split()) for d in descriptions[key]]
  return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
  lines = list()
  for key, desc_list in descriptions.items():
    for desc in desc_list:
      lines.append(key + ' ' + desc)
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()



directory = r'C:\Users\DELL\Documents\komal_new\Flickr8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features, open(r'C:\Users\DELL\Documents\komal_new\models/features.pkl', 'wb'))


filename = r'C:\Users\DELL\Documents\komal_new\Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)

descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
save_descriptions(descriptions, r'C:\Users\DELL\Documents\komal_new\models\descriptions.txt')
k.clear_session()
