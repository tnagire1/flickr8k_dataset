from pickle import load
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.layers import Dense
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from gtts import gTTS
from playsound import playsound
import load_data as ld
import generate_model as gen
import argparse
import os
from tensorflow.keras import backend as k
# extract features from each photo in the directory
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def extract_features(filename):
  # load the model
  model = VGG19()
  # re-structure the model
  hidden_layer = model.layers[-3].output
  # Connect a new layer on it
  new_output = Dense(4096)(hidden_layer)
  # Build a new model

  model = Model(inputs=model.inputs, outputs=new_output)
  print(model.summary())

  image = load_img(filename, target_size=(224, 224))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  print(f'image{image.shape}')
  feature = model.predict(image, verbose=0)
  print(f'feature{feature.shape}')
  return feature

# generate a description for an image
def generate_desc(model, tokenizer, photo, index_word, max_length, beam_size=5):

  captions = [['startseq', 0.0]]
  in_text = 'startseq'
  for i in range(max_length):
    all_caps = []
    for cap in captions:
      sentence, score = cap
      if sentence.split()[-1] == 'endseq':
        all_caps.append(cap)
        continue
      sequence = tokenizer.texts_to_sequences([sentence])[0]
      sequence = pad_sequences([sequence], maxlen=max_length)
      y_pred = model.predict((photo,sequence), verbose=0)[0]
      yhats = np.argsort(y_pred)[-beam_size:]

      for j in yhats:
        # map integer to word
        word = index_word.get(j)
        # stop if we cannot map the word
        if word is None:
          continue
        # Add word to caption, and generate log prob
        caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
        all_caps.append(caption)

    # order all candidates by score
    ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
    captions = ordered[:beam_size]

  return captions

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    # Use best caption
    predicted.append(yhat[0].split())
  # calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def eval_test_set(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    # Use best caption
    predicted.append(yhat[0].split())
  predicted = sorted(predicted)
  actual = [x for _,x in sorted(zip(actual,predicted))]

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Generate image captions')
  parser.add_argument("-i", "--image", help="Input image path")
  parser.add_argument("-m", "--model", help="model checkpoint")
  args = parser.parse_args()


  # load the tokenizer
  tokenizer = load(open('../models/tokenizer.pkl', 'rb'))
  index_word = load(open('../models/index_word.pkl', 'rb'))
  # pre-define the max sequence length (from training)
  tokenizer.oov_token = None
  max_length = 34
  # load the model
  if args.model:
    filename = args.model
  else:
    filename = r'C:\Users\DELL\Documents\komal_new\models\wholeModel_vgg19.h5'
  model = load_model(filename)

  if args.image:
    # load and prepare the photograph
    photo = extract_features(args.image)
    # generate description
    captions = generate_desc(model, tokenizer, photo, index_word, max_length)

    # for cap in captions:
    #   # remove start and end tokens
    #   seq = cap[0].split()[1:-1]
    #   desc = ' '.join(seq)
    #   print(desc)
    #  print('{} [log prob: {:1.2f}]'.format(desc,cap[1]))

    final_cap = ' '.join(captions[4][0].split()[1:-1])
    print("caption =" , final_cap)
    language = 'en'
    # myobj = gTTS(text=final_cap, lang=language, slow=False)
    # myobj.save("../imgs/Predicted_audio.mp3")
    # src = "../imgs/Predicted_audio.mp3"

    import pyttsx3
    engine = pyttsx3.init()
    engine.say(final_cap)


    import webbrowser

    print("Enter Yout Choice \n 1. For searching realated to caption \n 2. For searching related images ")
    ch = input()
    if (ch == '1'):
      url = "https://www.google.com.tr/search?q={}".format(final_cap)
      webbrowser.open_new_tab(url)
    else:
      # url = "http://images.google.com/images?um=1&hl=en&safe=active&nfpr=1&q="+final_cap
      url = "https://www.google.com.tr/images?q={}".format(final_cap)
      webbrowser.open_new_tab(url)
    # playsound(src)

  else:
    # load test set
    test_features, test_descriptions = ld.prepare_dataset('test')[1]

    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, index_word, max_length)
k.clear_session()