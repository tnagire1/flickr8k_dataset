import load_data as ld
import generate_model as gen
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from pickle import dump
patience =50
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


batch_size=128

def train_model(weight = None, epochs = 10):
  data = ld.prepare_dataset('train')
  train_features, train_descriptions = data[0]
  test_features, test_descriptions = data[1]

  tokenizer = gen.create_tokenizer(train_descriptions)
  dump(tokenizer, open(r'C:\Users\DELL\Documents\komal_new\models\tokenizer.pkl', 'wb'))
  index_word = {v: k for k, v in tokenizer.word_index.items()}
  dump(index_word, open(r'C:\Users\DELL\Documents\komal_new\models\index_word.pkl', 'wb'))

  vocab_size = len(tokenizer.word_index) + 1
  print('Vocabulary Size: %d' % vocab_size)

  max_length = gen.max_length(train_descriptions)
  print('Description Length: %d' % max_length)

  model = gen.define_model(vocab_size, max_length)

  if weight != None:
    model.load_weights(weight)

  filepath = r'C:\Users\DELL\Documents\komal_new/models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                save_best_only=True, mode='min')
  reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                patience=int(patience / 4), verbose=1)
  early_stop = EarlyStopping('val_loss', patience=patience)
  steps = len(train_descriptions)//batch_size
  val_steps = len(test_descriptions)//batch_size
  train_generator = gen.data_generator(train_descriptions, train_features, tokenizer, max_length)
  val_generator = gen.data_generator(test_descriptions, test_features, tokenizer, max_length)

  model.fit(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
        callbacks=[checkpoint,reduce_lr,early_stop], validation_data=val_generator, validation_steps=val_steps)

  try:
      model.save(r'C:\Users\DELL\Documents\komal_new/models/wholeModel.h5', overwrite=True)
      model.save_weights(r'C:\Users\DELL\Documents\komal_new/models/weights.h5',overwrite=True)
  except:
      print("Error in saving model.")
  print("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=500)
