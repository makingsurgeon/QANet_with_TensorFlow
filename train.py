import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import pickle
import collections
from QANet import QANet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from utils.output import write_predictions
from utils.evaluation import evaluate
from params import Params

params = Params()

# load train data
with open("dataset/train_set.pkl", 'rb') as f:
    train_data = pickle.load(f)
train_data['start_label_fin'] = np.argmax(train_data['y_start'], axis=-1)
train_data['end_label_fin'] = np.argmax(train_data['y_end'], axis=-1)

# load validation data
with open("./dataset/dev_set.pkl", 'rb') as f:
    dev_data = pickle.load(f)
with open('./dataset/dev_examples.pkl', 'rb') as f:
    eval_examples = pickle.load(f)
with open('./dataset/dev_features.pkl', 'rb') as f:
    eval_features = pickle.load(f)

# load embedding matrix
word_mat = np.load('./dataset/word_emb_mat.npy')
char_mat = np.load('./dataset/char_emb_mat.npy')

ems = []
f1s = []
model = QANet(params, word_mat=word_mat, char_mat=char_mat)
optimizer = Adam(learning_rate=params.learning_rate, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.9999)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
              loss_weights=[0.5, 0.5, 0, 0])
RawResult = collections.namedtuple("RawResult",
                                ["qid", "start_logits", "end_logits"])


class QANetCallback(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        # the learning rate warm_up step
        self.global_step = 1
        self.max_f1 = 0

    def on_train_begin(self, logs=None):
        learning_rate = min(params.learning_rate,
                            params.learning_rate / np.log(params.warm_up_steps) * np.log(self.global_step))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

    def on_epoch_begin(self, epoch, logs=None):
        # use a list to store the loss of each epoch
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        # the learning warm-up scheme
        learning_rate = min(params.learning_rate,
                            params.learning_rate / np.log(params.warm_up_steps) * np.log(self.global_step))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        loss = logs.get('loss')
        self.losses.append(loss)

    def on_epoch_end(self, epoch, logs=None):
        logits1, logits2, _, _ = self.model.predict(x=[dev_data['context_id'], dev_data['question_id'],
                                                       dev_data['context_char_id'], dev_data['question_char_id']],
                                                    batch_size=params.batch_size,
                                                    verbose=1)
        all_results = []
        for i, qid in enumerate(dev_data['qid']):
            start_logits = logits1[i, :]
            end_logits = logits2[i, :]
            all_results.append(RawResult(qid=qid,
                                      start_logits=start_logits,
                                      end_logits=end_logits))
        # save the prediction results
        output_prediction_file = os.path.join('output_prediction.json')
        output_nbest_file = os.path.join('output_nbest.json')
        write_predictions(eval_examples, eval_features, all_results,
                          n_best_size=20, max_answer_length=params.ans_limit,
                          do_lower_case=False, output_prediction_file=output_prediction_file,
                          output_nbest_file=output_nbest_file)
        # model evaluation
        metrics = evaluate('original_data/dev-v1.1.json', output_prediction_file, None)
        ems.append(metrics['exact'])
        f1s.append(metrics['f1'])
        # save the EM and F1 scores
        result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
        result.to_csv('./logs/result.csv', index=False)
        # write the losses of each epoch
        with open("./logs/losses_epoch" + str(epoch) + ".txt", 'a', encoding='utf=8') as f:
            for loss in self.losses:
                f.write(str(loss) + '\n')
        if f1s[-1] > self.max_f1:
            self.max_f1 = f1s[-1]
            model.save_weights("./saved_model_epoch" + str(epoch+1) + "/QANet_model_epoch" + str(epoch+1) + ".tf")


qanet_callback = QANetCallback()
qanet_callback.set_model(model)

model.fit(x=[train_data['context_id'], train_data['question_id'],
             train_data['context_char_id'], train_data['question_char_id']],
          y=[train_data['y_start'], train_data['y_end'], train_data['start_label_fin'],
             train_data['end_label_fin']],
          batch_size=params.batch_size,
          epochs=params.epoch,
          callbacks=[qanet_callback])

model.summary()
