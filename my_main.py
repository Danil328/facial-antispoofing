import gdown
import os
import glob
import pandas as pd
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import gc
from my_tools import kaggle_bag
import numpy as np

def auc(y_true, y_pred):
    auc_score = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc_score

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def load_weights():
    print('load weights ...')
    gdown.download('https://drive.google.com/uc?id=15TKOlGEgk-3m6R8TeMzFxn_RQzwB6hjM', os.path.join(path_models, 'model_1.h5'), quiet=True)
    gdown.download('https://drive.google.com/uc?id=1bhLypjl0BNtaFSjOLlICO0ecCkZ-rNKF', os.path.join(path_models, 'model_2.h5'), quiet=True)
    gdown.download('https://drive.google.com/uc?id=1hrkhV4JdN_JpHkkxpGnBVGeBzKLnBRMz', os.path.join(path_models, 'model_3.h5'), quiet=True)
    gdown.download('https://drive.google.com/uc?id=1Giv-wG23AqZ9IEnwfDUJlHULPn7Tsb8C', os.path.join(path_models, 'model_4.h5'), quiet=True)
    gdown.download('https://drive.google.com/uc?id=1-82xRiQCGyzIeO4krbPtxvRde-s4fo-b', os.path.join(path_models, 'model_5.h5'), quiet=True)

def predict(model, test_df, model_name):
    print('predict ...')
    pred1 = []
    pred2 = []
    pred3 = []
    pred4 = []
    for img in test_df['path']:
        image1 = imread(img)
        image2 = np.fliplr(image1)
        image3 = np.flipud(image1)
        image4 = np.flipud(image2)

        image1 = resize(image1, (img_rows, img_cols, 3))
        image2 = resize(image2, (img_rows, img_cols, 3))
        image3 = resize(image3, (img_rows, img_cols, 3))
        image4 = resize(image4, (img_rows, img_cols, 3))

        pred1.append(model.predict(image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2])))[0][0])
        pred2.append(model.predict(image2.reshape((1, image2.shape[0], image2.shape[1], image2.shape[2])))[0][0])
        pred3.append(model.predict(image3.reshape((1, image3.shape[0], image3.shape[1], image3.shape[2])))[0][0])
        pred4.append(model.predict(image4.reshape((1, image4.shape[0], image4.shape[1], image4.shape[2])))[0][0])

    test_df['path'] = [*map(lambda x: x.split('/')[2], test_filelist)]
    
    test_df['pred'] = pred1
    test_df.columns = [0, 2]
    test_df.to_csv(os.path.join(tmp_predict, 'temp_1.csv'), index=False)

    test_df[2] = pred2
    test_df.to_csv(os.path.join(tmp_predict, 'temp_2.csv'), index=False)

    test_df[2] = pred3
    test_df.to_csv(os.path.join(tmp_predict, 'temp_3.csv'), index=False)
    
    test_df[2] = pred4
    test_df.to_csv(os.path.join(tmp_predict, 'temp_4.csv'), index=False)

    kaggle_bag(tmp_predict + 'temp_*.csv', os.path.join(tmp_predict, model_name.split('.')[0] + '.csv'))

path_models = 'models'
path_test = '/test'
tmp_predict = 'temp_predict'
path_submission = '/output'
img_rows = 224
img_cols = 224

if __name__ == '__main__':

    load_weights()

    test_filelist = glob.glob(os.path.join(path_test, '*.png'))
    test_df = pd.DataFrame()
    test_df['path'] = test_filelist

    for model_name in glob.glob(os.path.join(path_models, '*.h5')):
        try:
            del model
        except Exception:
            pass
        gc.collect()
        K.clear_session()

        print(model_name)
        model = load_model(model_name, custom_objects={'auc': auc, 'f1': f1})
        predict(model, test_df.copy(), model_name.split('/')[-1])

    kaggle_bag(tmp_predict + "/model*.csv", os.path.join(path_submission, 'submission1.csv'))
    os.remove(tmp_predict + "model_1.csv")
    os.remove(tmp_predict + "model_5.csv")
    kaggle_bag(tmp_predict + "/model*.csv", os.path.join(path_submission, 'submission2.csv'))
