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

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

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
    #gdown.download('https://drive.google.com/uc?id=15TKOlGEgk-3m6R8TeMzFxn_RQzwB6hjM', os.path.join(path_models, 'model_1.h5'), quiet=True)
    #gdown.download('https://drive.google.com/uc?id=1bhLypjl0BNtaFSjOLlICO0ecCkZ-rNKF', os.path.join(path_models, 'model_2.h5'), quiet=True)
    #gdown.download('https://drive.google.com/uc?id=1hrkhV4JdN_JpHkkxpGnBVGeBzKLnBRMz', os.path.join(path_models, 'model_3.h5'), quiet=True)
    #gdown.download('https://drive.google.com/uc?id=1Giv-wG23AqZ9IEnwfDUJlHULPn7Tsb8C', os.path.join(path_models, 'model_4.h5'), quiet=True)
    #gdown.download('https://drive.google.com/uc?id=1-82xRiQCGyzIeO4krbPtxvRde-s4fo-b', os.path.join(path_models, 'model_5.h5'), quiet=True)

def predict(model, test_df, model_name):
    print('predict ...')

    pred = []
    for img in test_df['path']:
        image = imread(img)
        image = resize(image, (img_rows, img_cols, 3))
        crop_image = image
        pred.append(
            model.predict(crop_image.reshape((1, crop_image.shape[0], crop_image.shape[1], crop_image.shape[2])))[0][0])

    test_df['path'] = [*map(lambda x: x.split('/')[2], test_filelist)]
    test_df['pred'] = pred
    test_df.columns = [0, 2]
    test_df.to_csv(os.path.join(tmp_predict, model_name.split('.')[0]+'.csv'), index=False)

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
    #os.remove("model1.txt")
    #os.remove("model5.txt")
    kaggle_bag(tmp_predict + "/model*.csv", os.path.join(path_submission, 'submission2.csv'))
