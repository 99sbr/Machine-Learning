
from __future__ import absolute_import
from __future__ import division

import pandas as pd
import numpy as np

import tempfile
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# def wide_preprocessing():
#   from sklearn.decomposition import PCA
#   pca2=PCA(10)
#   temp=pca2.fit_transform(train[COLUMNS])
#   temp2 = train['Y']
#   col = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
#   train = pd.DataFrame(temp,columns = col)
#   train['Y'] = temp2
#   return train,col

def input_fn(df):
  COLUMNS = list(df.columns)
  LABEL_COLUMN = "Y"
  #COLUMNS.remove("Y")
  CONTINUOUS_COLUMNS = COLUMNS
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  # Converts the label column into a constant Tensor.
  if(LABEL_COLUMN in df.columns):
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label
  else:
    return feature_cols


def build_estimator(model_type,train):
  COLUMNS = list(train.columns)
  LABEL_COLUMN = "Y"
  COLUMNS.remove("Y")
  CONTINUOUS_COLUMNS = COLUMNS
  deep_columns = [tf.contrib.layers.real_valued_column(k) for k in COLUMNS]
  wide_columns = [tf.contrib.layers.real_valued_column(k) for k in COLUMNS]

  if model_type == "wide":
    # global train
    # train,wide_columns = wide_preprocessing()
    # print train.head()
    print("WIDE learning is applying")
    m = tf.contrib.learn.LinearClassifier(
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    print("DEEP learning is applying")
    m = tf.contrib.learn.DNNClassifier(
                                       feature_columns=deep_columns,
                                       hidden_units=[500, 250, 101])
    
  else:
    print("WIDE+DEEP both learnings are applying")
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[500, 250, 101])

  return m



def k_fold(m,train,test):
  from sklearn.cross_validation import KFold
  n = len(train)
  n_folds = 5
  acc = []
  kf = KFold(n,n_folds,shuffle=False,random_state=40)
  for train_index,test_index in kf:
      x = np.array(train)
      x_train,x_test = x[train_index],x[test_index]
      df_train = pd.DataFrame(x_train,columns=train.columns)
      df_test = pd.DataFrame(x_test,columns=train.columns)
      m.fit(input_fn=lambda: input_fn(df_train), steps=100)
      results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
      acc.append(results['accuracy'])
      print(results['accuracy'])
  print '\n',np.mean(np.array(acc))
  y_pred = m.predict(input_fn=lambda: input_fn(test))
  return y_pred

def write_txt(y_pred,test_time):
  df_output = pd.DataFrame({'Time':test_time,'Y':y_pred})
  df_output.loc[df_output['Y'] == 0,"Y"] = -1
  df_output.to_csv('pred2.txt', index=None, sep=',', mode='a')

def main(_):
  train = pd.read_csv("./Dataset/train.csv")
  test = pd.read_csv("./Dataset/test.csv")
  test_time = np.array(test.Time)
  train.loc[train["Y"] == -1,"Y"] = 0
  train.drop('Time',axis = 1,inplace=True)
  test.drop('Time',axis = 1,inplace=True)
  model_type = "deep"
  m = build_estimator(model_type,train)
  y_pred = k_fold(m,train,test)
  write_txt(y_pred,test_time)

if __name__ == "__main__":
  tf.app.run()