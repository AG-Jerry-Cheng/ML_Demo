import numpy, pandas as pd, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
numpy.random.seed(10)

def PreprocessData(raw_df):
    df = raw_df
    x_OneHot_df = pd.get_dummies(data=df, columns=["fico_band", "platform"])
    #x_OneHot_df = pd.get_dummies(data=df, columns=["fico_band"])
    ndarray = x_OneHot_df.values
    Features = ndarray[:, 1:]
    Label = ndarray[:, 0]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, Label


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    all_df = pd.read_excel("data/cams_data_full.xls")
    cols=['result', 'fico_score', 'platform', 'down_payment', 'loan_amount', 'fico_band', 'age', 'year',
          'monthly_income', 'estimated_monthly_payment', 'payment_to_income', 'loan_to_value',
          'bankruptcies_on_record', 'debt_to_income_ratio', 'current_delinquencies', 'vehicle_msrp']
    all_df = all_df[cols]
    msk = numpy.random.rand(len(all_df)) < 0.8
    train_df = all_df[msk]
    test_df = all_df[~msk]
    print('total:', len(all_df),
          'train:', len(train_df),
          'test:', len(test_df))

    train_Features, train_Label = PreprocessData(train_df)
    test_Features, test_Label = PreprocessData(test_df)

    model = Sequential()

    # create input layer
    model.add(Dense(units=40, input_dim=21,
                    kernel_initializer='uniform',
                    activation='relu'))

    # create hidden layer
    model.add(Dense(units=30,
                    kernel_initializer='uniform',
                    activation='relu'))

    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    # create output layer
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    train_history = model.fit(x=train_Features,
                              y=train_Label,
                              validation_split=0.1,
                              epochs=30,
                              batch_size=30, verbose=2)

    #show_train_history(train_history, 'acc', 'val_acc')

    #show_train_history(train_history, 'loss', 'val_loss')

    # save model as JSON
    model_json = model.to_json()
    with open("model_store/model.json", "w") as json_file:
        json_file.write(model_json)

    # save model as HDF5
    model.save("model_store/model.h5")
    print("Saved model to 'model_store/' ")

    # load model (HDF5)
    loaded_model = tf.contrib.keras.models.load_model('model_store/model.h5')
    print("Loaded HDF5 model from disk 'model_store/' ")

    # load model (json)
    # json_file = open('model_store/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # print("Loaded json model from disk 'model_store/' ")

    # evaluate loaded model on test data
    score = loaded_model.evaluate(x=test_Features, y=test_Label, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))