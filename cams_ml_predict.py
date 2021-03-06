import numpy, pandas as pd
from keras.models import load_model
from sklearn import preprocessing
numpy.set_printoptions(threshold=numpy.nan)

def GetPredictionData():
    all_df = pd.read_excel("data/cams_data_full.xls")
    cols = ['result', 'fico_score', 'platform', 'down_payment', 'loan_amount', 'fico_band', 'age', 'year',
            'monthly_income', 'estimated_monthly_payment', 'payment_to_income', 'loan_to_value',
            'bankruptcies_on_record', 'debt_to_income_ratio', 'current_delinquencies', 'vehicle_msrp']
    all_df = all_df[cols]
    msk = numpy.random.rand(len(all_df)) < 0.8
    df = all_df[~msk]
    x_OneHot_df = pd.get_dummies(data=df, columns=["fico_band", "platform"])
    #x_OneHot_df = pd.get_dummies(data=df, columns=["fico_band"])
    ndarray = x_OneHot_df.values
    Features = ndarray[:, 1:]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures


if __name__ == "__main__":
    predict_Features = GetPredictionData()

    # load model and prediction
    model = load_model('model_store/model.h5')
    # load model - json
    # with open('model_store/model.json') as ff:
    #     model_json=ff.read()
    #     model=keras.models.model_from_json(model_json)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    preds = model.predict(predict_Features)
    print (preds)