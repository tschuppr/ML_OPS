import pandas as pd
from pycaret.classification import *

if __name__ == "__main__":
    df = pd.read_csv('data_outputs/transformed_labelled_train.csv')
    s = setup(data=df)
    model = create_model('xgboost')
    predictions_df = predict_model(model, data=pd.read_csv('data_outputs/transformed_unlabelled_test.csv'), raw_score=True)
    predictions_df.to_csv('data_outputs/xgboost_predictions.csv', index=False)
