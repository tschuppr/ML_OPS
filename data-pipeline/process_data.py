import argparse

import pandas as pd
from pycaret.classification import *


def main(file_name, is_labeled):
    df = pd.read_csv('data/' + file_name)
    if is_labeled:
        kwargs = {'ignore_features': ['Name', 'Ticket', 'Cabin', 'Embarked']}
        kwargs.update({'target': 'Survived'})
    else:
        kwargs = {'ignore_features': ['Name', 'Ticket', 'Cabin']}
    s = setup(data=df,
              numeric_imputation='mean',
              categorical_features=['Sex'],
              remove_outliers=True,
              **kwargs)
    s.get_config('dataset_transformed').to_csv('data_outputs/transformed_' + file_name, index=False)
    if is_labeled:
        s.get_config('X_train_transformed').to_csv('data_outputs/X_train_transformed.csv', index=False)
        s.get_config('y_train_transformed').to_csv('data_outputs/y_train_transformed.csv', index=False)
        s.get_config('X_test_transformed').to_csv('data_outputs/X_test_transformed.csv', index=False)
        s.get_config('y_test_transformed').to_csv('data_outputs/y_test_transformed.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fn", dest="file_name",
                        required=True, help="Input CSV file name")
    parser.add_argument("-l", "--lb", dest="is_labeled",
                        required=False, help="boolean label")
    args = parser.parse_args()
    main(args.file_name, args.is_labeled)
