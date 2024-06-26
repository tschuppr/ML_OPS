## Installing packages

## Dataset for the Tutorial

```python
import pandas as pd
df = pd.read_csv('labelled_train.csv')
df.head()
```

## Using DVC to version the data

Data Version Control (DVC) is a data versioning software that works on top of
Git. Data versioning is enabled by replacing large files with small metafiles,
which can be easily tracked by Git. These small metafiles point to the original
large files, which are stored on-premise or on cloud storage. These files are
thus decoupled from the code base.


Typical DVC workflow:
- Initialize a DVC project in a Git repo with dvc init.
- Copy data files or dataset directories for modeling into the project and use dvc add to tell DVC to cache and track them.
- Create a simple dvc.yaml file to codify a data processing pipeline. It uses your own source code and specifies further data outputs for DVC to control.
- Execute or restore any version of your pipeline using dvc repro, or experiment on it with dvc exp features.
- Sharing the repository will not include locally cached data. Use remote storage with dvc push and dvc pull to share data artifacts.

```shell
dvc init
dvc import https://github.com/tschuppr/ML_OPS/ data
git rm -r --cached 'data'
dvc add data
```

Remote Storage: Google Drive
```shell
dvc remote add myremote gdrive://appDataFolder
dvc remote default myremote
dvc push
```
## PyCaret

We use a XGBoost model to compare performances.
```shell
pip install xgboost
```

Raw data import without pre-process
```python
from pycaret.classification import *
import pandas as pd
df = pd.read_csv('data/labelled_train.csv')
s = setup(data=df, target='Survived')
model = create_model('xgboost')
# evaluate_model(model) can only be used in Notebook
plot_model(model, plot = 'auc')
plot_model(model, plot = 'confusion_matrix')
predict_model(model, data=pd.read_csv('data/unlabelled_test.csv'), raw_score=True)
```

Data pre-process
```python
from pycaret.classification import *
import pandas as pd
df = pd.read_csv('data/labelled_train.csv')
s = setup(data=df,
          target='Survived',
          numeric_imputation='mean',
          categorical_features=['Sex', 'Embarked'],
          ignore_features=['Name', 'Ticket', 'Cabin', 'Parch'],
          remove_outliers=True)
create_model('xgboost', enable_categorical=True)
```

Export
```python
from pycaret.classification import *
import pandas as pd
df = pd.read_csv('data/labelled_train.csv')
s = setup(data=df,
          target='Survived',
          numeric_imputation='mean',
          categorical_features=['Sex', 'Embarked'],
          ignore_features=['Name', 'Ticket', 'Cabin', 'Parch'],
          remove_outliers=True)
s.get_config()
transformed_df = s.get_config('dataset_transformed')
transformed_df.to_csv('data/transformed_labelled_train.csv', index=False)
```

## PyCaret additions

- tune_model
- compare_models
- interpret_model
- calibrate_model