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
git rm -r --cached 'data'
dvc add data
```
