# ML_OPS
ML_OPS courses given at Dauphine

## How to install

Clone the git in the folder you want to put the code into and enter the folder.
```sh
git clone https://github.com/tschuppr/ML_OPS.git
cd ML_OPS
```

Install the venv
```sh
python -m venv .
venv\Scripts\activate # On cmd windows
#activate venv\Scripts\activate # On linux
pip install -r requirements. txt
```

## Launch an MLFlow instance
After the environment has been activated you can launch an MLFlow instance using :
```sh
mlflow ui
```