# QuantNN: getting started

## Install python dependencies
```sh
cd src
pip3 install --upgrade -r ./mkdocs_requirements.txt
pip3 install --upgrade -r ./requirements.txt
```

## Run the local documentation server
```sh
# Run this in root directory of QuantNN.
mkdocs serve
```
Now go to http://localhost:8000

## Download historical data
```sh
# You only have to run these two lines once per terminal session
cd bin
export PYTHONPATH=$PYTHONPATH:/projects/QuantNN/src

# Download 15 minute bars from poloniex
# Argument is timeframe in seconds. Valid values for this program are 300, 900, 1800, 7200, 14400, and 86400.
python3 ../src/qnn/programs/download_poloniex_bardata.py 900
```

## Test a model on a problem
```sh
# You only have to run these two lines once per terminal session
cd bin
export PYTHONPATH=$PYTHONPATH:/projects/QuantNN/src

# Test (this loads the seq2seq model configuration file "QNNSeq1" and uses it on problem file "eth_btc_problem1")
python3 ../src/qnn/programs/traintest_seq2seq_model.py eth_btc_problem1 QNNSeq1
```
