# QuantNN TODO
## Code

### programs
- [traintest_seq2seq_model] finish writing code to compute performance metrics on model performance on the test set.

### core
- Restore model without defining graph for real-time predictions: (https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125) (https://www.tensorflow.org/programmers_guide/saved_model)

### ML
- In the `EncoderDecoderNetwork` class `target_sequence` should not be required for inference. So we should be able to call tensorflow_session.run() without a `target_sequence` as variable in `feed_dict`.
- Change fit() method in ML models to accept an object that supplies input/target sequences on demand rather than passing the precomputed sequences as a large numpy array. This will reduce memory requirements significantly.

### viz
- Write code to plot model training curves and use that in the `traintest_seq2seq_model` problem and also call it at every *val* step in the encoder-decoder ML model.

### networking
- Write websocket feed client for IEX.
- Write websocket feed client for bitstamp.
- Write socket server.

### data
- Required data:
	1. CEXIO 1-min chart data or trade data
	2. GDAX 1-min chart data
	3. KRAKEN trade data
	4. POLONIEX trade data
	
- Optional data:
	1. BITFINEX 1-min chart data (rate limited)
	2. ETHEREUM BLOCKCHAIN data (need a way to track ERC20 token transfers)
	3. BINANCE 1-min chart data (starts at Jan 2018, with gaps during mid Feb)
	4. WAVES 1-min chart data (low volume)
	5. BLINKTRADE trade data (low volume; foreign exchanges)
	6. BTCC 1-min chart data (low volume; this Chinese-based exchange was shut down recently)

- Save normalizing factors (z-score: mean, std; truncate: mean, std) and scaling factors (scaling log-ratio to avoid rounding too much via float32 operations) used on training data to normalize inference data.
- Implement a fast thread-safe database that can append new data.

### trading
- Record recent orderbook data & trade data for exchanges that will be used for trading.
