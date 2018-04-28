# QuantNN TODO
## Code
### programs
- [traintest_seq2seq_model] finish writing code to compute performance metrics on model performance on the test set.

### core
- Allow writing/restoring only parameter *values* to/from dict rather than all parameter information (such as value bounds and parameter type).

### ML
- In the `EncoderDecoderNetwork` class `target_sequence` should not be required for inference. So we should be able to call tensorflow_session.run() without a `target_sequence` as variable in `feed_dict`.
- Change fit() method in ML models to accept an object that supplies input/target sequences on demand rather than passing the precomputed sequences as a large numpy array. This will reduce memory requirements significantly.
- Allow restoring saved graphs and connecting nodes from saved graphs with new nodes, with the option to either propagate or stop gradient flow to the restored trainable variables.

### viz
- Write code to plot model training curves and use that in the `traintest_seq2seq_model` problem and also call it at every *val* step in the encoder-decoder ML model.

### networking
- Write websocket feed client for IEX.
- Write websocket feed client for bitstamp.
- Write socket server.
