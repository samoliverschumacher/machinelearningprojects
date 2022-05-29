This project was created to help me test various features of the [LSTM builder I developed.](https://github.com/samoliverschumacher/neuralnets/tree/main/rnngenerator/)

**The problem:** Convert a 3-digit number into its equivalent roman numeral.

`romannumeralNNet.m` uses the LSTM builder to create an encoder-decoder sequence to sequence LSTM 
model to convert one-hot encoded numbers into a one-hot encoded roman numeral.

Features tested;
1. Attention mechanism
2. Bidirectional LSTM with Attention
3. regular encoder-decoder LSTM

All are encoder-decoders;
- Embedding layer: one-hot encoded strings to vector representation
- Reccurrent LSTM layer encoder
- Reccurent LSTM layer decoder which takes its own sequential outputs as inputs to next timestep, until prediction gets to end of output
- projection layer; takes RNN hidden unit output and projects it to the one-hot encoded vector space

