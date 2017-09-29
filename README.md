## Natural Language Generation using CNN as the encoder ##

NLG for [E2E dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/) 

### Dataset Example ###
![### Dataset Example ###](https://github.com/superthierry/NLG/blob/CNN_Encoder/example.png)

### requirement ###
- [tensorflow==1.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [processed data](https://www.dropbox.com/s/6fdr5tjmbsios2e/raw_data.pickle?dl=0)

### Model Result ###
- **Bleu**: 0.7135
- **NIST**: 8.4988
- **METEOR**：0.4734
- **ROUGE-L**：0.7318
- **CIDEr**: 2.3468

### Official Baseline ###
- **Bleu**: 0.6986
- **NIST**: 8.5649
- **METEOR**：0.4706
- **ROUGE-L**：0.7279
- **CIDEr**: 2.3934

## attention meomry cell could improve BLEU to 71.92, however, decrease other metrics scores. Still need some more experiments ##





