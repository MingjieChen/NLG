# NLG
NLG for E2E dataset 

### requirement ###
- [tensorflow==1.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- pickle


processed data https://www.dropbox.com/s/6fdr5tjmbsios2e/raw_data.pickle?dl=0

model weights https://www.dropbox.com/s/xzj61pv1ektbbq7/train.zip?dl=0

download processed data and model weights, unzip the zip file, put them in the directory

**dagger.py restore model then evaluate dev data one by one as the same order in the dev data.**

__result.txt is compare between model output and task baseline__

![result is 67.7](https://github.com/superthierry/NLG/blob/master/result.png)
