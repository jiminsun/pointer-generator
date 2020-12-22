# Pointer-Generator Network

This directory contains the Pytorch implementation of the Pointer-Generator Network for text summarization, presented in [Get To The Point: Summarization with Pointer-Generator Networks (See et al., 2017)](https://arxiv.org/abs/1704.04368).

While the original paper trains the model on an English dataset, this project aims at building a *Korean* summarization model. 
Thus, we additionally incorporate Korean preprocessing & tokenization techniques to adapt the model to Korean.

Most of the code is implemented from scratch, but we also referred to the following repositories. 
Any direct references are mentioned explicitly on the corresponding lines of code.
* https://github.com/abisee/pointer-generator - the original author's implementation in tensorflow
* https://github.com/atulkum/pointer_summarizer
* https://github.com/rohithreddy024/Text-Summarizer-Pytorch

Note that the overall pipeline relies on `pytorch-lightning`.


## Requirements

```
pip install -r requirements.txt
```

## How to run
### Prepare data

TBU

### Run training
First, set up the desired model configurations in `config.json`.

To begin training your model, run:
```
python train.py
```

Details on optional command-line arguments are specified below:
```
Pointer-generator network

optional arguments:
  -h, --help            show this help message and exit
  -cp CONFIG_PATH, --config-path CONFIG_PATH
                        path to config file
  --mds {combi}         multi-news labeling method to employ. if None, nikl dataset is used.
  -m MODEL_PATH, --model-path MODEL_PATH
                        path to load model in case of resuming training from an existing checkpoint
  --load-vocab          whether to load pre-built vocab file
  --stop-with {loss,r1,r2,rl}
                        validation evaluation metric to perform early stopping
  -e EXP_NAME, --exp-name EXP_NAME
                        suffix to specify experiment name
  -d DEVICE, --device DEVICE
                        gpu device number to use. if cpu, set this argument to -1
  -n NOTE, --note NOTE  note to append to result output file name
```
Running the file will create a subdirectory in `logs` with the experiment name.
All checkpoints, test set predictions, the constructed vocab file, tensorboard logs, and hyperparameter configurations will be saved in this directory.

### Run evaluation

```
python test.py --model-path $PATH-TO-CHECKPOINT
```

This will report the ROUGE scores on the command-line and save the predicted outputs in `.tsv` format in the experiment directory where you have loaded the checkpoint.
 
