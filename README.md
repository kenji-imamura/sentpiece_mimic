# Re-implementation of SentencePiece in Python and Additional Training of Inuktitut Subwords

Thiss program is re-implementation of the core of
[SentencePiece](https://github.com/google/sentencepiece) in python.
It must be helpful to learn the behavior of SentencePiece and/or to
study new subword tokenization.

This program was used in the paper
[(Imamura and Sumita, 2022)](to appear),
which additionally learned Inuktitut subwords without changing existing models.

***
## Usage as the Mimic of SentencePiece

### Requirements
The original [SentencePiece](https://github.com/google/sentencepiece) is necessary to run this program,

### Training
`spm_normalize` is a SentencePiece command.
The program of `spm_train.py` is very slow in comparison with the original `spm_train` command.
```
% cat $TRAIN_CORPUS \
      | spm_normalize --use_internal_normalization \
      		      --normalization_rule_name nmt_nfkc \
      > $TRAIN_CORPUS.norm
% python3 ./src/spm_train.py -i $TRAIN_CORPUS.norm -m $MODEL_PREFIX.model -v $VOCAB_SIZE
```

The above script is (almost) the same as the following SentencePiece command.
```
% spm_train --input=$TRAIN_CORPUS --model_prefix=$MODEL_PREFIX --vocab_size=$VOCAB_SIZE \
  	    --character_coverage=1.0 --model_type=unigram
```

### Tokenization
```
% cat $TEST_CORPUS \
      | spm_normalize --use_internal_normalization \
      		      --normalization_rule_name nmt_nfkc \
      | python3 ./src/spm_encode.py -m $MODEL_PREFIX.model \
      > $TEST_CORPUS.bpe
```

The above script is the same as the following SentencePiece command.
```
% cat $TEST_CORPUS \
      | spm_encode --model=$MODEL_PREFIX.model \
      > $TEST_CORPUS.bpe
```

### Detokenization
```
% cat $TEST_CORPUS.bpe \
      | python3 ./src/spm_clean.py
```

The above script is the same as the following SentencePiece command.
```
% cat $TEST_CORPUS.bpe \
      | spm_decode --model=$MODEL_PREFIX.model
```

***
## Adding Subwords to a SentencePiece Model for Multilingual Pretrained Models (Imamura and Sumita, 2022)

In [this paper](to appear),
we learned Inuktitut subwords in addition to the original SentencePiece model attached to the pretrained [mBART-50](https://arxiv.org/abs/2008.00401) model.  Inuktitut was not included in the original models.  Note that we did not use Huggingface's implementation in our paper even if we used mBART-50.

### Preprocessing

The `PREPROC.sh` script sequentially run the following commands.

- Download the SentencePiece model attached to mBART-50.
```
% wget https://huggingface.co/facebook/mbart-large-50/resolve/main/sentencepiece.bpe.model
```

- Convert the model into the text one.
```
% spm_export_vocab --model=sentence.bpe.model \
  		   --output=mbart50.vocab
```

- Download the Inuktitut monolingual corpus used at WMT20.
```
wget http://web-language-models.s3.amazonaws.com/wmt20/deduped/iu.xz
xz -dc iu.xz > commoncrawl.raw.iu
```

### Learning Additional Model

In this phase, we use `spm_add_train.iu.py`, in which `spm_train.py` was customized to additional training of Inuktitut subwords.  The following example is the one that adds 8,000 subwords.  We finally obtain `mbart50+iu.vocab`, which is concatenated the original vocabulary of mBART-50 (`mbart50.vocab`) and that of additionally learned Inuktitut vocabulary (`add-iu.vocab`).
```
% cat commoncrawl.raw.iu \
      | spm_normalize --use_internal_normalization \
      		      --normalization_rule_name nmt_nfkc \
      > commoncrawl.norm.iu
% python3 ./src/spm_add_train.iu.py \
  	  -i commoncrawl.norm.iu -v 8000 \
  	  --spm_model mbart50.vocab \
	  --add_model add-iu.vocab
% cat mbart50.vocab add-iu.vocab > mbart50+iu.vocab
```

### Test

`TRAIN-TEST.sh` is an example of additional training and its test.
```
% head -n 100 commoncrawl.norm.iu \
       | python3 ./src/spm_encode.py \
       	 	 --model mbart50+iu.vocab
```

## Citation
```bibtex
@misc{imamura-sumita-2022-sentpiece,
  title = "Extending the Subwording Model of Multilingual Pretrained Models for New Languages",
  author = "Imamura, Kenji and Sumita, Eiichiro",
  booktitle = "arXiv",
  month = Dec,
  year = "2022",
}
```
