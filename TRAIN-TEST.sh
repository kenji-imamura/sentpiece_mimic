#! /bin/sh
#
# Additional training of a SPM model for Inuktutit subword tokenization
#

trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

SCRIPTS=./src
TMP=./tmp
VOCAB_SIZE=8000

#
# Training
#
cat $TMP/commoncrawl.raw.iu \
    | spm_normalize --use_internal_normalization \
      		    --normalization_rule_name nmt_nfkc \
    > $TMP/commoncrawl.norm.iu

python3 $SCRIPTS/spm_add_train.iu.py \
	--input $TMP/commoncrawl.norm.iu \
	--spm_model $TMP/mbart50.vocab \
	--add_model $TMP/add-iu.vocab \
	--vocab_size $VOCAB_SIZE

cat $TMP/mbart50.vocab $TMP/add-iu.vocab \
    > $TMP/mbart50+iu.vocab

#
# Test
#
cat $TMP/commoncrawl.norm.iu \
    | head -n 100 \
    | python3 $SCRIPTS/spm_encode.py \
	      --model $TMP/mbart50+iu.vocab

