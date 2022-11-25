#! /bin/sh
#
# Preprocessing for Inuktutit subwpord tokenization
#
trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

TMP=./tmp
mkdir -p $TMP

#
# Download the SentencePiece model of mBART-50.
#
wget -P $TMP https://huggingface.co/facebook/mbart-large-50/resolve/main/sentencepiece.bpe.model

#
# Convert the binary model into the text model.
#
spm_export_vocab --model=$TMP/sentencepiece.bpe.model --output=$TMP/mbart50.vocab

#
# Download a Inktutit corpus used at WMT20.
# This was extracted from CommonCrawl.
#
wget -P $TMP http://web-language-models.s3.amazonaws.com/wmt20/deduped/iu.xz
xz -dc $TMP/iu.xz > $TMP/commoncrawl.raw.iu
rm -f $TMP/iu.xz
