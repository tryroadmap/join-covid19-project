# export TRAINING_DATA=input/train_folds.csv
export TRAINING_DATA=input/train.csv
export TEST_DATA=input/test.csv
export PARAMS=input/params.json

export MODEL=$1

FOLD=0 python -m src.train
#add more folds
python -m src.predict
