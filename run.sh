# export TRAINING_DATA=input/train_folds.csv
export TRAINING_DATA=input/train.csv
export TEST_DATA=input/test.csv
export WORLD_POPULATION=input/population_by_country_2020.csv
# export PARAMS=input/params.json
export COUNTRY=$2
export MODEL=$1

# FOLD=0 python -m src.train
# #add more folds
# python -m src.predict

# python3 src/logistic_predict.py

if [ $MODEL == "linear" ];then 
	python3 src/predict.py
elif [ $MODEL == "logistic" ]; then 
	python3 src/logistic_predict.py
elif [ $MODEL == "dataset" ]; then
	python3 src/dataset.py 
elif [ $MODEL == "tendency" ]; then
	python3 src/tendency.py 
fi 
