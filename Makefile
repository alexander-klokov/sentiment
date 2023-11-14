SRC=src

train:
	python3 $(SRC)/sentiment_train.py

predict:
	python3 $(SRC)/sentiment_predict.py
