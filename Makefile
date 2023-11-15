SRC=src

train:
	python3 $(SRC)/sentiment_train.py

test:
	python3 $(SRC)/sentiment_test.py

predict:
	python3 $(SRC)/sentiment_predict.py "$(phrase)"
