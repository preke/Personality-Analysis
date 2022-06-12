from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-large")

model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-large")


[[0.5277777777777778, 0.5555555555555556, 0.5416666666666666, 0.5694444444444444, 0.5138888888888888, 0.5694444444444444, 0.5555555555555556, 0.5833333333333334, 0.5833333333333334, 0.5277777777777778]]