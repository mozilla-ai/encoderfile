
from encoderfile import read_metadata

info = read_metadata("encoderfile-py/python/tests/sentiment-analyzer.encoderfile")

print(info.encoderfile_config.name)        # "sentiment-analyzer"
print(info.encoderfile_config.model_type)  # "sequence_classification"
print(info.encoderfile_config.version)     # "1.0.0"
print(info.model_config.id2label)          # {0: "NEGATIVE", 1: "POSITIVE"}