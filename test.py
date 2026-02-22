import encoderfile as ef

print(ef.__all__)

ef.EncoderfileBuilder.from_config("test_config.yml").build()
