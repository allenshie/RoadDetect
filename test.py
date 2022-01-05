import configparser
config = configparser.ConfigParser()
config.read("./config/1F.ini")
print(config['sql'])