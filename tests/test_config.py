from cogniweave import get_config, init_config

init_config(_config_file="./tests/config.toml")

print(get_config())
