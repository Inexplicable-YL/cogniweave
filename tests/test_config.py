from cogniweave.config import get_config, init_config

init_config(_config_file="./.cache/config.toml")

print(get_config())
