import munch
import toml

def get_config(filepath: str = 'configs/default.toml'):
    return munch.munchify(
        toml.load(filepath)
    )