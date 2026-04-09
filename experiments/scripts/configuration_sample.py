import random
import argparse

parser = argparse.ArgumentParser(
    prog='RandomConfigurationSampling',
    description='Sample K new random configurations to be added to the configuration file.',
    epilog='Pass as argument the name of the configuration file.'
)

parser.add_argument("-c", "--configuration",
                    type=str,
                    required=True,
                    help="Absolute path to configuration file.")

args = parser.parse_args()

def read_existing_configs(filepath):
    """Read existing configurations from file."""
    configs = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    configs.add(line)
    except FileNotFoundError:
        # If file doesn't exist yet, start empty
        pass
    return configs

def generate_random_config(n_bits=11):
    """Generate a random configuration string like c01010101010."""
    bits = ''.join(str(random.randint(0, 1)) for _ in range(n_bits))
    return f'c{bits}'

def sample_new_configs(existing_configs, K, n_bits=11):
    """Generate K new unique configurations not already in existing_configs."""
    new_configs = set()

    while len(new_configs) < K:
        config = generate_random_config(n_bits)
        if config not in existing_configs and config not in new_configs:
            new_configs.add(config)

    return new_configs

def append_configs_to_file(filepath, configs):
    """Append new configurations to the file."""
    with open(filepath, 'a') as f:
        for config in configs:
            f.write(config + '\n')

if __name__ == "__main__":
    K = 8

    current_configurations = read_existing_configs(args.configuration)
    sample_configurations = sample_new_configs(current_configurations, K)
    append_configs_to_file(args.configuration, sample_configurations)

    print(f"Added {len(sample_configurations)} new configurations.")
