import hashlib


def hash_obs(observation, size=16):
    """Compute a hash that uniquely identifies the current state of the environment.
    :param size: Size of the hashing
    """
    sample_hash = hashlib.sha256()

    img = observation['image']
    agent_dir = observation['direction']
    to_encode = [img.tolist(), agent_dir]
    for item in to_encode:
        sample_hash.update(str(item).encode('utf8'))

    return sample_hash.hexdigest()[:size]