import secrets
import string
import json
from datetime import datetime

def generate_api_key(prefix: str = "ws") -> str:
    """
    Generate a secure API key
    Format: prefix_environment_randomstring
    Example: ws_test_Ab3Cd5Ef7Gh9Ij2Kl4Mn6Op8Qr0St
    """
    alphabet = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(32))
    return f"{prefix}_{random_part}"

def generate_keys_for_environments():
    """Generate API keys for all environments"""
    keys = {}

    # Development key
    keys['development'] = {
        'key': generate_api_key('ws_dev'),
        'created_at': datetime.now().isoformat(),
        'environment': 'development',
        'rate_limit': 100
    }

    # Testing keys (multiple for QC team)
    keys['testing'] = []
    for i in range(3):  # Generate 3 testing keys
        keys['testing'].append({
            'key': generate_api_key('ws_test'),
            'created_at': datetime.now().isoformat(),
            'environment': 'testing',
            'team': f'QC_Team_{i+1}',
            'rate_limit': 50
        })

    # Production key
    keys['production'] = {
        'key': generate_api_key('ws_prod'),
        'created_at': datetime.now().isoformat(),
        'environment': 'production',
        'rate_limit': 1000
    }

    # Save to secure location
    with open('api_keys_generated.json', 'w') as f:
        json.dump(keys, f, indent=2)

    print("Generated API keys:")
    print(f"Development: {keys['development']['key']}")
    for test_key in keys['testing']:
        print(f"Testing ({test_key['team']}): {test_key['key']}")
    print(f"Production: {keys['production']['key']}")

    return keys

if __name__ == "__main__":
    generate_keys_for_environments()