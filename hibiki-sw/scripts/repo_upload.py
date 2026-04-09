import json, os, subprocess

KAGGLE_USERNAME = 'victormugambi'  # CHANGE THIS (same as above)

# Path to the hibiki-sw directory in your local clone
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
print(f'Repo root: {REPO_ROOT}')

repo_metadata = {
    'title': 'hibiki-sw-code',
    'id': f'{KAGGLE_USERNAME}/hibiki-sw-code',
    'licenses': [{'name': 'other'}]
}
with open(f'{REPO_ROOT}/dataset-metadata.json', 'w') as f:
    json.dump(repo_metadata, f, indent=2)

print()
print('=== Upload repo to Kaggle ===')
print(f'kaggle datasets create -p {REPO_ROOT}')
print()
print('To update the repo after code changes:')
print(f'kaggle datasets version -p {REPO_ROOT} -m "Update code"')