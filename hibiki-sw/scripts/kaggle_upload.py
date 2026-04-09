import json, os, shutil, csv

csv.field_size_limit(10_000_000)

KAGGLE_USERNAME = 'victormugambi'  # CHANGE THIS

CV_CORPUS_DIR = os.path.expanduser('~/cv-corpus')
sw_dir = r'C:\Users\User\Downloads\common-voice-swahili\cv-corpus-25.0-2026-03-09\sw'

# Output directory containing only validated clips
validated_upload_dir = os.path.expanduser('~/cv-swahili-validated')
validated_clips_dir = f'{validated_upload_dir}/clips'
os.makedirs(validated_clips_dir, exist_ok=True)

# Read validated.tsv and collect clip filenames
validated_tsv = f'{sw_dir}/validated.tsv'
clip_filenames = set()
with open(validated_tsv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        clip_filenames.add(row['path'])

print(f'Validated clips referenced in validated.tsv: {len(clip_filenames)}')

# Copy only validated clips
src_clips_dir = f'{sw_dir}/clips'
copied = 0
missing = 0
for filename in clip_filenames:
    src = os.path.join(src_clips_dir, filename)
    dst = os.path.join(validated_clips_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1

print(f'Copied: {copied}, Missing: {missing}')

# Copy TSV files (validated + splits)
for tsv in ['validated.tsv', 'train.tsv', 'dev.tsv', 'test.tsv']:
    src = os.path.join(sw_dir, tsv)
    if os.path.exists(src):
        shutil.copy2(src, validated_upload_dir)
        print(f'Copied {tsv}')

# Write Kaggle dataset metadata
sw_metadata = {
    'title': 'cv-swahili',
    'id': f'{KAGGLE_USERNAME}/cv-swahili',
    'licenses': [{'name': 'CC-BY-4.0'}]
}
with open(f'{validated_upload_dir}/dataset-metadata.json', 'w') as f:
    json.dump(sw_metadata, f, indent=2)

upload_size_gb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, files in os.walk(validated_upload_dir)
    for f in files
) / (1024 ** 3)
print(f'\nUpload directory size: {upload_size_gb:.1f} GB')
print(f'Ready to upload: {validated_upload_dir}')
print()
print('=== Run this in your terminal to upload ===')
print(f'kaggle datasets create -p {validated_upload_dir} --dir-mode tar')