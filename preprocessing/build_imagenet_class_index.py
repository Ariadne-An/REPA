import json
from pathlib import Path
import timm

info_dir = Path(timm.__file__).parent / 'data' / '_info'

synset_file = info_dir / 'imagenet_synsets.txt'
lemma_file = info_dir / 'imagenet_synset_to_lemma.txt'

synsets = [line.strip() for line in synset_file.open('r') if line.strip()]
lemmas = {}
for line in lemma_file.open('r'):
    line = line.strip()
    if not line or '\t' not in line:
        continue
    syn, lemma = line.split('\t', 1)
    lemmas[syn] = lemma

mapping = {}
for idx, syn in enumerate(synsets):
    mapping[str(idx)] = [syn, lemmas.get(syn, '')]

out_path = Path('assets/imagenet_class_index.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(mapping, indent=2))
print(f'Wrote {out_path} with {len(mapping)} entries')
