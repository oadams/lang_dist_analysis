from pathlib import Path

import torch
import tqdm
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
# Italian Example
#out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-it.wav')
#print(text_lab)

# French Example
#out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-fr.wav')
#print(text_lab)

def embed_audio(audio_path, classifier):
    waveform = classifier.load_audio(audio_path)
    # Fake a batch:gg
    batch = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    return classifier.encode_batch(batch, rel_length)

#print(embed_audio('speechbrain/lang-id-commonlanguage_ecapa/example-it.wav', classifier).shape)

language_codes = [
    "ar", "eu", "br", "ca", "zh", "zh", "zh", "cv", "cs", "dv", 
    "nl", "en", "eo", "et", "fr", "fy", "ka", "de", "el", "cnr", 
    "id", "ia", "it", "ja", "kab", "rw", "ky", "lv", "mt", "mn", 
    "fa", "pl", "pt", "ro", "rm", "ru", "sah", "sl", "es", "sv", 
    "ta", "tt", "tr", "uk", "cy"
]

#for code in language_codes:
#    embed_audio(f'speechbrain/lang-id-commonlanguage_ecapa/example-{code}.wav', classifier)
#    out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-fr.wav')
#    print(text_lab)

from torch.nn.utils.rnn import pad_sequence


def encode_lang(lang, split, classifier):
    lang_path = Path('common_voice_kpd') / lang
    lang_embs = []
    for audio_path in (lang_path / split).glob('**/*.wav'):
        waveform = classifier.load_audio(str(audio_path))
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        emb = classifier.encode_batch(batch, rel_length)
        lang_embs.append(emb)
    emb = torch.cat(lang_embs).squeeze(1)
    emb_dir = Path('lang_embs_2')
    emb_dir.mkdir(exist_ok=True)
    torch.save(emb, emb_dir / f'{lang}_{split}_emb.pt')

for path in tqdm.tqdm(list(Path('common_voice_kpd').glob('*'))):
    lang = path.name
    encode_lang(lang, 'test', classifier)