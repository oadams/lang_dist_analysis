from collections import defaultdict
from pathlib import Path

import lang2vec.lang2vec as l2v
import pandas as pd
import seaborn as sns
import torch
import tqdm
from speechbrain.pretrained import EncoderClassifier


def encode_lang(lang_path, classifier):
    lang_path = Path(lang_path)
    lang = lang_path.name
    print('Encoding language: ', lang)
    emb_path = lang_path.parent / f'{lang}_emb.pt'
    if emb_path.exists():
        print(f'Already encoded {lang}, {emb_path} exists')
        return
    lang_embs = []
    for audio_path in tqdm.tqdm(list(lang_path.glob('**/*.wav'))):
        waveform = classifier.load_audio(str(audio_path))
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        emb = classifier.encode_batch(batch, rel_length)
        lang_embs.append(emb)
    emb = torch.cat(lang_embs).squeeze(1)
    torch.save(emb, emb_path)

"""
def encode_langs():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
    for path in tqdm.tqdm(list(Path('common_voice_kpd').glob('*'))):
        lang = path.name
        encode_lang(lang, 'test', classifier)

langs = []
embs = []
for path in tqdm.tqdm(list(Path('common_voice_kpd').glob('*'))):
    langs.append(path.name)
    emb = torch.load(f'lang_embs_2/{path.name}_test_emb.pt')
    emb = emb.squeeze(1).mean(0)
    embs.append(emb)
print(langs)
embs = torch.stack(embs)
embs = torch.nn.functional.normalize(embs, dim=1)
sims = embs @ embs.T

# Flatten the matrix to perform sorting
flattened_matrix = sims.view(-1)  # Flatten the 2D matrix into a 1D tensor
sorted_indices = torch.argsort(flattened_matrix, dim=0)

# Retrieve row and column indices based on sorted indices
row_indices = sorted_indices // sims.shape[1]
col_indices = sorted_indices % sims.shape[1]

language_codes = {
    "Arabic": "ara",
    "Basque": "eus",
    "Breton": "bre",
    "Catalan": "cat",
    "Chinese_China": "zho",
    "Chinese_Hongkong": "zho",
    "Chinese_Taiwan": "zho",
    "Chuvash": "chv",
    "Czech": "ces",
    "Dhivehi": "div",
    "Dutch": "nld",
    "English": "eng",
    "Esperanto": "epo",
    "Estonian": "est",
    "French": "fra",
    "Frisian": "fry",
    "Georgian": "kat",
    "German": "deu",
    "Greek": "ell",
    "Hakha_Chin": "cnr",  # No specific code in ISO 639-3, using ISO 639-2 code
    "Indonesian": "ind",
    "Interlingua": "ina",
    "Italian": "ita",
    "Japanese": "jpn",
    "Kabyle": "kab",
    "Kinyarwanda": "kin",
    "Kyrgyz": "kir",
    "Latvian": "lav",
    "Maltese": "mlt",
    "Mangolian": "mon",
    "Persian": "fas",
    "Polish": "pol",
    "Portuguese": "por",
    "Romanian": "ron",
    "Romansh_Sursilvan": "roh",
    "Russian": "rus",
    "Sakha": "sah",
    "Slovenian": "slv",
    "Spanish": "spa",
    "Swedish": "swe",
    "Tamil": "tam",
    "Tatar": "tat",
    "Turkish": "tur",
    "Ukranian": "ukr",
    "Welsh": "cym"
}
code2lang = {v: k for k, v in language_codes.items()}

def speechbrain_distance(lang1code, lang2code):
    lang1 = code2lang[lang1code]
    lang2 = code2lang[lang2code]

    lang1idx = langs.index(lang1)
    lang2idx = langs.index(lang2)
    return 1-sims[lang1idx, lang2idx].item()

def distance_df(distance_functions):
    records = []
    langpairs = []
    for i in tqdm.tqdm(range(len(langs))):
        for j in range(len(langs)):
            if i >= j:
                continue
            forget_it = False
            row_dists = []
            for f in distance_functions:
                try:
                    row_dists.append(f(language_codes[langs[i]], language_codes[langs[j]]))
                except:
                    forget_it = True
                    break
            if not forget_it:
                langpairs.append((langs[i], langs[j]))
                records.append(row_dists)
    df = pd.DataFrame(records, index=langpairs, columns=[f.__name__ for f in distance_functions])
    return df

#df = distance_df([speechbrain_distance, l2v.inventory_distance, l2v.geographic_distance])
#df
"""

def create_speechbrain_embeddings():
    for lang_path in Path('ATDS/5h-audio-all').glob('*'):
        if not lang_path.is_dir():
            continue
        print(lang_path)
        classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
        encode_lang(lang_path, classifier)

def compute_speechbrain_similarities():
    langs = []
    embs = []
    for path in tqdm.tqdm(list(Path('ATDS/5h-audio-all').glob('*.pt'))):
        lang = path.stem.split('_')[0]
        emb = torch.load(path)
        emb = emb.mean(0)
        embs.append(emb)
        langs.append(lang)
    print(langs)
    embs = torch.stack(embs)
    embs = torch.nn.functional.normalize(embs, dim=1)
    sims = embs @ embs.T
    records = []
    for i in range(len(langs)):
        for j in range(len(langs)):
            records.append((langs[i], langs[j], sims[i, j].item()))
    df = pd.DataFrame(records, columns=['ref_lang', 'comp_lang', 'speechbrain_similarity'])
    df.to_csv('speechbrain_similarities.csv')
    return df



if __name__ == '__main__':
    create_speechbrain_embeddings()
    df = compute_speechbrain_similarities()
    print(df)