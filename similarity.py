from collections import defaultdict
from pathlib import Path
import re

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

def load_atds_sims():
    dfs = []
    for path in Path('ATDS/').glob('atds_*.csv'):
        df = pd.read_csv(path)
        dfs.append(df[['ref_lang', 'comp_lang', 'atds']])
    df = pd.concat(dfs).dropna()
    return df

def load_wandb_wers():
    #df = pd.read_csv('ATDS/wandb_export_2023-12-08T09 20 38.557+11 00.csv')
    #df = pd.read_csv('ATDS/wandb_export_2023-12-08T10 33 37.801+11 00.csv')
    df = pd.read_csv('20231210_asr-results.csv')
    columns = [col for col in df.columns if re.match(r'.*xls-r_cpt.*test/wer', col)]
    df = df[columns]
    df = df.T
    df2 = pd.read_csv('ATDS/wandb_export_test2h_raw_wer.csv')
    columns = [col for col in df2.columns if re.match(r'.*xls-r_cpt.*test-2h/raw_wer', col)]
    df2 = df2[columns]
    df2 = df2.T
    df = pd.concat([df, df2])
    def get_ref_lang(run):
        match = re.match(r'.*xls-r_cpt_([a-z]*)-.*_([a-z]*)-.*wer', run)
        if match is None:
            return None
        ref_lang = match.group(1)
        return ref_lang
    def get_comp_lang(run):
        match = re.match(r'.*xls-r_cpt_([a-z]*)-.*_([a-z]*)-.*wer', run)
        if match is None:
            return None
        comp_lang = match.group(2)
        return comp_lang
    df['ref_lang'] = df.index.map(get_ref_lang)
    df['comp_lang'] = df.index.map(get_comp_lang)
    df['wer'] = df[3]
    df = df[['ref_lang', 'comp_lang', 'wer']]
    df = df.drop_duplicates()
    return df

def load_wers():
    df = pd.read_csv('20231210_asr-results.csv')
    df = df.dropna()
    def get_comp_lang(cpt_data):
        match = re.match(r'.*_(.*?)-.*', cpt_data)
        if match is None:
            return None
        ref_lang = match.group(1)
        return ref_lang
    df['comp_lang'] = df.CPT_data.map(get_comp_lang)
    df = df.dropna()
    df = df[df['comp_lang'] != 'aug']
    df = df[df['target_lang'] == 'Punjabi']
    df['ref_lang'] = df['target_lang']
    df['ref_lang'] = df['ref_lang'].str.lower()
    df['wer'] = df['test_wer']
    return df

#wandb = load_wandb_wers()
#wer = load_wers()

language_codes = {
    'tamil': 'tam',
    'malayalam': 'mal',
    'urdu': 'urd',
    'sepedi': 'nso',
    'gujarati': 'guj',
    'malay': 'msa',
    'odia': 'ori',
    'sesotho': 'sot',
    'marathi': 'mar',
    'iban': 'iba',
    'setswana': 'tsn',
    'bengali': 'ben',
    'punjabi': 'pan',
    'indonesian': 'ind',
    'hindi': 'hin',
}

def lang2vec_learned_distance(lang1, lang2):
    if lang1 not in l2v.LEARNED_LANGUAGES or lang2 not in l2v.LEARNED_LANGUAGES:
        return None
    x = torch.tensor(l2v.get_features(lang1, 'learned')[lang1])
    y = torch.tensor(l2v.get_features(lang2, 'learned')[lang2])
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=0, eps=1e-8).item()

def lang2vec_mean_distance(lang1, lang2):
    lang2vec_funcs = [
        l2v.syntactic_distance,
        l2v.geographic_distance,
        l2v.phonological_distance,
        l2v.genetic_distance,
        l2v.inventory_distance,
        l2v.featural_distance,
    ]
    dists = []
    for l2v_func in lang2vec_funcs:
        dists.append(l2v_func(language_codes[lang1], language_codes[lang2]))
    return sum(dists) / len(dists)

def add_lang2vec_sims(df):
    import lang2vec.lang2vec as l2v
    lang2vec_funcs = [
        l2v.syntactic_distance,
        l2v.geographic_distance,
        l2v.phonological_distance,
        l2v.genetic_distance,
        l2v.inventory_distance,
        l2v.featural_distance,
    ]
    for lang2vec_func in lang2vec_funcs:
        df[lang2vec_func.__name__] = df.apply(lambda row: 1 - lang2vec_func(language_codes[row.ref_lang], language_codes[row.comp_lang]), axis=1)
    #df['lang2vec_learned_distance'] = df.apply(lambda row: lang2vec_learned_distance(language_codes[row.ref_lang], language_codes[row.comp_lang]), axis=1)
    df['lang2vec_mean_distance'] = df[[lang2vec_func.__name__ for lang2vec_func in lang2vec_funcs]].mean(axis=1)
    return df

def create_wer_and_sim_df():
    wers = load_wers()
    atds = load_atds_sims()
    speechbrain = compute_speechbrain_similarities()
    df = atds.join(speechbrain.set_index(['ref_lang', 'comp_lang']), on=['ref_lang', 'comp_lang'])
    df = df.join(wers.set_index(['ref_lang', 'comp_lang']), on=['ref_lang', 'comp_lang'])
    df = df.dropna()
    df = add_lang2vec_sims(df)
    return df


def create_sim_df():
    atds = load_atds_sims()
    speechbrain = compute_speechbrain_similarities()
    df = atds.join(speechbrain.set_index(['ref_lang', 'comp_lang']), on=['ref_lang', 'comp_lang'])
    df = df.dropna()
    df = add_lang2vec_sims(df)
    return df

if __name__ == '__main__':
    df = create_wer_and_sim_df()