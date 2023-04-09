#!/usr/bin/env python3
"""
NLP analysis with shifterator


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import json
import pdb
import pandas as pd
import re
import sys
import shifterator as shift
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import argparse

from collections import Counter

def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: str(elem).replace(',', ' '))
    # remove accents
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([0-9])|([^\w\s])|(\w+:\/\/\S+)|^rt|http.+?", "", str(elem)))  
    df[text_field] = df[text_field].apply(lambda elem: str(elem).strip())
    df[text_field] = df[text_field].apply(lambda elem: str(elem).replace('   ', ' '))
    df[text_field] = df[text_field].apply(lambda elem: str(elem).replace('  ', ' '))
    df[text_field] = df[text_field].apply(lambda elem: str(elem).replace('\n', ' '))

    # df[text_field] = df[text_field].apply(lambda elem: str(elem).replace('_', ' '))
    return df

def load_json(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = json.loads(data)
    return data

def get_stopwords():
    return {'emotion', 'me', 'makes', 'that', 'better', 'and'}

def free_words_shift(anno, sh):
    stp_wrds = get_stopwords()
    # free mood compare valence
    anno_val_pos = anno[anno.valenceValue == '1']
    anno_val_neg = anno[anno.valenceValue == '-1']
    txt_pos = ' '.join([_ for _ in anno_val_pos.freeMood.tolist() if _ != 'None'])
    print(f"Number of positive valence words: {len(txt_pos.split(' '))}")
    c_v = CountVectorizer(stop_words=stp_wrds)
    mat_pos = c_v.fit_transform([txt_pos])
    try:
        pos_dict = dict(
            zip(c_v.get_feature_names_out(), mat_pos.toarray()[0].tolist())
        )
    except:
        pos_dict = dict(zip(c_v.get_feature_names(), mat_pos.toarray()[0].tolist()))

    txt_neg = ' '.join([_ for _ in anno_val_neg.freeMood.tolist() if _ != 'None'])
    print(f"Number of negative valence words: {len(txt_pos.split(' '))}")
    c_v = CountVectorizer(stop_words=stp_wrds)
    mat_neg = c_v.fit_transform([txt_neg])
    try:
        neg_dict = dict(
            zip(c_v.get_feature_names_out(), mat_neg.toarray()[0].tolist())
        )
    except:
        neg_dict = dict(zip(c_v.get_feature_names(), mat_neg.toarray()[0].tolist()))


    fig, axi = plt.subplots(ncols=1, nrows=1, figsize=(10.5, 9), tight_layout=True, dpi=100)
    if sh == 'prop':
        ps = shift.ProportionShift(type2freq_1=pos_dict,
                                 type2freq_2=neg_dict)
        ps.get_shift_graph(
            system_names=['Pos. Valence', 'Neg. Valence'],
            title='Proportion shift of valence',
            top_n=35,
            cumulative_inset=False,
            show_plot=False,
            ax=axi,
            filename=f'./figs_paper/freetext_{sh}.svg',
        )
    elif sh == 'shan':
        en = shift.EntropyShift(type2freq_1=pos_dict,
                                type2freq_2=neg_dict)
        en.get_shift_graph(
            system_names=['Pos. Valence', 'Neg. Valence'],
            title='Entropy shift of valence',
            top_n=35,
            cumulative_inset=False,
            show_plot=False,
            ax=axi,
            filename=f'./figs_paper/freetext_{sh}.svg',
        )
    elif sh == 'jsd':
        jsd_shift = shift.JSDivergenceShift(type2freq_1=pos_dict,
                                         type2freq_2=neg_dict,
                                         weight_1=0.5,
                                         weight_2=0.5,
                                         base=2,
                                         alpha=1)
        jsd_shift.get_shift_graph(
            system_names=['Pos. Valence', 'Neg. Valence'],
            title='JSD shift of valence',
            top_n=35,
            cumulative_inset=False,
            show_plot=False,
            ax=axi,
            filename=f'./figs_paper/freetext_{sh}.svg',
        )


def reasoning_shift(anno, sh):
    stp_wrds = get_stopwords()
    # comparison reasoning
    txt_core_aro =  ' '.join([_ for _ in anno.arousalComment.tolist() if _ != 'None'])
    txt_core_val =  ' '.join([_ for _ in anno.valenceComment.tolist() if _ != 'None'])
    txt_core = f'{txt_core_aro} {txt_core_val}'
    print(f"Number of arousal/valence words: {len(txt_core.split(' '))}")
    c_v = CountVectorizer(stop_words=stp_wrds)
    mat_core = c_v.fit_transform([txt_core])
    try:
        core_dict = dict(
            zip(c_v.get_feature_names_out(), mat_core.toarray()[0].tolist())
        )
    except:
        core_dict = dict(zip(c_v.get_feature_names(), mat_core.toarray()[0].tolist()))

    txt_emo = ' '.join([_ for _ in anno.moodComment.tolist() if _ != 'None'])
    # txt_free = ' '.join([_ for _ in anno.freeMood.tolist() if _ != 'None'])
    # txt_emo = txt_emo + ' ' + txt_free
    print(f"Number of emotion words: {len(txt_emo.split(' '))}")
    c_v = CountVectorizer(stop_words=stp_wrds)
    mat_mood = c_v.fit_transform([txt_emo])
    try:
        mood_dict = dict(
            zip(c_v.get_feature_names_out(), mat_mood.toarray()[0].tolist())
        )
    except:
        mood_dict = dict(zip(c_v.get_feature_names(), mat_mood.toarray()[0].tolist()))

    fig, axi = plt.subplots(ncols=1, nrows=1, figsize=(10.5, 9), tight_layout=True, dpi=100)
    if sh == 'prop':
        
        ps = shift.ProportionShift(type2freq_1=core_dict,
                                type2freq_2=mood_dict)
        ps.get_shift_graph(
            system_names=['Core Affects', 'Emotion words'],
            title='Proportion shift of Reasoning',
            top_n=35,
            cumulative_inset=False,
            show_plot=False,
            ax=axi,
            filename=f'./figs_paper/core_vs_words_{sh}.svg',
        )
    elif sh == 'shan':
        en = shift.EntropyShift(type2freq_1=core_dict,
                                type2freq_2=mood_dict)
        en.get_shift_graph(
            system_names=['Core Affects', 'Emotion words'],
            title='Entropy shift of Reasoning',
            top_n=35,
            cumulative_inset=False,
            show_plot=False,
            ax=axi,
            filename=f'./figs_paper/core_vs_words_{sh}.svg',
        )
    elif sh == 'jsd':
        jsd_shift = shift.JSDivergenceShift(type2freq_1=core_dict,
                                         type2freq_2=mood_dict,
                                         weight_1=0.5,
                                         weight_2=0.5,
                                         base=2,
                                         alpha=1)
        jsd_shift.get_shift_graph(
            system_names=['Core Affects', 'Emotion words'],
            title='JSD shift of Reasoning',
            top_n=35,
            cumulative_inset=False,
            show_plot=False,
            ax=axi,
            filename=f'./figs_paper/core_vs_words_{sh}.svg',
        )
  



if __name__ == "__main__":
    # usage: python3 nlp_shifterator.py -shift [prop/shan/jsd]
    parser = argparse.ArgumentParser()
    parser.add_argument('-shift',
                        '--shift',
                        help='Select word shift method: proportion [prop], Shannon-Entropy [shan], Jensen-Shannon Divergence [jsd]',
                        action='store',
                        required=True,
                        dest='shifter')
    args = parser.parse_args()

    if args.shifter not in ['prop', 'shan', 'jsd']:
        print('Select valid wordshift method!')
        sys.exit(0)

    # fn = './data/data_24_11_2021.json'
    fn = './data/data_07_03_2022.json'
    tags = ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace', 'tenderness', 'transcendence']
    data = load_json(fn)

    anno = pd.DataFrame(data['annotations'])
    users = pd.DataFrame(data['users'])

    anno = clean_text(anno, 'freeMood')
    anno = clean_text(anno, 'arousalComment')
    anno = clean_text(anno, 'valenceComment')
    anno = clean_text(anno, 'moodComment')

    reasoning_shift(anno, args.shifter)

    free_words_shift(anno, args.shifter)
    # pdb.set_trace()

