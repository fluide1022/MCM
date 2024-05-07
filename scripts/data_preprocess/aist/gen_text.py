import argparse
import os
import random

from tqdm import tqdm

PROMPTS = [
    'A {} dancer is performing {} {}'
]

subjects = [
    'a {} dancer',
    'a professional {} dancer',
    'an experienced {} dancer',
    'a {} dance artist',
    'a trained dance performer'
]
verbs = {
    'basic dance': [
        'dances {} style basic dance',
        'performs {} basic dance',
        'demonstrates a basic dance in {} style',
        'is performing {} style basic dance',
        'is dancing {}',
        'is performing {} style',
        'demonstrates {} style'
    ],
    'advanced dance': [
        'dances {} style advanced dance',
        'performs {} style challenging dance',
        'demonstrates a highly technical dance in {} style',
        'is performing {} style advanced dance',
        'is dancing {} with high technique',
        'is performing {} style',
        'demonstrates {} style',
    ],
    'moving camera': [
        'demonstrates {} style dance'
        'dances {} style',
        'performs {}',
        'is dancing {} with moving camera around'
        'is performing {} style',
        'demonstrates {} style'
    ],
    'group dance': [
        'participates in a group dance of {} style',
        'joins in a group dance of {}',
        'takes part in a group dance featuring {} style',
        'dances in a {} style group dance',
        'is performing {}',
        'demonstrates {} style',
        'dances {}'
    ],
    'showcase': [
        'showcases a {} style dance',
        'puts on a dance show featuring {}',
        'is having a {} style showcase',
        'is performing a showcase of {} performance',
        'dances {}',
        'performs in a {} style showcase',
    ],
    'cypher': [
        'gives a {} style dance in the Cypher',
        'dances {} in the Cypher',
        'showcases a dance in {} style during the Cypher',
        'presents a dance of {} style in the Cypher',
        'dances {}',
        'is performing {} style',
        'demonstrates {} style'
    ],
    'battle': [
        'performs {} in the dance battle',
        'dances {} as part of the dance battle',
        'is dancing {}',
        'is performing {} style',
        'presents a dance in {} style during the dance battle',
        'showcases a {} performance during the battle',
        'executed a {} routine in the dance battle',
        'is giving a {} style performance'
    ]

}

advs = [
    'following the music',
    'to the music',
    'along with the beat',
    '',
    'in time with the music',
    'in rhythm with the music',
    'spontaneously to the music rhythm'
]


def make_sentence(subject: str, verb: str, adv: str):
    words = subject.split(' ') + verb.split(' ') + adv.split(' ')
    return ' '.join(words) + '.'


def make_sentences(gender: str, style: str, dance_form: str, num_sentence):
    sub = random.choices(subjects, k=num_sentence)
    verb = random.choices(verbs[dance_form], k=num_sentence)
    adv = random.choices(advs, k=num_sentence)
    random.shuffle(sub), random.shuffle(verb), random.shuffle(adv)
    sentences = []
    for s, v, a in zip(sub, verb, adv):
        sentences.append(make_sentence(s.format(gender), v.format(style), a))
    return sentences


def get_gender(dancer_id):
    dance_id = int(dancer_id.strip('d'))
    # female
    if dance_id in [1, 2, 3, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19, 25, 26, 27]:
        return 'female'
    return 'male'


def get_genre(genre_id: str):
    genre_dict = {
        'BR': 'Break',
        'PO': 'Pop',
        'LO': 'Lock',
        'MH': 'Middle Hip-hop',
        'LH': 'LA style Hip-hop',
        'HO': 'House',
        'WA': 'Waack',
        'KR': 'Krump',
        'JS': 'Street Jazz',
        'JB': 'Ballet Jazz'
    }
    return genre_dict[genre_id[-2:]]


def get_dance_form(situation: str):
    dance_form_dict = {
        'BM': 'basic dance',
        'FM': 'advanced dance',
        'MM': 'moving camera',
        'GR': 'group dance',
        'SH': 'showcase',
        'CY': 'cypher',
        'BT': 'battle'
    }
    return dance_form_dict[situation[1:3]]


def get_dance_info(name: str):
    genre, situation, _, dancer_id = name.split('_')[:4]
    gender = get_gender(dancer_id)
    genre = get_genre(genre)
    dance_form = get_dance_form(situation)
    return gender, genre, dance_form


def get_token(text):
    tokens = []
    for word in text.split(' '):
        tokens.append(word + '/UNK')
    return ' '.join(tokens)


def make_file(save_path, texts):
    rows = ''
    for text in texts:
        token = get_token(text)
        rows += text + '#' + token + '#0.0#0.0\n'

    with open(save_path, 'w') as fp:
        fp.write(rows)


if __name__ == '__main__':
    args = argparse.ArgumentParser('generate text file for aist++')
    args.add_argument('--vec_root', default='data/aist_plusplus_final/vecs_joints_22')
    args.add_argument('--text_root', default='data/aist_plusplus_final/texts')
    args.add_argument('--num_sentence', default=3)
    args = args.parse_args()
    os.makedirs(args.text_root, exist_ok=True)
    for file in os.listdir(args.vec_root):
        name = file.split('.')[0]
        gender, style, dance_form = get_dance_info(name)
        texts = make_sentences(gender, style, dance_form, args.num_sentence)
        make_file(os.path.join(args.text_root, name+'.txt'), texts)
