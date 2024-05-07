from typing import Tuple, List

import numpy as np

FPS = 20


def analyze_caption(caption: str) -> Tuple[str, Tuple, List[str]]:
    """
    :param caption: a caption in hm3d format.
                    exp: person is lifting up dumbbells.#person/NOUN person/NOUN is/AUX lift/VERB up/ADP dumbbell/NOUN#0.0#0.0
    :return: description str, (start_timestamp, end_timestamp), word-part_of_speech list(['person/NOUN',...])
    """
    raw_caption, word_pos, f_tag, to_tag = caption.split('#')

    f_tag = float(f_tag)
    to_tag = float(to_tag)

    f_tag = 0 if np.isnan(f_tag) else int(f_tag * FPS)
    to_tag = 0 if np.isnan(to_tag) else int(to_tag * FPS)

    # when f_tag and to_tag is not 0, it means the description responds only to motion[f_tag:to_tag]

    word_pos = word_pos.split(' ')

    return raw_caption, (f_tag, to_tag), word_pos