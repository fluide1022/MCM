import random

SCENE_PROMPTS = {
    'descript': [
        'A movement that matches the description of the action',
        'A motion that matches the description of the action',
        'A movement that corresponds to the description of the action',
        'A motion that corresponds to the description of the action',
        'A movement that fits the description of the action',
        'A motion that fits the description of the action',
        'A movement that follows the description of the action',
        'A motion that follows the description of the action',
        'A movement that reflects the description of the action',
        'A motion that reflects the description of the action',
        'A movement that conforms to the description of the action',
        'A motion that conforms to the description of the action',
        'A movement that aligns with the description of the action',
        'A motion that aligns with the description of the action',
        'A movement that adheres to the description of the action',
        'A motion that adheres to the description of the action',
    ],
    'dance': [
        "dancing in a room."
        "A person is dancing",
        "Someone is dancing",
        "A human is dancing",
        "An individual is dancing",
        "A dancer is moving",
        "A mover is dancing",
        "A person is performing dance",
        "Someone is performing dance",
        "A human is performing dance",
        "An individual is performing dance",
        "A dancer is performing movement",
        "A mover is performing movement",
        "A person is expressing dance",
        "Someone is expressing dance",
        "A human is expressing dance",
        "An individual is expressing dance",
        "A dancer is expressing movement",
        "A mover is expressing movement",
        "A person is in motion with dance",
        "Someone is in motion with dance",
        "A person is dancing to the music", "One is following the rhythm and dancing",
        "Someone is moving with the melody", "A human is expressing the song",
        "Someone's dancing to the tune on the stage.",
        "A dancer is performing to the music",
        "A professional dancer is performing a dance along with the music.",
        "Along with the music, a professional dancer is giving a dance performance.",
        "A dance performance is being given by a professional dancer along with the music.",
        "With the music, a professional dancer is showcasing a dance.",
        "A professional dancer is demonstrating a dance with the music.",
        "A dance is being demonstrated by a professional dancer with the music.",
        "With the music, a dance demonstration is being done by a professional dancer.",
        "A professional dancer is presenting a dance in sync with the music.",
        "In sync with the music, a professional dancer is presenting a dance.",
        "A dance presentation is being done by a professional dancer in sync with the music.",
        "A professional dancer is executing a dance to the music.",
        "To the music, a professional dancer is executing a dance.",
        "A dance execution is being done by a professional dancer to the music.",
        "A professional dancer is delivering a dance in harmony with the music.",
        "In harmony with the music, a professional dancer is delivering a dance.",
        "A dance delivery is being done by a professional dancer in harmony with the music.",
        "A professional dancer is displaying a dance matching the music.",
        "Matching the music, a professional dancer is displaying a dance.",
        "A dance display is being done by a professional dancer matching the music.",
        "A professional dancer is exhibiting a dance following the music.",
        "A dancer is performing to the background music.",
        "To the background music, a dancer is giving a performance.",
        "A performance is being given by a dancer to the background music.",
        "With the background music, a dancer is showcasing their skills.",
        "A dancer is demonstrating their skills with the background music.",
        "A skill demonstration is being done by a dancer with the background music.",
        "With the background music, a skill showcase is being done by a dancer.",
        "A dancer is presenting their art in sync with the background music.",
        "In sync with the background music, a dancer is presenting their art.",
        "An art presentation is being done by a dancer in sync with the background music.",
        "A dancer is executing their moves to the background music.",
        "To the background music, a dancer is executing their moves.",
        "A move execution is being done by a dancer to the background music.",
        "A dancer is delivering their expression in harmony with the background music.",
        "In harmony with the background music, a dancer is delivering their expression.",
        "An expression delivery is being done by a dancer in harmony with the background music.",
        "A dancer is displaying their talent matching the background music.",
        "Matching the background music, a dancer is displaying their talent.",
        "A talent display is being done by a dancer matching the background music.",
        "A dancer is exhibiting their style following the background music.",
        "A graceful dancer is performing to the soothing background music in a studio.",
        "To the upbeat background music, an energetic dancer is giving an impressive performance on a stage.",
        "A captivating performance is being given by a skilled dancer to the lively background music in front of an "
        "audience.",
        "With the soft background music, a delicate dancer is showcasing their elegant skills in a garden.",
        "A passionate dancer is demonstrating their powerful skills with the intense background music in a gym.",
        "A stunning skill demonstration is being done by an agile dancer with the fast-paced background music in a "
        "park.",
        "With the romantic background music, a charming skill showcase is being done by a lovely dancer in a ballroom.",
        "A creative dancer is presenting their unique art in sync with the experimental background music in an art "
        "gallery.",
        "In sync with the classical background music, an artistic dancer is presenting their refined art in a museum.",
        "An amazing art presentation is being done by an expressive dancer in sync with the dramatic background music "
        "in a theater.",
        "A dynamic dancer is executing their smooth moves to the funky background music in a club.",
        "To the catchy background music, a cheerful dancer is executing their fun moves on a beach.",
        "A spectacular move execution is being done by an adventurous dancer to the thrilling background music on a "
        "mountain top.",
        "A soulful dancer is delivering their heartfelt expression in harmony with the sentimental background music "
        "in a church.",
        "In harmony with the relaxing background music, a calm dancer is delivering their serene expression in a spa.",
        "An inspiring expression delivery is being done by an optimistic dancer in harmony with the motivational "
        "background music in a school.",
        "A talented dancer is displaying their amazing talent matching the pop background music on TV.",
        "Matching the rock background music, an awesome dancer is displaying their cool talent on YouTube.",
        "A brilliant talent display is being done by an innovative dancer matching the techno background music on "
        "TikTok.",
        "A stylish dancer is exhibiting their trendy style following the hip-hop background music in a mall."
    ],
    'accompany': [
        "A person is speaking",
        "Someone is speaking",
        "A human is speaking",
        "An individual is speaking",
        "A speaker is talking",
        "A talker is speaking",
        "A person is communicating verbally",
        "Someone is communicating verbally",
        "A human is communicating verbally",
        "An individual is communicating verbally",
        "A speaker is communicating orally",
        "A talker is communicating orally",
        "A person is expressing speech",
        "Someone is expressing speech",
        "A human is expressing speech",
        "An individual is expressing speech",
        "A speaker is expressing words",
        "A talker is expressing words",
        "A person is using language vocally",
        "Someone is using language vocally"
    ],
    'interpolate': [
        'A movement with missing frames'
    ],
    'kungfu': [
        'A martial arts movement'
    ]
}

CLS_PROMPT = [
    'a person is doing {}',
    'a motion called {} is performed',
    'he is doing {}',
    'she is doing {}',
    'the action performed is called {}'
]


def prompt_scene(scene: str):
    """
    Args:
        scene: scene like description, dance to music...

    Returns:
        prompted scene like: a motion matches description
    """
    prompts = SCENE_PROMPTS.get(scene)
    return random.choice(prompts)


def prompt_cls(cls: str):
    if cls is None:
        return None
    prompts = CLS_PROMPT
    prompt: str = random.choice(prompts)
    prompt = prompt.format(cls)
    return prompt
