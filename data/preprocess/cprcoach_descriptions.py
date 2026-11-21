DESCRIPTION_VERSION = 'cprcoach_editorial_visual_descriptions_v1'
DESCRIPTION_BASIS = 'label names and inspected training examples; not official definitions'
ERROR_DESCRIPTIONS = {
    1: 'The hands are laid flat across one another with the fingers extended and '
       'not interlocked during compression. Look at the finger arrangement; '
       'having one hand above the other alone does not identify this error.',
    2: 'The hands are clenched into fists, with the knuckle side pressing against '
       'the chest instead of an open hand heel supporting the compression.',
    3: 'Only one hand presses on the chest; the other hand does not support the '
       'pressing hand during the compressions.',
    4: 'The elbows remain visibly bent or repeatedly flex during the downward '
       'press, so the arms do not maintain a straight supporting position.',
    5: 'The arms approach the chest at a slant, with the shoulders displaced '
       'from directly above the hands, producing an oblique pressing direction.',
    6: 'The hands visibly lift away from the chest between presses and then '
       'come down again, producing a bouncing or striking compression motion.',
    7: 'The rescuer performs compressions from a low squat with the feet '
       'supporting the body and the knees raised off the floor.',
    8: 'The rescuer stands on the feet with the knees off the floor and bends '
       'over the mannequin lying on the floor to perform compressions.',
    9: 'The hands press at an off-target location on the torso, displaced from '
       'the intended central chest compression area. Assess the contact location.',
    10: 'The presses show a small, shallow downward movement, with limited '
        'visible chest depression across repeated compressions. Assess the '
        'compression excursion over time, not just the hand position in one frame.',
    11: 'Successive compressions occur at a slow pace, with prolonged intervals '
        'between downward presses. Assess the repetition timing over the video.',
    12: 'The presses show an exaggerated, deep downward excursion into the '
        'chest, accompanied by pronounced downward movement of the hands and '
        'upper body. Assess compression depth over the movement cycle.',
    13: 'The hand contact location wanders between different points on the '
        'chest across successive presses, rather than remaining at a stable '
        'compression location.',
}

PROVISIONAL_DESCRIPTION_IDS = frozenset({10, 12})
