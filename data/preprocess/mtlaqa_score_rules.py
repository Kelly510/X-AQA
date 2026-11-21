FINA_2009 = 'https://ffn.extranat.fr/html/ressources/jug/fina_pl_2009_2013.pdf'
FINA_2015 = 'https://www.diving-gbdf.com/downloads/fina-diving-rules.pdf'
FINA_2013_CHANGE = 'https://www.worldaquatics.com/news/1913937/pr-52-fina-technical-congresses-diving-open-water-water-polo-and-synchro'


EVENTS = {
    1: (2012, 'London Olympics', '10m platform', 'men'),
    2: (2012, 'London Olympics', '10m platform', 'men'),
    3: (2012, 'London Olympics', '10m platform', 'men'),
    4: (2012, 'London Olympics', '10m platform', 'women'),
    5: (2012, 'London Olympics', '10m platform', 'women'),
    6: (2012, 'London Olympics', '10m platform', 'women'),
    7: (2014, 'Nanjing Youth Olympics', '10m platform', 'men'),
    9: (2014, 'Nanjing Youth Olympics', '10m platform', 'women'),
    10: (2014, 'Nanjing Youth Olympics', '10m platform', 'women'),
    13: (2014, 'FINA Diving World Series, Beijing', '10m platform', 'women'),

    14: (2016, 'London European Championships', '10m platform', 'women'),
    17: (2012, 'London Olympics', '3m springboard', 'men'),
    18: (2012, 'London Olympics', '3m springboard', 'men'),
    22: (2014, 'Nanjing Youth Olympics', '3m springboard', 'women'),
}

BANDS = ((10.0, 'excellent'), (8.5, 'very good'), (7.0, 'good'),
         (5.0, 'satisfactory'), (2.5, 'deficient'), (0.5, 'unsatisfactory'),
         (0.0, 'completely failed'))
SCALE = """Use this judging scale:
- 10.0: excellent.
- 8.5-9.5: very good.
- 7.0-8.0: good.
- 5.0-6.5: satisfactory.
- 2.5-4.5: deficient.
- 0.5-2.0: unsatisfactory.
- 0.0: completely failed.
"""

CRITERIA = """
Execution criteria and deductions:
1. Starting position, approach and take-off
- For a standing start, use a natural, straight posture with head erect and arms straight in any position. For an armstand, establish a steady, straight vertical balance. An incorrect starting position, unstable armstand or crow-hop on a standing dive warrants 0.5-2.0.
- A running approach must be smooth and continuous forwards to the end of the apparatus; approach faults warrant 0.5-2.0.
{takeoff}
- Insufficient take-off height or confidence, failure to leave from the end of the apparatus, or twisting visibly initiated on the apparatus warrants 0.5-2.0.

2. Flight and announced body position
- Assess rotation control and the announced shape. Straight: extend hips and knees. Pike: bend at the hips with knees straight. Arms may be placed freely in straight and pike positions. Tuck: use a compact shape with hips and knees bent, knees together and hands on the lower legs. Free position permits combinations of shapes; a permitted transition is not an error.
- Keep feet together and toes pointed through flight and entry. Flight-position defects warrant 0.5-2.0. In an announced pike or tuck dive with twist, failure to clearly display that position also warrants 0.5-2.0.
- Distinguish a defective shape from performing a different position: a clearly different position from the announced one limits the award to 2.0; performing only part of the dive in a different position limits it to 4.5. Do not apply these limits to permitted free-position changes or ordinary form defects.
{clearance}

3. Opening and entry
- Assess opening timing, completion of rotation, alignment and control. Enter vertically without residual twist, with the body straight. Judge short or over-rotated entries and remaining twist by visible severity; do not invent an angle-to-deduction formula or infer the cause from entry angle alone.
- For head-first entry, extend the arms beyond the head in line with the body, hands close together. One or both arms remaining below the head limits the award to 4.5. For feet-first entry, keep arms close to the body and elbows straight; one or both arms beyond the head limits the award to 4.5. Other arm-position faults warrant 0.5-2.0.
- Consider splash with alignment and control. A clean entry is a positive execution feature, but a small splash cannot compensate for incomplete rotation or poor alignment; do not assign a fixed splash deduction or an automatic bonus.

Overall assessment:
- Judge execution independently of difficulty. The announced dive describes the intended movement; its difficulty does not earn execution credit.
- Form one overall-impression award from the quality scale and visible execution. Within a deduction range, use the lower end for a minor fault and the higher end for a severe fault. Do not give phases equal weights.
- Count each distinct fault once within the award: do not downgrade the overall impression for it and then subtract it again, or count it repeatedly across frames or overlapping descriptions. Independent errors in different phases remain assessable.
- Apply the most restrictive applicable maximum to the resulting award. A maximum is a ceiling, not another deduction. Do not invent unlisted caps or unseen referee decisions.
- Evaluate from the starting position until the body is fully submerged; disregard the walk to the starting position and underwater movements. Judge only visible evidence. Cropped, obscured or unsampled phases do not establish faults, and misleading perspective does not establish exact angles or unsafe clearance.
"""


def event_profile(key: tuple[int, int]) -> dict:
    source, dive = key
    if source == 26:
        year, event = 2014, 'Glasgow Commonwealth Games'
        apparatus, gender = ('3m springboard', 'women') if dive <= 60 else ('10m platform', 'men')
    else:
        try:
            year, event, apparatus, gender = EVENTS[source]
        except KeyError:
            raise ValueError(f'Unverified MTL-AQA source video: {source}; audit it before assigning event rules') from None
    return dict(event_year=year, event_name=event, apparatus=apparatus, gender=gender,
                event_attribution='source_video_titles_and_manual_frames', source_video_id=source,
                rule_edition='FINA 2009-2013' if year==2012 else 'FINA 2013-2017',
                rule_sources=[FINA_2009] if year==2012 else [FINA_2013_CHANGE, FINA_2015],
                rubric_scope='selected_visible_execution_rules', rubric_version='mtlaqa_execution_event_v1')


def rubric(meta: dict) -> str:
    springboard = meta['apparatus'] == '3m springboard'
    takeoff = ('- On the springboard, take off from both feet simultaneously.' if springboard else
               '- On the platform, forward and reverse take-offs may use one foot; assess an armstand start according to its balance and control.')
    clearance = ('- Reduce the award for sideways deviation, unsafe proximity or contact with the apparatus according to severity and its effect on execution.'
                 if meta['event_year'] == 2012 else
                 '- Sideways deviation or touching the end of the apparatus with feet or hands lowers the award according to severity. Clearly unsafe proximity to the apparatus or touching its end with the head limits the award to 2.0.')
    return SCALE + CRITERIA.format(takeoff=takeoff, clearance=clearance)


def score_prompt(key: tuple[int, int], target: str = 'award') -> tuple[str, dict]:
    meta = event_profile(key)
    criteria = rubric(meta)
    if target == 'award':
        intro = f"You are one execution judge at a {meta['apparatus']} diving competition.\n\nGive one overall execution award from 0.0 to 10.0 in increments of 0.5.\n\n"
        ending = (
            '\nOutput only this JSON object and nothing else: '
            '{"execution_award": <number from 0.0 to 10.0 in increments of 0.5>}.'
        )
    elif target == 'panel':
        intro = f"You are a panel of seven execution judges at a {meta['apparatus']} diving competition.\n\nGive seven execution awards, each from 0.0 to 10.0 in increments of 0.5. Apply the following criteria to each award.\n\n"
        ending = ('\nPanel total: sort the seven awards, discard the two highest and two lowest, and multiply the sum of the remaining three by the announced difficulty.\n'
                  'Output only JSON: {"awards":[<seven awards, ascending>],"final_score":<total>}.')
    else:
        raise ValueError(f'Unknown score target: {target}')
    return intro + criteria + ending, meta


def comment_prompt(key: tuple[int, int]) -> tuple[str, dict]:
    meta = event_profile(key)
    intro = (f"You are a diving commentator at a {meta['apparatus']} diving competition.\n\n"
             'Give a brief English commentary on the execution of the dive in the video in one '
             'or a few sentences, speaking naturally as you would during a sports broadcast and '
             'letting your tone and reactions reflect the performance you see.\n\n'
             'The execution judges apply the scale and criteria below. Read the dive against '
             'them to decide which faults and merits are worth remarking on and how strongly, '
             "then say what you see in a commentator's words rather than in numbers.\n\n")
    ending = '\nOutput only the commentary. Do not state an award or a deduction value.'
    return intro + rubric(meta) + ending, meta
