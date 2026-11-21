from functools import lru_cache
import hashlib
import json
from pathlib import Path

RULE_VERSION = 'rg_event_execution_penalty_v5'
RULE_URL = 'https://agf.az/ups/files/rules/en_rg_cop_2017_2020_with_errata_dec._17.pdf'
OLD_RULE_URL = 'https://matsz.hu/news/824/rg_cop_2013-2016_valid_1st_january_2015_e_with_marks.pdf'
NEWS_2018 = 'https://www.fig-docs.com/website/newsletters/RG/RG_NL_1_March_updated_June_2018-e.pdf'
NEWS_2019 = 'https://www.fig-docs.com/website/newsletters/RG/GR_Newsletter%201_March%20updated%20June%202019_e.pdf'
PROFILES = {
    2015: dict(event='34th Rhythmic Gymnastics World Championships', event_location='Stuttgart',
               rule_profile='fig_rg_2015', rule_edition='2013-2016, valid 2015-01-01',
               rule_sources=[OLD_RULE_URL], rubric_pages=[3, 5, 6, 7, 8, 23, 24, 25, 26, 27]),
    2018: dict(event='36th Rhythmic Gymnastics World Championships', event_location='Sofia',
               rule_profile='fig_rg_2018', rule_edition='2017-2020, valid 2018-02-01; newsletter June 2018',
               rule_sources=[RULE_URL, NEWS_2018], rubric_pages=[4, 8, 9, 10, 11, 12, 18, 19, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]),
    2019: dict(event='37th Rhythmic Gymnastics World Championships', event_location='Baku',
               rule_profile='fig_rg_2019', rule_edition='2017-2020, valid 2018-02-01; newsletter June 2019',
               rule_sources=[RULE_URL, NEWS_2019], rubric_pages=[4, 8, 9, 10, 11, 12, 18, 19, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]),
}
APPARATUS = {'Ball': {'name_zh': '球'}, 'Clubs': {'name_zh': '棒'},
             'Hoop': {'name_zh': '圈'}, 'Ribbon': {'name_zh': '带'}}


@lru_cache(maxsize=1)
def event_manifest():
    path = Path(__file__).resolve().parents[1] / 'annotations' / 'rhythmicgymnastics_events.json'
    return json.loads(path.read_text())['clips']


def event_metadata(clip: str, raw_annotation: str) -> dict:
    evidence = event_manifest()[clip]
    if hashlib.sha256(raw_annotation.encode()).hexdigest() != evidence['annotation_sha256']:
        raise ValueError(f'RG source annotation changed; re-audit event mapping: {clip}')
    year = evidence['event_year']
    return dict(PROFILES[year], event_year=year,
                event_assignment='audited_inference_from_venue_frames_and_official_results',
                event_audit='docs/rhythmicgymnastics/EVENT_AUDIT.md',
                rubric_scope='selected_execution_and_penalty_clauses',
                **{k: v for k, v in evidence.items() if k != 'event_year'})


TASK = """You are a rhythmic gymnastics judge assessing a complete individual routine with {apparatus}.
Estimate two separate quantities: execution_score, which includes artistic and technical execution, and penalty, which contains additional time, boundary and coordinator penalties. Start execution from 10.00 and subtract artistic and technical deductions; do not award extra execution points for harder elements. Do not subtract penalty from execution_score.

Observation:
The input contains video frames without audio. Musical accents, phrasing, rhythm and correspondence between music and movement cannot be established from these inputs. Do not invent musical faults, penalize the absence of audio, or assume that all music-related criteria were satisfied. The requested execution score is an estimate of the full execution target, not a complete audiovisual judging reconstruction. Assess visible body expression, connections and spatial composition where supported. Missing, cropped or occluded phases are not faults. Use continuous, adequately sampled movement evidence for timing and shape judgments; do not convert perspective or missing frames into exact angles or durations."""

ART_NEW = """Artistic execution:
- Develop a coherent movement character through the routine, including dance steps and connections. The character deduction is one overall choice: 0.10 when character is present in most but not all connections/dance steps; 0.30 when it is present only in some connections or not developed in the dance steps; 0.50 when it is absent between difficulties and appears only in dance steps; 0.70 when neither dance steps nor connections has defined character. Judge visible movement style, without claiming to verify its fit to unheard music.
- Logical, smooth connections: deduct 0.10 for each unjustified interruption, up to 1.00. Look for abrupt transitions, prolonged preparation or unnecessary stops, distinguishing intentional choreography. An unjustified starting position or a composed start/end without apparatus contact carries the listed 0.30 composition deduction; an accidental final apparatus loss belongs to the apparatus-loss rule below.
- Body expression: insufficient participation of body segments or face is 0.10; rigid, segmented movement throughout is 0.30. Use the applicable severity once for this aspect. Fewer than two different partial or full body waves is a separate 0.20 deduction.
- Space and variety: the direction/trajectory category carries 0.20 for insufficient variety, prolonged work in one place or failure to use the whole floor; do not charge these descriptions repeatedly for the same deficiency. Insufficient variety of levels and ways of travelling is a separate 0.20. Apparatus variety and required fundamental groups are specified below.
- A music-based dynamic-change judgment cannot be completed without audio. Observe visible contrasts of energy, speed and intensity, but do not manufacture a music-related deduction."""

ART_OLD = """Artistic execution:
- Seek a coherent artistic composition with logical connections and varied body/apparatus movements. An isolated break in a logical connection is 0.10; lack of unity and continuity over part of the routine is 0.50, or 1.00 over the entire routine. Use the applicable scope rather than charging the same connection defect again under a broader description.
- Lack of variety in body/apparatus movements is 0.10. An unjustified or extreme start/final position is 0.30. Starting and finishing positions should maintain apparatus contact; distinguish intended composition from accidental apparatus loss, whose deduction is specified below.
- Body expression: isolated segmented movement is 0.10; absence of body/facial expression through most of the routine is 0.30; total absence is 0.50. Use the applicable scope for the same deficiency. Lack of variation in speed and intensity throughout is 0.30 where established from movement.
- Insufficient variety in use of the floor, levels, directions/trajectories and ways of travelling is one 0.30 spatial-variety deduction. Do not count each description as a separate deduction for the same deficiency."""

BODY_SHARED = """- Incomplete movement or body adjustment while travelling without a throw is 0.10.
- Loss of balance: extra movement without travelling is 0.10; with travelling, 0.30; unintentional support on hand(s) or apparatus, 0.50; a fall, 0.70. These are alternative severities for one balance loss. Intentional floor support is not a fall.
- During a rotation intended on releve, unintended heel support is 0.10; a nonvertical axis ending with a step is 0.30.
- A heavy landing in a pre-acrobatic element is 0.10; unauthorized pre-acrobatic technique is 0.30. Do not treat permitted deliberate shapes or movements as faults."""
BODY_NEW = """Body technique:
- An incorrectly held body segment, including an incorrect foot/knee position, bent elbow or raised shoulders, is 0.10 each time; asymmetrical shoulders/trunk during a body difficulty is 0.30.
{shared}
- In jumps, balances and rotations, a recognizable shape deviation is 0.10/0.30/0.50 for small/medium/large deviation. Apply this scale to shape errors, not indiscriminately to every fault. Assess each distinct incorrect required body position; separate split and ring errors can both count. A separate kip movement has its own 0.10/0.30/0.50 deduction. Within one body difficulty, repeated deviations of the same position are assessed once at the greatest observed severity.
- Jump landing: 0.10 for a heavy landing, or 0.30 for a visibly arched back in the final landing phase. Do not add both for the same landing.
- Static balance: a shape not held for at least one second is 0.30; dynamic balances do not require a stop in each phase. During rotation, a shape neither fixed nor held in the base rotation is 0.30. Hops without travelling are 0.10; hops with travelling are 0.30.
- Walking on the hands in a handstand, with at least two visible support-changing steps, is 0.30.""".format(shared=BODY_SHARED)
BODY_OLD = """Body technique:
- An incorrectly held body segment is 0.10 each time, capped at 1.00 for this category.
{shared}
- A jump with insufficient shape amplitude or a heavy landing is assessed using the listed 0.10 fault. A balance with insufficient amplitude or a shape not fixed/held is 0.10. A rotation with insufficient amplitude or a shape not fixed/well-defined is 0.10; sliding during rotation is 0.10, while hops or an interruption are 0.30.
- Remaining motionless while not in contact with the apparatus is a 0.30 static-gymnast fault; do not confuse a deliberate moving element with immobility.""".format(shared=BODY_SHARED)

FUNDAMENTALS = {
    'Ball': 'a series of at least three small bounces, a high bounce or visible rebound; a roll over at least two large body segments; a figure-eight with arm circles; and a one-hand catch from a medium or large throw',
    'Clubs': 'mills with at least four small circles and alternating crossed/uncrossed wrists; asymmetric work of the two clubs; small throws of both clubs together or alternately with 360-degree rotation and catch; and small circles with both clubs, one in each hand',
    'Hoop': 'passage through the hoop; a roll over at least two large body segments; rotation around the hand or another body part; and free axial rotation between the fingers or on a body part',
    'Ribbon': 'passage through or over the ribbon pattern; spirals of four to five tight, evenly high loops; snakes of four to five tight, evenly high waves; and echappe with the stick rotating during flight',
}
APP_NEW = {
    'Ball': """For the ball, assess fluid control, bounces, body rolls, figure-eights and one-hand catches. Deduct 0.10 for visibly squeezing with the fingers or grasping against the forearm, an unintentionally incomplete body roll, a bouncing roll, or an involuntarily assisted two-hand catch. The assisted-catch exception is a catch outside the visual field; a deliberate permitted handling is not an involuntary fault.""",
    'Clubs': """For clubs, assess mills, small circles, asymmetric work and throws/catches with both clubs. Deduct 0.10 for irregular handling or excessively separated arms in mills, unintended alteration of rotational synchronization in throws/catches, or imprecise planes in asymmetric work. Intentional asymmetry is permitted. If both clubs are lost in succession, apply one loss deduction using the total steps to retrieve the farther club.""",
    'Hoop': """For the hoop, assess passage, body rolls, rotations around the body and axial rotations. Deduct 0.10 for unintended plane changes, vibration or irregular axial rotation, an unintentionally incomplete or bouncing body roll, sliding along the arm during rotation, or forearm contact on a post-throw catch. Upper-arm contact on that catch is 0.30; feet caught during passage are 0.30. Distinguish accidental contact from intended handling.""",
    'Ribbon': """For the ribbon, assess passage through/over patterns, spirals, snakes and echappe. Deduct 0.10 for malformed or loose/uneven patterns; imprecise handling/transitions, an unintended middle-of-stick grip or ribbon snaps; or up to one metre of the end unintentionally remaining on the floor during pattern work, throws or echappe. More than one metre remaining on the floor is 0.30. A knot without interruption is 0.30; a knot or involuntary body wrapping with interruption is 0.50. Intentional floor patterns and justified catches of the material within about 50 cm of its attachment are permitted.""",
}
APP_OLD = {
    'Hoop': """For the hoop, assess passage, smooth body rolls and controlled rotations. Incorrect handling with unintended plane changes or vibration is 0.10 each time, capped at 1.00 for that handling category. Irregular axial rotation, forearm contact after a throw, a bouncing/incomplete body roll, sliding on the arm, and feet caught in passage are listed 0.10 faults. Upper-arm contact after a throw is 0.30. Intended permitted handling is not accidental contact.""",
    'Ribbon': """For the ribbon, assess passage, spirals/snakes and echappe. Pattern alteration, incorrect handling/transitions or ribbon snaps, and insufficiently tight or uneven loops/waves are listed at 0.10 each time with a 1.00 cap on each listed repeated-fault category; do not count a malformed pattern twice under synonymous descriptions. A knot without interruption is 0.10; a knot or involuntary body wrapping with interruption is 0.30. The end unintentionally remaining on the floor during the specified work is 0.10 each time; deliberate permitted floor patterns are not accidental dragging.""",
}


def apparatus_rules(apparatus: str, year: int) -> str:
    if year == 2015:
        return "\n".join([
            'Apparatus execution:',
            '- Loss/retrieval: 0.30 without travelling, 0.50 with one or two steps, 0.70 with three or more steps or loss outside the floor regardless of distance. Loss followed by replacement apparatus, or loss with no apparatus contact at the end, is 0.70. Select the applicable loss case once.',
            '- An imprecise throw caught after one or two corrective steps is 0.10, or 0.30 after three or more. An incorrect catch/involuntary assistance by hand or body, or unintended body contact altering the trajectory, is 0.10. Deliberate choreography is not a corrective fault.',
            '- Static apparatus or nonconforming decorative handling is 0.30. Holding/squeezing the apparatus with the body through more than one body movement, holding it through a preparation plus its difficulty, or through two difficulties in succession is static apparatus. Brief decorative elements must preserve apparatus continuity, occur at most three times, and contain no more than four body movements without apparatus. Imbalanced right/left hand work is 0.30; work should be evenly distributed.',
            APP_OLD[apparatus]])
    extra = ''
    specific = APP_NEW[apparatus]
    catch_rule = ('An incorrect catch with involuntary body contact is 0.30.' if apparatus == 'Ball' else 'An involuntarily hand-assisted catch is 0.10; an incorrect catch with involuntary body contact is 0.30.')
    if year == 2019:
        extra = '\n- When a chase ends in apparatus loss, use the loss deduction including steps taken trying to catch it, rather than adding a successful-catch trajectory deduction. A separate body fall also incurs its body deduction.'
        if apparatus == 'Ribbon':
            specific = specific.replace('A knot without interruption is 0.30; a knot or involuntary body wrapping with interruption is 0.50. ', 'Involuntary body wrapping with interruption is 0.50. ')
            extra += '\n- Select the knot outcome once: 0.30 if it clears without interruption, 0.50 if the routine stops to untie it, or 0.70 if it makes the apparatus unusable and replacement is taken. Separately assess an actual interruption of composition/connections; a small knot does not automatically incur an artistic deduction.'
        if apparatus == 'Ball':
            extra += '\n- A one-hand catch immediately assisted by the second hand can satisfy the fundamental catch requirement while still incurring the applicable technical fault; do not also mark that group missing.'
    return '\n'.join([
        'Apparatus execution and variety:',
        '- Loss/retrieval: 0.50 without travelling, 0.70 with one or two steps, 1.00 with three or more steps or loss outside the floor regardless of distance. Loss followed by replacement while the original stays inside, or loss with no contact at the end, is 1.00. Select the applicable loss case once.',
        '- An imprecise throw caught after one corrective step is 0.30, or 0.50 after two or more corrective steps/chasse steps in rotational throwing elements. Intentionally choreographed travel is not an imprecise-trajectory fault. ' + catch_rule + ' Use the specific apparatus catch rule where it applies, rather than adding a generic catch charge for the same contact.',
        '- Firmly holding/squeezing stationary apparatus for more than four seconds, rather than moving it or maintaining unstable balance, is 0.30. Unusable apparatus is 0.70.',
        specific,
        '- Lack of variety in apparatus planes, directions and techniques is 0.20. Include at least one element from each of these fundamental groups: ' + FUNDAMENTALS[apparatus] + '. A genuinely missing group is 0.30 each; do not infer absence from occlusion or sparse frames.',
    ]) + extra

PENALTY = """Additional penalty:
- Routine time is 75-90 seconds, measured from the first movement to final complete stillness; add 0.05 per extra or missing second only when the actual interval is observable. File length, camera cuts and preparation are not competition time.
- Boundary: 0.30 for each body or apparatus occurrence touching the floor outside the boundary. Body and apparatus exits are separate occurrences. Touching the line or passing beyond it in the air without ground contact is not a boundary fault. The exception for apparatus lost on the last movement must be distinguished from a gymnast finishing outside, which remains penalized.
- Unauthorized retrieval, an unplaced replacement apparatus, or replacement while the original remains inside carries 0.50 where the relevant circumstances are visible. A legitimate replacement after loss outside does not by itself create that coordinator penalty. An apparatus loss outside can incur its execution deduction and a separate boundary penalty; these belong to different outputs.
- Do not invent administrative, equipment-compliance, clothing, start-order or coach-communication penalties from incomplete visual evidence. These require proof of the specific violation, not an assumption about unseen checks or instructions."""
PENALTY_NEW_EXTRA = """- Two clubs touching outside together are one apparatus boundary occurrence; separate exits are separate occurrences. Removing a broken/unusable apparatus outside is not a boundary penalty. Apply ending-related exceptions only when the ending can be established; unheard music cannot establish a music-dependent exception."""
OVERALL_SCORING = """Overall scoring rules:
Assess distinct faults once at their prescribed scope (per occurrence, element, category or whole routine), respecting the stated caps and alternatives. Do not charge the same continuous fault once per frame or subtract the same deduction under synonymous descriptions. Separate body, apparatus, artistic and boundary consequences may all count when each is actually present and the rules call for them; an apparatus error does not automatically imply an artistic interruption."""

OUTPUT = """Output format:
Return an estimate in the range 0-10 for execution_score and a nonnegative penalty. The following quantization is this task's output convention, not the precision of original panel marks: round execution_score to the nearest 0.1 and penalty to the nearest 0.05, with halfway values rounded up. Return only JSON with two numeric fields, formatting execution_score with one decimal place and penalty with two: {"execution_score": E, "penalty": P}."""


def system_prompt(apparatus: str, rule_profile: str) -> str:
    if apparatus not in APPARATUS:
        raise ValueError(f'Unknown RG apparatus: {apparatus}')
    years = {v['rule_profile']: y for y, v in PROFILES.items()}
    if rule_profile not in years:
        raise ValueError(f'Unknown RG rule profile: {rule_profile}')
    year = years[rule_profile]
    if year == 2015 and apparatus not in APP_OLD:
        raise ValueError('This release only has audited old-cycle hoop/ribbon samples')
    return '\n\n'.join([
        TASK.format(apparatus=apparatus.lower()),
        ART_OLD if year == 2015 else ART_NEW,
        BODY_OLD if year == 2015 else BODY_NEW,
        apparatus_rules(apparatus, year),
        PENALTY + ('\n' + PENALTY_NEW_EXTRA if year != 2015 else ''),
        OVERALL_SCORING,
        OUTPUT,
    ])


def user_prompt(apparatus: str) -> str:
    if apparatus not in APPARATUS:
        raise ValueError(f'Unknown RG apparatus: {apparatus}')
    return (f'<video>\nApparatus: {apparatus}.\n'
            'Assess this complete individual routine and return its execution score and additional penalty.')
