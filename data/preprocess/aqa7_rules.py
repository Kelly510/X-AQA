
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "data.preprocess"

from .aqa7_execution_rules import DIVING_PROMPT, DIVING_OUTPUT, DIVING_SOURCE, diving_metadata, vault_rule

PAPER = "https://arxiv.org/pdf/1812.06367"
LONDON_GUIDE = "https://downloads.ctfassets.net/v8zaoys2pgzt/KByet28ZFZOWOeIk1AiAC/7f1309a660ab1275b0c4d979d2b76adb/london-2012-media-guide-final-version.pdf"

VIDEO_GUIDANCE = """
Video assessment:
Base your assessment on the visible performance. Ignore black padding. The video has been temporally normalized: do not treat playback seconds as measured real-world flight time or penalize padding as a pause in the performance. Do not invent faults in missing or obscured phases.
"""


ASSESSMENT_VIDEO_GUIDANCE = """
Video assessment:
Base judgments on visible movement. Do not infer faults or exact angles/distances from missing, obscured or cropped phases or misleading perspective. Ignore black padding. The video is temporally normalized: playback seconds do not measure real-world movement time, and padding is not a pause in the performance.
"""

BIG_AIR = """You are a judging-panel assessor evaluating one {equipment} Big Air attempt.

Give one overall score from 0 to 50 for this attempt. Include technical difficulty and execution; do not sum scores from multiple attempts.

Assessment criteria:
1. Take-off and amplitude
- Assess useful height and distance that allow the trick to be completed. Judge amplitude relative to the jump and camera view, not raw pixel height.

2. Flight: difficulty, execution and style
- Difficulty: consider rotations, multiple-axis or inverted motion, and take-off/landing direction. More rotation alone does not establish a better performance.
- Execution: assess control of the rotation axis, body position and equipment, the quality of a deliberate grab where visible, and completion of the intended rotation. A missed or prematurely released grab, loss of control or incomplete rotation lowers the assessment.
- Progression/style: consider the visible sophistication and intentionality of the trick. Do not invent a novelty bonus or assume it is the first-ever performance of that trick.

3. Landing and ride-away
- Assess alignment at contact, balanced absorption and a controlled ride-away.
- Distinguish a small correction from a sustained hand drag, major loss of balance or a crash. With otherwise comparable difficulty and execution, a crashed attempt must score lower than a cleanly landed attempt.

Overall assessment:
- Integrate difficulty, execution, amplitude, progression/style and landing into one overall-impression score. Do not assign equal component weights or calculate separate component scores to add together.
- Account for each distinct defect once, even if several criteria describe it or it persists across frames. Independent errors still affect the overall score.
- No fixed points-per-fault tariff or numerical quality-band boundaries are specified here; do not invent them or rescale the task to 100 points.
"""

SYNC = """You are a diving judging-panel assessor evaluating one synchronized {event} dive.

Estimate the total score for this dive, including both divers' execution, synchronization and difficulty.

Judging scale:
Individual execution and synchronization awards use 0.0-10.0 in increments of 0.5:
- 10.0: excellent.
- 8.5-9.5: very good.
- 7.0-8.0: good.
- 5.0-6.5: satisfactory.
- 2.5-4.5: deficient.
- 0.5-2.0: unsatisfactory.
- 0.0: completely failed.
This scale applies to the component awards, not to the final dive total.

Assessment criteria:
1. Starting position and take-off
- Execution: assess each diver's starting posture, balance and confident take-off with adequate height. Incorrect starting position or deficient take-off warrants 0.5-2.0 from the affected diver's execution award.
- Synchronization: compare starting positions, approach and take-off timing, and the height achieved by the two divers.

2. Flight
- Execution: assess each diver's rotation control and body position. Intended tuck or pike shapes are not form errors; defects in the appropriate body position warrant 0.5-2.0 from that diver's execution award.
- Synchronization: compare the timing and coordination of the movements through flight.

3. Opening and entry
- Execution: assess each diver's opening, rotation completion, vertical alignment, body straightness and entry control. Consider splash with alignment and control.
- Synchronization: compare entry angles, distance from the {apparatus}, and simultaneous water contact.
- Judge visible entry faults and synchronization differences by severity; do not invent a degrees-to-points or milliseconds-to-points formula.

Overall assessment and total:
- Judge each diver's execution independently of difficulty and of the other diver's execution. Judge synchronization by how well the pair matches: matching poor dives may synchronize well, while two good dives may be out of time.
- Use the quality scale for overall component judgments. Within a 0.5-2.0 deduction range, use the lower end for a minor fault and the higher end for a severe fault. Do not downgrade a component for an error and then subtract the same error again, or count a continuous fault once per frame. A fault affecting both execution and synchronization may affect both distinct components.
- The eleven-judge panel has three execution judges for each diver and five synchronization judges. Discard the highest and lowest execution awards for each diver, and the highest and lowest synchronization awards. Retain one execution award per diver and three synchronization awards.
- The dive total is 0.6 multiplied by difficulty multiplied by the sum of those five retained awards. Synchronization therefore supplies three of five retained awards. Do not subtract execution deductions directly from the dive total or add a separate difficulty bonus.
- The announced dive, difficulty and judge marks are not supplied as structured inputs. Estimate the total from the visible performance; do not claim exact reconstruction or invent referee decisions, unlisted numerical caps or a set of official judge marks.
- Assess the dive from the starting position until both divers are fully submerged; disregard the walk to the starting position and movements beneath the water.
"""

def rubric(sport: str, index: int) -> tuple[str, str, dict]:
    if sport == "diving":
        return DIVING_PROMPT + VIDEO_GUIDANCE + DIVING_OUTPUT, "As one judge, what execution award would you give this dive?", diving_metadata()
    if sport == "gym_vault":
        prompt, meta = vault_rule(index, video_guidance=ASSESSMENT_VIDEO_GUIDANCE)
        return prompt, "As one execution judge, what execution score would you give this vault?", meta
    if sport in ("ski_big_air", "snowboard_big_air"):
        equipment = "ski" if sport == "ski_big_air" else "snowboard"
        return BIG_AIR.format(equipment=equipment) + ASSESSMENT_VIDEO_GUIDANCE + (
            '\nOutput only this JSON object and nothing else: '
            '{"overall_score": <integer from 0 to 50>}.\n'
        ), "What score would your panel give this single Big Air attempt?", {
            "event_year": None, "event_name": "X Games (exact event unverified)",
            "event_attribution": "dataset_paper_and_sampled_video", "discipline": sport,
            "rule_edition": None, "rubric_scope": "historical_format_qualitative_year_unverified",
            "rule_sources": [PAPER],
        }
    if sport in ("sync_diving_3m", "sync_diving_10m"):
        event = "3m springboard" if sport == "sync_diving_3m" else "10m platform"
        meta = diving_metadata() | {"discipline": sport, "event_attribution": "manual_video_event_identification",
                                  "rubric_scope": "historical_panel_total_estimation", "rule_sources": [DIVING_SOURCE, LONDON_GUIDE, PAPER]}
        return SYNC.format(event=event, apparatus="springboard" if sport == "sync_diving_3m" else "platform") + ASSESSMENT_VIDEO_GUIDANCE + (
            '\nOutput only this JSON object and nothing else: '
            '{"dive_total": <nonnegative number with two decimal places>}.\n'
        ), "What total score would your panel give this synchronized dive?", meta
    raise ValueError(f"Unknown rubric: {sport}/{index}")
