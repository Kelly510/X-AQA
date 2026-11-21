FIG_BASE = "https://gymnasticsresults.com/technical/code-of-points/"
DIVING_SOURCE = "https://de.scribd.com/document/226936907/Diving-officials-manual"


VAULT_EVENTS = (
    (1, 15, 2012, "MAG", "London Olympics"),
    (16, 23, 2011, "MAG", "Tokyo Worlds"),
    (24, 38, 2013, "MAG", "Antwerp Worlds"),
    (39, 46, 2015, "MAG", "Glasgow Worlds"),
    (47, 60, 2013, "WAG", "Antwerp Worlds"),
    (61, 76, None, "WAG", None),
    (77, 92, 2011, "WAG", "Berlin Europeans"),
    (93, 104, 2015, "WAG", "Montpellier Europeans"),
    (105, 120, 2014, "WAG", "Nanning Worlds"),
    (121, 134, 2008, "WAG", "Beijing Olympics"),
    (135, 145, 2012, "WAG", "London Olympics"),
    (146, 160, 2007, "WAG", "Stuttgart Worlds"),
    (161, 176, 2009, "WAG", "London Worlds"),
)


DIVING_CRITERIA = """Execution criteria and deductions:

1. Starting position and take-off
- For a standing start, stand naturally with the body straight, head erect and arms straight in any position. For an armstand start, establish a steady, straight vertical balance.
- Deduct 0.5-2.0 for an incorrect starting position, loss of steady balance in an armstand, or a crow-hop on a standing dive.
- For a running dive, the approach must be smooth and continuous forwards to the end of the platform; approach faults warrant 0.5-2.0.
- Deduct 0.5-2.0 for insufficient take-off height or confidence, failure to take off from the end of the platform, or twisting visibly initiated on the platform.

2. Flight
- Assess rotation control and body shape. Straight: keep hips and knees extended. Pike: bend at the hips while keeping the knees extended. Arms may be placed freely in straight and pike positions. Tuck: bend the hips and knees into a compact shape, with knees together and hands on the lower legs. Free position, permitted for designated twisting dives, allows combinations of these shapes. An intended tuck, pike or permitted free-position transition is not a form error.
- Throughout flight and entry, keep the feet together and toes pointed. Deduct 0.5-2.0 for flight-position faults. For a dive identifiable as pike or tuck with twist, failure to clearly show that position warrants 0.5-2.0; do not impose a fixed pike or tuck shape on a free-position dive.
- Evaluate platform clearance and the flight path. Reduce the award for contact with the platform, unsafe proximity, or sideways deviation from the direct line of flight, according to severity and its effect on execution.

3. Opening and entry
- Assess opening timing, completion of rotation, body straightness and entry control. When the visible sequence supports the distinction, failure to complete rotation is more serious than a controlled dive with a mistimed opening at a similar entry angle; do not infer the cause from the entry angle alone.
- Apply these quality-band guides only when the camera view supports a reliable judgment of the relevant angle:
  * Tilt away from vertical: more than 5 degrees limits the award to good or lower (at most 8.0); more than 35 degrees limits it to deficient or lower (at most 4.5).
  * Twist remaining at entry: more than 5, 15, or 35 degrees away from the correct untwisted orientation limits the award to at most 8.0, 6.5, or 4.5, respectively. This is rotation around the body's long axis, not tilt away from vertical or the body's orientation relative to an arbitrary camera view.
  If an angle cannot be judged reliably, assess the visible alignment qualitatively without assigning an angle-based ceiling.
- For a head-first entry, extend the arms beyond the head in line with the body and bring the hands close together. If one or both arms remain below the head, the award must not exceed 4.5. Other arm-position faults warrant 0.5-2.0.
- Consider splash with alignment and control. With otherwise equivalent execution and a vertical entry, a clean rip entry merits about one point more than a comparable entry without a rip. Use this comparison when forming the overall award, not as an extra bonus; a small splash cannot compensate for incomplete rotation or poor alignment.
"""

VAULT_END = """
Overall assessment:
- Choose each deduction according to the visible severity, using the stated small/medium/large values or permitted range. Do not invent numerical angle thresholds or fixed deductions for unlisted faults.
- Count each distinct execution fault once, even if it persists across frames or matches overlapping descriptions. Independently recurring errors in separate phases remain assessable. Apply any stated landing limit only to the deductions it covers.
- Subtract the execution deductions from 10.0. Do not lower an overall-impression score first and then subtract the same faults again. Exclude difficulty and neutral/line penalties; do not reconstruct a competition total or infer an unseen referee decision.
"""

VAULT_OUTPUT = """
Output only this JSON object and nothing else:
{"execution_score": <number from 0.0 to 10.0 in increments of 0.1>}.
"""


def vault_rule(index: int, *, video_guidance: str = "") -> tuple[str, dict]:
    matches = [e for e in VAULT_EVENTS if e[0] <= index <= e[1]]
    if len(matches) != 1:
        raise ValueError(f"Missing/overlapping vault provenance: {index}")
    _, _, year, discipline, event = matches[0]
    era = "2006-2008" if year and year < 2009 else "2009-2012" if year and year < 2013 else "2013-2016" if year else None
    sources = ([FIG_BASE + f"wag/{e}-wag-cop.pdf" for e in ("2006-2008", "2009-2012", "2013-2016")]
               if era is None else [FIG_BASE + f"{discipline.lower()}/{era}-{discipline.lower()}-cop.pdf"])
    prompt = f"""You are one execution judge evaluating a {'women' if discipline == 'WAG' else 'men'}'s gymnastic vault.

Give one execution score on a 0.0-10.0 scale, starting from 10.0 and deducting for execution faults.

Execution criteria and deductions:
1. First flight and support on the table
"""
    if discipline == "WAG":
        prompt += """- Bent knees in flight or bent arms during table support: 0.1 / 0.3 / 0.5 for small / medium / large errors.
- Leg or knee separation in flight: 0.1 or 0.3; crossed legs during twisting: 0.1. These flight-form deductions also apply during second flight.
"""
    else:
        prompt += """- Bent arms during support, bent legs or poor body posture: 0.1 / 0.3 / 0.5 for small / medium / large errors. The corresponding form deductions also apply during second flight.
"""
    prompt += """
2. Second flight
- Assess body position and control after leaving the table. An intended tuck or pike is not itself a bent-body fault.
"""
    if discipline == "WAG" and era is not None:
        if era == "2013-2016":
            prompt += """- Insufficient height: 0.1 / 0.3 / 0.5. Insufficient distance: 0.1 or 0.3.
- Insufficient or late opening of a tuck/pike vault: 0.1 or 0.3.
"""
        else:
            prompt += "- Insufficient height: 0.1 / 0.3 / 0.5 / 0.8. Insufficient distance: 0.1 / 0.3 / 0.5.\n"
            prompt += "- Insufficient or late opening of a tuck/pike vault: 0.1 or 0.3.\n"
    elif discipline == "MAG":
        prompt += "- Insufficient height: 0.1 / 0.3 / 0.5.\n"
    prompt += "\n3. Landing\n"
    if discipline == "WAG":
        prompt += """- An extra small step or slight hop: 0.1 each; a very large step or jump: 0.3 each.
- Extra arm swing to regain balance: 0.1; poor landing posture: 0.1 or 0.3; a deep squat: 0.5.
"""
        if era is None:
            prompt += "- Treat a fall as a major execution failure. No fixed fall deduction or landing cap is specified here; do not invent one.\n"
        else:
            fall = "0.8" if era == "2006-2008" else "1.0"
            prompt += f"- Supporting the body with one or both hands on the landing mat, or falling to knees/hips after feet-first contact: {fall}.\n"
            if era == "2006-2008":
                prompt += "- The combined deduction for steps/hops is at most 0.7; the very-large-step/jump distance criterion is 1 metre.\n"
            else:
                prompt += "- A very large step/jump is guided by more than shoulder width. Without a fall, total landing deductions cannot exceed 0.8.\n"
    else:
        prompt += """- A minor landing adjustment or excessive arm swing: 0.1. A small step/hop: 0.1 each; a large step/hop: 0.3.
- Touching the mat with one or both hands without supporting body weight: 0.3.
- A fall after feet-first landing: 1.0.
"""
        if era == "2009-2012":
            prompt += "- Supporting body weight with one or both hands after feet-first landing, without a fall: 0.5.\n"
        else:
            prompt += "- Supporting body weight with one or both hands after feet-first landing: 1.0. Treat support and a fall in the same landing failure as one 1.0 deduction.\n"
            prompt += "- Without a fall or hand support, the combined deduction for steps and hops cannot exceed 1.0. This is not a cap on all landing faults.\n"
    return prompt + VAULT_END + video_guidance + VAULT_OUTPUT, {
        "event_year": year, "event_name": event, "discipline": discipline,
        "event_attribution": "manual_video_and_event_results" if year else "unresolved",
        "rule_edition": f"FIG {discipline} {era}" if era else "FIG WAG historical common subset",
        "rule_sources": sources, "rubric_scope": "selected_visible_faults" if year else "limited_year_unverified",
    }


def diving_metadata() -> dict:
    return {"event_year": 2012, "event_name": "London Olympics", "discipline": "diving",
            "event_attribution": "dataset_paper", "rule_edition": "FINA 2009-2013",
            "rule_sources": [DIVING_SOURCE], "rubric_scope": "selected_visible_faults"}


DIVING_PROMPT = """You are one execution judge at a 10-m platform diving competition.

Give one overall execution award from 0.0 to 10.0 in increments of 0.5.

Use this judging scale:
- 10.0: excellent.
- 8.5-9.5: very good.
- 7.0-8.0: good.
- 5.0-6.5: satisfactory.
- 2.5-4.5: deficient.
- 0.5-2.0: unsatisfactory.
- 0.0: completely failed.

""" + DIVING_CRITERIA + """
Overall assessment:
- Form one overall-impression award using the quality scale and execution criteria, without assigning equal phase weights. The listed deductions lower the award for the corresponding faults; use the lower end of each range for a minor fault and the higher end for a severe fault.
- Account for each distinct fault once: do not lower the overall impression for a fault and then subtract for it again, or count it again because it appears in several frames or matches overlapping descriptions. Separate errors in different phases still affect the award.
- Apply the most restrictive applicable angle guide or mandatory cap to the resulting award. These are ceilings, not additional deductions or a points-per-degree formula. Choose the final award in half-point increments within 0.0-10.0.
- Judge execution independently of difficulty, from the starting position until the body is fully submerged; disregard the walk to the starting position and movements beneath the water. Base judgments only on clearly visible evidence. Do not infer faults or dive requirements from cropped, obscured or unsampled footage or misleading perspective, or invent an announced dive, referee-declared penalty or unlisted numerical cap.
"""


DIVING_OUTPUT = """
Output only this JSON object and nothing else:
{"execution_award": <number from 0.0 to 10.0 in increments of 0.5>}."""
