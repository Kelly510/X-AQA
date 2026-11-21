POSITIVE_FEATURES = {
    "jump": (
        "Strong elevation and travel distance on every jump in the element.",
        "Controlled take-off followed by a secure landing.",
        "An easy, fluent execution throughout, including the connections in combinations.",
        "Visible steps leading into the jump or an inventive, unexpected entrance.",
        "A well-controlled body shape from launch through landing.",
    ),
    "spin": (
        "Strong rotation speed or a clear increase in speed.",
        "Clearly established, controlled shapes, including the airborne and landing phases of a flying spin.",
        "An easy, fluent execution throughout the spin.",
        "Rotation around a stable center rather than drifting across the ice.",
        "Inventive or original movement choices.",
    ),
    "step": (
        "Deep skating edges with precise steps and turns.",
        "Fluent execution with sustained energy and little visible effort.",
        "Inventive or original movement choices.",
        "Committed movements with coordinated control of the entire body.",
        "Clearly controlled increases and decreases in speed.",
    ),
    "choreographic": (
        "Inventive or original movement choices.",
        "Fluent execution with sustained energy and little visible effort.",
        "Broad use of the ice surface.",
        "Clearly defined, precise movements.",
        "Committed movements with coordinated control of the entire body.",
    ),
}

REDUCTIONS = {
    "jump": (
        "Fall: subtract 5.",
        "Two-foot landing or a step-out used to recover the landing: subtract 3 to 4.",
        "Support on both hands: subtract 2 to 3; touching with one hand or the free foot: subtract 1 to 2.",
        "Poor speed, elevation, travel distance or airborne posture: subtract 1 to 3.",
        "A weak landing, such as an awkward position, scraping or an incorrect landing edge: subtract 1 to 3.",
    ),
    "spin": (
        "Fall: subtract 5.",
        "Touching the ice with the free foot or a hand: subtract 1 to 3.",
        "Drifting away from the rotation center: subtract 1 to 3.",
        "Slow rotation or an unintended drop in speed: subtract 1 to 3.",
        "Poorly formed or awkward positions: subtract 1 to 3.",
        "A poorly controlled flying entrance or change of foot: subtract 1 to 3.",
        "An incorrect take-off or landing in a flying spin: subtract 1 to 2.",
    ),
    "step": (
        "Fall: subtract 5.",
        "A stumble: subtract 1 to 3.",
        "Poor execution of steps, turns or body positions: subtract 1 to 3.",
        "Insufficient flow or energy: subtract 1 to 3.",
    ),
    "choreographic": (
        "Fall: subtract 5.",
        "A stumble: subtract 1 to 3.",
        "Loss of control or insufficient energy: subtract 1 to 3.",
        "Poorly executed movements: subtract 1 to 3.",
        "Little creativity or originality: subtract 1 to 3.",
    ),
}


def rule_key(category: str, element: str) -> str:
    if category in ("jump", "spin"):
        return category
    if category == "sequence":
        if element.startswith("StSq"):
            return "step"
        if element.startswith("ChSq"):
            return "choreographic"
    raise ValueError(f"no GOE rubric for {category!r}, {element!r}")


def detailed_rules(category: str, element: str, season: str = "2018/19", segment: str = "sp") -> str:
    if season not in ("2018/19", "2019/20", "2020/21", "2021/22"):
        raise ValueError(f"unsupported FineFS season: {season!r}")
    if segment not in ("sp", "fs"):
        raise ValueError(f"unsupported segment: {segment!r}")
    key = rule_key(category, element)
    old = season == "2018/19"
    recent = season in ("2020/21", "2021/22")
    positive = list(POSITIVE_FEATURES[key])
    if key == "choreographic" and not old:
        positive[2] = "Broad use of the ice surface or an interesting movement pattern."
    features = "\n".join(f"- {text}" for text in positive)
    faults = list(REDUCTIONS[key])
    if key == "jump":
        faults += [
            "Missing half a revolution or more on landing: subtract 3 to 4.",
            ("Missing more than a quarter but less than half a revolution: subtract 2 to 3."
             if recent else "Missing at least a quarter but less than half a revolution: subtract "
             + ("2 to 3." if old else "1 to 2.")),
            ("Landing exactly a quarter revolution short: subtract 2; less than a quarter short: subtract 1."
             if recent else "Missing less than a quarter revolution: subtract 1 to 2."),
            "Clearly wrong take-off edge on a flip or Lutz: subtract " + ("2 to 4." if recent else "3 to 4."),
            "Visibly flat or ambiguous flip/Lutz take-off edge: subtract " + ("1 to 2." if recent else "1 to 3."),
            "Poor take-off: subtract " + ("1 to 3." if recent else "2 to 3."),
            "Long preparation: subtract " + ("2 to 3." if old else "1 to 3."),
        ]
        if "+" in element:
            faults += [
                "Two three-turns between jumps: subtract 2 to 3.",
                "Loss of flow, direction or rhythm between jumps: subtract " + ("1 to 3." if recent else "2 to 3."),
            ]
            if "Eu" in element:
                faults.append("An Euler with visibly incomplete rotation: subtract 1 to 2." if old
                              else "An Euler performed as a step-over: subtract 1 to 2.")
    elif key == "spin":
        faults += [
            "Fewer revolutions than a known requirement: subtract 1 to 3; apply only when the complete spin and requirement are available.",
            "An unbalanced number of revolutions on the two feet in a change-foot spin: subtract 1.",
        ]
        if not old:
            faults.append("Loss of balance: subtract 1 to 3.")
    elif key == "step":
        if old:
            faults[-1] = "Steps and turns occupy less than half of the sequence pattern: subtract 2 to 3."
        if segment == "sp":
            faults.append("A listed jump with more than half a revolution inside the step sequence: subtract 1.")
    elif key == "choreographic":
        if old:
            faults[2] = "Loss of control while executing the sequence: subtract 1 to 3."
            faults[3] = "Poorly executed movements: subtract 1 to 2."
            faults.append("Failure to demonstrate a recognizable sequence: subtract 2 to 3.")
        else:
            faults.append("Gaps or missing connections between choreographic movements: subtract 2 to 3."
                          if recent else "Lack of choreographic movements: subtract 2 to 3.")
    reductions = "\n".join(f"- {text}" for text in faults)
    required_features = {
        "jump": "strong elevation and distance, controlled take-off and landing, and easy, fluent execution throughout",
        "spin": "strong speed or acceleration, clear controlled shapes, and easy, fluent execution throughout",
        "step": "deep edges with precise steps and turns, and fluent, energetic execution with little visible effort",
        "choreographic": "inventive or original movements, and fluent, energetic execution with little visible effort",
    }[key]
    high_grade = (
        f"Grades +4 and +5 require {required_features}; otherwise cap the starting grade at +3."
        if key in ("jump", "spin") else
        f"For a visual estimate of +4 or +5, both of the following must be clearly present: {required_features}."
    )
    cap = ("A fall, two-foot landing, step-out, clearly wrong flip/Lutz take-off edge, or missing "
           "half a revolution or more limits the starting grade to +2."
           if key == "jump" else "A fall limits the starting grade to +2.")
    if recent:
        cap += " Multiple distinct errors in this element also cap the starting grade at +2."
    return (
        "1. Positive features:\n" + features + "\n\n"
        "2. Starting grade and limits:\n"
        "Use the number of clearly demonstrated features as a visual starting estimate: "
        "none = 0; one = +1; two = +2; three = +3; four = +4; five = +5. "
        + high_grade + " Apply the following limits before subtracting errors: " + cap + "\n\n"
        "3. Visible error reductions:\n" + reductions + "\n"
        "Use the smaller deduction for a mild fault and the larger deduction for a severe fault. "
        "Add deductions for distinct errors; count the same fault only once, using its most specific category. "
        "An ambiguous camera view is not evidence of a flat or wrong edge, missing rotation, or another fault.\n\n"
        "4. Final grade:\n"
        "Subtract the total deduction from the capped starting grade, then clamp to [-5, +5]. "
        "For example, starting at +2 and subtracting 5 for a fall gives -3. "
        "Use a half-point when the visible quality lies between adjacent integer grades. "
        "Assess only the supplied element; do not add program-level deductions or change its base value."
    )
