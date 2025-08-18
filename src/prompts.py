
def build_prompt(question: str, context: str | None = None, choices: list[str] | None = None) -> tuple[str, str]:
    """Build a user prompt and a system instruction string.
    If choices are provided, ask the model to answer with the LETTER only.
    """
    sys = (
        "You are a careful clinical QA assistant. "
        "Answer concisely using only information from the user message. "
        "If the answer cannot be determined from the provided text, reply exactly: Insufficient evidence."
    )
    if choices:
        letters = [chr(ord('A') + i) for i in range(len(choices))]
        choice_lines = "\n".join(f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices))
        user = (
            f"Question: {question}\n"
            f"{'Context: ' + context if context else ''}\n\n"
            f"Options:\n{choice_lines}\n\n"
            f"Respond with the single letter from [{', '.join(letters)}] only."
        )
    else:
        user = f"Question: {question}\n" + (f"Context: {context}\n" if context else "") + "\nAnswer in one short sentence."
    return sys, user
