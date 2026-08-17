import re

non_countable_quantities = [
    # time
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "[a-z]*seconds",

    # length/area
    "(tera|giga|mega|deci|kilo||micro|centi|milli|nano|pico|deca)meters",
    "meters",
    "metres",  # mispelt meters
    "acres",
    "leagues",
    "fathoms",
    "nautical miles",
    "hectares",
    "(square |SQ )?inches",
    "(square |SQ )feet",  # Just feet can be a false positive
    "(square |SQ )?ft",
    "(square |SQ )?miles",
    "(square | SQ)?yards",
    "passing yards",

    # currency
    "dollars",
    "cents",
    "pounds",
    "euros",

    # speed
    "seed",
    "mph",
    "kph",

    # Comparisons "how many more..."
    "more",
    "fewer",
    "less",

    "likes",  # almost always from a screenshot

    # volume
    "cubic",
    "gallons",
    "quarts",
    "pints",
    "fluid ounces",
    "[a-z]*liters",
    # ambiguous and probably more often used as an object then a volume
    # cup
    # tablespoons
    # teaspoons

    # weight
    "weight",
    "[a-z]*grams",
    "pounds",
    "tons",
    "ounces",

    "ways", "different ways",

    # other
    "degrees", "calories",
    "hertz", "horsepower", "[a-z]*bytes",
    "psi", "atmospheres", "[a-z]*watts",
]
non_countable_re_str = "|".join(non_countable_quantities)
non_countable_end_re_str = "|".join(non_countable_quantities + ["money", "the"])

counting_patterns = [
    f'how ?many (?!{non_countable_re_str})',
    r'(?<!do not )(count|tally) (all|every|each|the) ',
    "(there are|a total of) _{3,4}",
    f"(what|(what's|what (is|was|were)|states?|indicates?) the( exact| precise)?) (total count|count|total|total number|number|num|total amount|amount) of (?!{non_countable_end_re_str})",
]
count_any = re.compile("^(?!approximately).*(\\b|^|\n)(?P<all>" + "|".join(counting_patterns) + ")\\b.*", re.IGNORECASE | re.MULTILINE | re.DOTALL)
how_many_re = re.compile(".*(\\b|^)(" + counting_patterns[0] + ")\\b.*", re.IGNORECASE)

counting_start = re.compile("^(?P<all>" + f'(how ?many |count the )(?!{non_countable_re_str})' + ")\\b.*", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def has_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(text))


# These are some common pattern in the PixMo data
complex_question_ops = [
    "in the style of",
    "in (german|spanish)",
    "in a (table|json)",
    "(choices|options):",  # multiple-choice question
]
complex_question_re = re.compile(".*(\\b|^)(" + "|".join(complex_question_ops) + ")\\b.*", re.IGNORECASE)


def is_pixmo_point_and_count_question(question, answer=None):
    """
    Could this  question be counting question that the model should use pointing for?

    This check is conservative, so it will have a high recall but low precision
    """
    return bool(count_any.fullmatch(question))


def is_counting_question(question, style=None, check_multi_line=False):
    return bool(count_any.fullmatch(question))
        # remainder = question[match.end("all"):]
        # if check_multi_line and two_sentence_pattern.match(question):
        #     return False
        # return True
    # return False
        # if style and style.startswith("mantis") and ". Which group has" in question:
#         # This handles a few mantis questions that ask "Count the.. Which group has..."

    # if style and style.startswith("ai2_diagram"):
    #     # AI2D has a lot tricky numeric questions like
    #     # "What would happen if the number of lobsters increased?"
    #     # So just use how-many and 'what is the number of', which works well on AI2D
    #     return how_many_re.fullmatch(question) or "what is the number of" in question.lower()
    #
    # if count_any.fullmatch(question):
    #     if style and style.startswith("mantis") and ". Which group has" in question:
    #         # This handles a few mantis questions that ask "Count the.. Which group has..."
    #         return False
    #     return True
    # else:
    #     return False


def test_is_counting_question():
    assert is_counting_question("how many times does he smile?")
    assert is_counting_question("There are ___  cats.")
    assert is_counting_question("Count all the cats")
    assert not is_counting_question("Do not count all the cats")
    assert is_counting_question("Count the cats")
    assert is_counting_question("What is the exact number of performers in the video?")
    assert is_counting_question("Tell me, how many cats?")
    assert is_counting_question("Count the cats?")
    assert is_counting_question("What's the number of dogs?")
    assert not is_counting_question("What is the number of degrees in the cricle?")
    assert is_counting_question("What number of zebras are standing in front of the tree surrounded by a chain link fence?")
    assert is_counting_question("What is the number of nice elephants who are living inside the zoo enclosure?")
    assert is_counting_question("What amount of children are sitting in front of the TV, when Mrs. Allen opens the door?")
    assert is_counting_question("How many cup are shown in this video?")
    assert is_counting_question("""
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
How many taillights does the player's car have?    
    """.strip())
    assert is_counting_question("""
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
In the video, how many times does the male protagonist do hanging leg raises per set in the first phase of training?    
    """.strip())

    assert not is_counting_question("What amount of money was spent?")
    assert not is_counting_question("What is the maximum number of shoes present?")
    assert not is_counting_question("What is the number written on top of the middle green bananas?")
    assert not is_counting_question("What would happen if the number of lobsters increased?",
                                    style="ai2_diagram_v2_mix_transparent")
    assert not is_counting_question("What number is on the yellow train?")
    assert not is_counting_question("What country is likely hosting this vehicle evident by the writing on its side?")
    assert not is_counting_question("Count the tomatoes in this group. Which group has one less than the group you counted?",
                                    style="mantis")
    assert not is_counting_question("Approximately how many people live in this city?")
    assert not is_counting_question("How many watts does a night lamp use?")
    assert not is_counting_question("How many miles are there?")
    assert not is_counting_question("What is one change to the ecosystem that would increase the number of frogs?")


if __name__ == '__main__':
    dbg = [
        # "Count the number of rabbits in image 3 and add that to the number of rabbits in image 2.",
        "how many thermometers have rust."
    ]
    for query in dbg:
        print(query)
        print(is_pixmo_point_and_count_question(query, ""))
