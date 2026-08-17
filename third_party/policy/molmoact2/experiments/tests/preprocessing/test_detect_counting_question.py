from olmo.preprocessing.detect_counting_question import is_pixmo_point_and_count_question


def test_is_counting_question():
    assert is_pixmo_point_and_count_question("how many times does he smile?")
    assert not is_pixmo_point_and_count_question("Does he smile?")
    assert is_pixmo_point_and_count_question("There are ___  cats.")
    assert is_pixmo_point_and_count_question("Count all the cats")
    assert not is_pixmo_point_and_count_question("Do not count all the cats")
    assert is_pixmo_point_and_count_question("Count the cats")
    assert is_pixmo_point_and_count_question("What is the exact number of performers in the video?")
    assert is_pixmo_point_and_count_question("Tell me, how many cats?")
    assert is_pixmo_point_and_count_question("Count the cats?")
    assert is_pixmo_point_and_count_question("What's the number of dogs?")
    assert not is_pixmo_point_and_count_question("What is the number of degrees in the cricle?")
    assert is_pixmo_point_and_count_question("What number of zebras are standing in front of the tree surrounded by a chain link fence?")
    assert is_pixmo_point_and_count_question("What is the number of nice elephants who are living inside the zoo enclosure?")
    assert is_pixmo_point_and_count_question("What amount of children are sitting in front of the TV, when Mrs. Allen opens the door?")
    assert is_pixmo_point_and_count_question("How many cup are shown in this video?")
    assert is_pixmo_point_and_count_question("""
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
How many taillights does the player's car have?    
    """.strip())
    assert is_pixmo_point_and_count_question("""
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
In the video, how many times does the male protagonist do hanging leg raises per set in the first phase of training?    
    """.strip())

    assert not is_pixmo_point_and_count_question("What amount of money was spent?")
    assert not is_pixmo_point_and_count_question("What is the maximum number of shoes present?")
    assert not is_pixmo_point_and_count_question("What is the number written on top of the middle green bananas?")
    assert not is_pixmo_point_and_count_question("What number is on the yellow train?")
    assert not is_pixmo_point_and_count_question("What country is likely hosting this vehicle evident by the writing on its side?")
    assert not is_pixmo_point_and_count_question("Approximately how many people live in this city?")
    assert not is_pixmo_point_and_count_question("How many watts does a night lamp use?")
    assert not is_pixmo_point_and_count_question("How many miles are there?")
    assert not is_pixmo_point_and_count_question("What is one change to the ecosystem that would increase the number of frogs?")

