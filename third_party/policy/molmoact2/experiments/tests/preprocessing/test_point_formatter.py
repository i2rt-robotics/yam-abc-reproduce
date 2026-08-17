import numpy as np
from olmo.preprocessing.data_formatter import seconds_to_timestamp
from olmo.preprocessing.point_formatter import UnifiedPointFormatter, LegacyPointFormatting
from olmo.util import parse_timestamp


def test_seconds_to_timestamp():
    assert "00:00:01.00" == seconds_to_timestamp(1, 2)
    assert "00:00:01.0" == seconds_to_timestamp(1, 1)
    assert "00:01:40.1" == seconds_to_timestamp(100.1, 1)
    assert "03:00:00.1" == seconds_to_timestamp(60*60*3 + 0.09, 1)
    assert "03:00:00.0" == seconds_to_timestamp(60*60*3 + 0.01, 1)
    assert "03:00:00.01" == seconds_to_timestamp(60*60*3 + 0.01, 2)


def test_parse_timestamp():
    assert 1 == parse_timestamp("00:00:01.00")
    assert 61 == parse_timestamp("00:01:01.00")
    assert 61 == parse_timestamp("01:01.00")
    assert 61.01 == parse_timestamp("01:01.01")
    assert 60*60+61 == parse_timestamp("01:01:01.00")


def test_legacy_compact():
    formatter = LegacyPointFormatting(pointing_format="compact_v1")
    assert (formatter.format_image_points(np.array([[0.99, 0.8]]), 1.0, "label") ==
            'Counting the <points label\t-\t1 99.0 80.0/> shows a total of 1.')

    actual = formatter.format_video_points(
        [0.1],
        [[[0.911, 0.8], [0.9, 0.4]]], 100.0, "aasdf",
        mode="point"
    )
    assert actual == '<points 0.1\t1 0.9 0.8 2 0.9 0.4/>'

    actual = formatter.format_video_points(
        [0.1, 0.2],
        [[[0.911, 0.8]], [[0.9, 0.4]]], 100.0, "aasdf",
        mode="point"
    )
    assert actual == '<points 0.1\t1 0.9 0.8/> <points 0.2\t2 0.9 0.4/>'

    formatter = LegacyPointFormatting(pointing_format="compact_v2")
    assert (formatter.format_image_points(np.array([[0.99, 0.8]]), 1.0, "label") ==
            'Counting the <points label\t-\t1 990 800/> shows a total of 1.')


def test_legacy_object_track():
    formatter = LegacyPointFormatting()
    frame_data = [
        {'frame': 0, 'time': '00:00.00', 'points': {0: {'point': [259.1968107876712, 339.93107876712327], 'occluded': False}}},
        {'frame': 6, 'time': '00:01.00', 'points': {0: {'point': [138.28081145415143, 426.57192596013783], 'occluded': False}}},
        {'frame': 12, 'time': '00:02.00', 'points': {0: {'point': [248.41219319816267, 334.89588456363384], 'occluded': False}}}
    ]
    out = formatter.format_video_tracks(frame_data, 1000, "debug", None)
    assert out == """
time 0.00
{0: [25.9, 34.0]}
time 1.00
{0: [13.8, 42.7]}
time 2.00
{0: [24.8, 33.5]}
""".strip()


def test_legacy_extract_points():
    points = np.array(LegacyPointFormatting.extract_multi_image_points(
        ' The "Fortnite characters shoot guns" are located at: <points 13.5\t1 44.4 69.9/> <points 14.0\t2 44.5 75.2/> <points 17.5\t3 45.1 71.2/> <points 18.0\t4 43.4 78.8/> <points 20.5\t5 45.1 70.2/> <points 56.0\t6 48.6 69.7/>.\nCounting shows the total number is: 6.',
        400, 640
    ))
    expected = np.array([
        [ 13.5,  177.6,  447.36],
        [ 14. ,  178. ,  481.28],
        [ 17.5,  180.4,  455.68],
        [ 18. ,  173.6,  504.32],
        [ 20.5,  180.4,  449.28],
        [ 56. ,  194.4,  446.08],
    ])
    assert np.allclose(points, expected, atol=0.1, rtol=0)

    points = np.array(LegacyPointFormatting.extract_points(
        ' <points x1="23.0" y1="17.7" x2="38.2" y2="24.1" x3="43.5" y3="5.0" x4="44.6" y4="13.8" x5="54.9" y5="20.4" x6="59.6" y6="7.7" x7="67.9" y7="12.3" x8="75.4" y8="25.0" x9="78.5" y9="7.7" x10="83.8" y10="20.3" x11="87.4" y11="12.5" x12="97.3" y12="17.3" alt="purple plates">purple plates</points>',
        360, 640
    ))
    expected = np.array([
        [82.8, 113.28],
        [137.52, 154.24],
        [156.6,  32.  ],
        [160.56, 88.32],
        [197.64, 130.56],
        [214.56, 49.28],
        [244.44, 78.72],
        [271.44, 160. ],
        [282.6,  49.28],
        [301.68, 129.92],
        [314.64, 80.  ],
        [350.28, 110.72],
    ])
    assert np.allclose(points, expected, atol=0.1, rtol=0)


def test_html1_image():
    formatter = UnifiedPointFormatter.build_for_format("html-v1")
    assert formatter.format_image_points([], 1.0, "point", "label") == "There are none."

    assert (formatter.format_image_points([[0.99, 0.8]], 1.0, "label") ==
            'Counting the <points coords="1 1 990 800">label</points> shows a total of 1.')
    assert (formatter.format_image_points([[0.911, 0.8], [0.9, 0.4]], 1.0, "aasdf") ==
            'Counting the <points coords="1 1 900 400 2 911 800">aasdf</points> shows a total of 2.')
    assert (formatter.format_image_points([[0.9, 0.8], [0.9, 0.7]], 1.0, "tmp") ==
            'Counting the <points coords="1 1 900 700 2 900 800">tmp</points> shows a total of 2.')
    assert (formatter.format_image_points([[0.90001, 0.8], [0.9, 0.7]], 1.0, "tmp") ==
            'Counting the <points coords="1 1 900 700 2 900 800">tmp</points> shows a total of 2.')


def test_html1_video():
    formatter = UnifiedPointFormatter.build_for_format("html-v1")
    actual = formatter.format_video_points(
        [0.1],
        [[[0.911, 0.8], [0.9, 0.4]]], 1.0, "aasdf"
    )
    assert actual == 'Counting the <points coords="0.1 1 900 400 2 911 800">aasdf</points> shows a total of 2.'

    actual = formatter.format_video_points(
        [0.1, 0.2],
        [[[0.911, 0.8]], [[0.9, 0.4]]], 1.0, "aasdf"
    )
    assert actual == 'Counting the <points coords="0.1 1 911 800\t0.2 2 900 400">aasdf</points> shows a total of 2.'


def test_html1_single_object_track():
    formatter = UnifiedPointFormatter.build_for_format("html-v1")
    frame_data = [
        {'frame': 0, 'time': '00:00.00', 'points': {0: {'point': [259.1968107876712, 339.93107876712327], 'occluded': False}}},
        {'frame': 6, 'time': '00:01.00', 'points': {0: {'point': [138.28081145415143, 426.57192596013783], 'occluded': False}}},
        {'frame': 12, 'time': '00:02.00', 'points': {0: {'point': [248.41219319816267, 334.89588456363384], 'occluded': False}}}
    ]
    out = formatter.format_video_tracks(frame_data, 1000, "debug")
    assert out == '<tracks coords="0.0 1 259 340\t1.0 1 138 427\t2.0 1 248 335">debug</tracks>'

    frame_data = [
        {'frame': 0, 'time': '00:00.00', 'points': {0: {'point': [259.1968107876712, 339.93107876712327], 'occluded': True}}},
        {'frame': 6, 'time': '00:01.00', 'points': {0: {'point': [138.28081145415143, 426.57192596013783], 'occluded': False}}},
        {'frame': 12, 'time': '00:02.00', 'points': {0: {'point': [248.41219319816267, 334.89588456363384], 'occluded': False}}}
    ]
    out = formatter.format_video_tracks(frame_data, 1000, "debug")
    assert out == '<tracks coords="1.0 1 138 427\t2.0 1 248 335">debug</tracks>'

    frame_data = [
        {'frame': 0, 'time': '00:00.00', 'points': {0: {'point': [259.1968107876712, 339.93107876712327], 'occluded': False}}},
        {'frame': 6, 'time': '00:01.00', 'points': {0: {'point': [138.28081145415143, 426.57192596013783], 'occluded': False}}},
        {'frame': 12, 'time': '00:02.00', 'points': {0: {'point': [248.41219319816267, 334.89588456363384], 'occluded': False}}}
    ]
    out = formatter.format_video_tracks(frame_data, 1000, "debug", start_end_only=True)
    assert out == '<tracks coords="0.0 1 259 340\t2.0 1 248 335">debug</tracks>'


def test_track_sorting_small():
    formatter = UnifiedPointFormatter.build_for_format("html-v1")
    frame_data = [
        {'frame': 0, 'time': '00:00.00', 'points': {1: {'point': [10, 30], 'occluded': False}, 0: {'point': [10.0000001, 20], 'occluded': False}}},
        {'frame': 1, 'time': '00:05.00', 'points': {0: {'point': [100, 100], 'occluded': False}, 1: {'point': [200, 200], 'occluded': False}}},
    ]
    out = formatter.format_video_tracks(frame_data, 1000, "debug")
    assert out == '<tracks coords="0.0 1 010 020 2 010 030\t5.0 1 100 100 2 200 200">debug</tracks>'


def test_track_sorting_chaos():
    """
    Chaotic scenario:
    - Frame 0: 3 points appear
    - Frame 1: 2 new points appear, one existing disappears
    - Frame 2: disappeared point returns, positions scrambled, new point in middle
    - Frame 3: everything swaps around
    """
    frame_data = [
        {'frame': 0, 'time': '00:00.00', 'points': {
            '3': {'point': [500, 200], 'occluded': False},
            '2': {'point': [200, 800], 'occluded': False},
            '1': {'point': [200, 100], 'occluded': False},  # same X as dog, smaller Y
        }},
        # XY sort: bird(200,100), dog(200,800), cat(500,200) -> indices 1, 2, 3

        {'frame': 1, 'time': '00:01.00', 'points': {
            '3': {'point': [50, 50], 'occluded': False},      # moved far left
            # dog disappears
            '1': {'point': [900, 900], 'occluded': False},   # moved far right
            '5': {'point': [100, 100], 'occluded': False},   # new
            '4': {'point': [60, 60], 'occluded': False}, # new, between cat and fish
        }},
        # New points XY sort: fish(100,100), elephant(60,60) -> elephant first -> indices 4, 5
        # Wait no: elephant(60,60) < fish(100,100) so elephant=4, fish=5
        # Output XY: cat(50,50)=3, elephant(60,60)=4, fish(100,100)=5, bird(900,900)=1

        {'frame': 2, 'time': '00:02.00', 'points': {
            '3': {'point': [999, 999], 'occluded': False},     # far right now
            '2': {'point': [1, 1], 'occluded': False},         # returns, far left!
            '1': {'point': [500, 500], 'occluded': False},    # middle
            '5': {'point': [400, 400], 'occluded': False},
            '4': {'point': [450, 450], 'occluded': False},
            '6': {'point': [425, 425], 'occluded': False},   # new, squeezed in middle
        }},
        # zebra is new -> index 6
        # Output XY: dog(1,1)=2, fish(400,400)=5, zebra(425,425)=6, elephant(450,450)=4, bird(500,500)=1, cat(999,999)=3
    ]


    # Frame 0: bird=1, dog=2, cat=3 (XY sorted)
    # Frame 1: cat=3, elephant=4, fish=5, bird=1 (output XY sorted, indices stable)
    # Frame 2: dog=2, fish=5, zebra=6, elephant=4, bird=1, cat=3

    expected = '<tracks coords="' \
                  '0.0 1 200 100 2 200 800 3 500 200\t' \
                  '1.0 1 900 900 3 050 050 4 060 060 5 100 100\t' \
                  '2.0 1 500 500 2 001 001 3 999 999 4 450 450 5 400 400 6 425 425' \
                  '">debug</tracks>'
    actual = UnifiedPointFormatter.build_for_format("html-v1").format_video_tracks(frame_data, 1000, "debug")
    assert actual == expected

    expected_v3 = '<tracks coords="' \
               '0.0 1 200 100 2 200 800 3 500 200;' \
               '1.0 3 050 050 4 060 060 5 100 100 1 900 900;' \
               '2.0 2 001 001 5 400 400 6 425 425 4 450 450 1 500 500 3 999 999' \
               '">debug</tracks>'
    actual = UnifiedPointFormatter(image_sep=";", sort_by_object_id=False).format_video_tracks(frame_data, 1000, "debug")
    assert actual == expected_v3


def test_html1_multi_object_track():
    formatter = UnifiedPointFormatter.build_for_format("html-v1")
    frame_data = [
        {'frame': 0, 'time': 1, 'points': {}},
        {'frame': 3, 'time': 2, 'points': {0: {'point': [300, 300]}}},
        {'frame': 6, 'time': 3, 'points': {0: {'point': [300, 300]}, 1: {'point': [400, 400]}}},
        {'frame': 9, 'time': 5, 'points': {}},
        {'frame': 12, 'time': 6, 'points': {2: {'point': [400, 100]}}},
        {'frame': 15, 'time': 7, 'points': {0: {'point': [300, 300]}, 1: {'point': [400, 400]}, 2: {'point': [400, 400]}}},
    ]
    out = formatter.format_video_tracks(frame_data, 1000, "debug")
    assert out == '<tracks coords="2.0 1 300 300	3.0 1 300 300 2 400 400	6.0 3 400 100	7.0 1 300 300 2 400 400 3 400 400">debug</tracks>'


def test_extract_points():
    points = UnifiedPointFormatter().extract_points(
        '<points coords="1 1 900 700 2 900 800">tmp</points>',
        1000, 1000)
    assert points == [(900, 700), (900, 800)]

    points = UnifiedPointFormatter().extract_points(
        '<points coords="1 1 900 700 2 900 800">tmp</points>',
        100, 50)
    assert points == [(90, 35), (90, 40)]

    points = UnifiedPointFormatter().extract_points(
        '<points label="tmp" x1=\"0\" y1=\"1\"/>', 1000, 1000)
    assert points == []

    points = UnifiedPointFormatter().extract_points(
        '"1 900 700 2 900 800"', 1000, 1000)
    assert points == []


def test_extract_multi_image_points():
    points = UnifiedPointFormatter().extract_multi_image_points(
        'Counting the <points label="aasdf" coords="0.1 1 900 400 2 911 800"/> shows a total of 2.',
        1000, 1000
    )
    assert points == [(0.1, 900, 400), (0.1, 911, 800)]

    points = UnifiedPointFormatter().extract_multi_image_points(
        'Counting the <points label="aasdf" coords="0.1 1 900 400 2 911 800"/> shows a total of 2.',
        100, 100
    )
    assert np.allclose(points, [(0.1, 90.0, 40.0), (0.1, 91.1, 80.0)])

    points = UnifiedPointFormatter().extract_multi_image_points(
        '<points label="aasdf" coords="0.1 1 900 400 2 911 800\t0.2 3 911 600"/>',
        1000, 1000
    )
    assert points == [(0.1, 900, 400), (0.1, 911, 800), (0.2, 911, 600)]

    points = UnifiedPointFormatter().extract_multi_image_points(
        '<points coords="0.1 1 900 400 2 911 800\t0.2 3 911 600">aasdf</points>',
        1000, 1000
    )
    assert points == [(0.1, 900, 400), (0.1, 911, 800), (0.2, 911, 600)]

    points = UnifiedPointFormatter().extract_multi_image_points(
        '<points coords="0.1 1 900 400 2 911 800;0.2 3 911 600">aasdf</points>',
        1000, 1000
    )
    assert points == [(0.1, 900, 400), (0.1, 911, 800), (0.2, 911, 600)]


def test_extract_trajectories():
    data = UnifiedPointFormatter().extract_tracks(
        '<tracks label="debug" coords="2.0 1 300 300\t3.0 1 300 300 2 400 400\t6.0 3 400 100\t8.0 1 300 300 2 400 400 3 400 400"/>',
        1000, 1000, video_fps=10
    )
    assert data == [
        {'time': 2.0, 'frame': 20, 'points': {'1': {'point': [300.0, 300.0]}}},
        {'time': 3.0, 'frame': 30, 'points': {'1': {'point': [300.0, 300.0]}, '2': {'point': [400.0, 400.0]}}},
        {'time': 6.0, 'frame': 60, 'points': {'3': {'point': [400.0, 100.0]}}},
        {'time': 8.0, 'frame': 80, 'points': {'1': {'point': [300.0, 300.0]}, '2': {'point': [400.0, 400.0]}, '3': {'point': [400.0, 400.0]}}}
    ]

