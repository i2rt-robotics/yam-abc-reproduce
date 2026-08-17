from transformers import AutoTokenizer

from olmo.preprocessing.data_formatter import DataFormatter


def _test_template(messages, formatter, tokenizer):
    message_json = [
        {"role": "user" if ix % 2 == 0 else "assistant", "content": msg} for ix, msg in enumerate(messages)
    ]

    # our template has no think tokens
    expected = tokenizer.apply_chat_template(message_json, tokenize=False, add_generation_prompt=False)
    expected = expected.replace("<think>\n\n</think>\n\n", "")

    # Have to reformat a bit since Qwen3 leaves some of the formatting text to the model
    # to generate
    actual = "".join(formatter.format_messages(messages))
    if len(messages) % 2 == 1:
        assert actual.endswith("<|im_end|>\n<|im_start|>assistant\n")
        assert actual[:-len("<|im_start|>assistant\n")] == expected
    else:
        assert expected.endswith("<|im_end|>\n")
        expected = expected[:-len("<|im_end|>\n")]
        assert expected == actual


def test_chat_template():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    formatter = DataFormatter(message_format="qwen3", always_start_with_space=False)
    for messages in [
        ["What is this?"],
        ["What is this?", "A cat"],
        [f"Number {i}" for i in range(4)],
        [f"Number {i}" for i in range(5)]
    ]:
        _test_template(messages, formatter, tok)
