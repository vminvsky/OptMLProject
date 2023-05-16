from typing import Callable, Dict, List, Tuple


def parse_output(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    labels = []
    for i, item in enumerate(input_list):
        item = item.lstrip(
            "0123456789. "
        )  # remove enumeration and any leading whitespace
        if (len(item)>3) & (len(item) <= 10):
            label = item.replace(":","")
        else:
            if item:  # skip empty items
                output_list.append(item)
                labels.append(label)
    return labels, output_list