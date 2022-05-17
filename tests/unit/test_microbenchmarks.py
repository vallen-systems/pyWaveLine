import re
from collections import defaultdict

import pytest

multiline_output = [
    b"acq_enabled=1",
    b"log_enabled=0",
    b"adc2uv=1.74",
    b"cct=-0.5 s",
    b"filter=10.5-350 kHz, order 4",
    b"cont=0",
    b"thr=3162.5 uV",
    b"ddt=250  us",
    b"status_interval=1000 ms",
    b"tr_enabled=1",
    b"dummy empty line",
    b"tr_decimation=2",
    b"tr_pre_trig=100",
    b"tr_post_dur=100",
    b"tr_max_samples=2097152",
]


def multiline_output_to_dict_orig(lines):
    return defaultdict( str, [
        (lambda k, v="": [k.strip(), v.strip()])
            (*line.decode().split("=", maxsplit=1))
        for line in lines
    ])


def multiline_output_to_dict_partition(lines):
    def line_to_key_value(line):
        k, _, v = line.partition("=")
        return k.strip(), v.strip()

    return defaultdict(str, [line_to_key_value(line.decode()) for line in lines])


def multiline_output_to_dict_split(lines):
    def line_to_key_value(line):
        k, *v = line.split("=", maxsplit=1)
        return k.strip(), v[0].strip() if v else ""

    return defaultdict(str, [line_to_key_value(line.decode()) for line in lines])


def multiline_output_to_dict_part_no_func(lines):
    return defaultdict(str, [
        (k.strip(), v.strip()) for k,_,v in 
        [line.decode().partition("=") for line in lines]
    ])

def multiline_output_to_dict_part_simple_no_strip(lines):
    #very simple, but no strip...
    return defaultdict(str, [line.decode().partition("=")[0:3:2] for line in lines ])

ml_kv_pattern1 = re.compile(r"\s*(\S+)\s*(?:=\s*(.*)\s*)?")
ml_kv_pattern2 = re.compile(r"\s*(\w+)\s*(?:=\s*(.*)\s*)?")

def multiline_output_to_dict_regex1(lines):
    return defaultdict(str, [ml_kv_pattern1.match(line.decode()).groups() for line in lines ])

def multiline_output_to_dict_regex2(lines):
    return defaultdict(str, [ml_kv_pattern2.match(line.decode()).groups() for line in lines ])

 
@pytest.mark.benchmark(
    group="multiline_output",
    #min_time=0.1,
    #max_time=0.5,
    #min_rounds=5,
    #timer=time.time,
    #disable_gc=True,
    #warmup=True,
)
@pytest.mark.parametrize("function", [
    multiline_output_to_dict_orig,
    multiline_output_to_dict_partition,
    multiline_output_to_dict_split,
    multiline_output_to_dict_part_no_func,
    multiline_output_to_dict_part_simple_no_strip,
    multiline_output_to_dict_regex1,
    multiline_output_to_dict_regex2,
    ])
def test_multiline_output(benchmark,function):
    benchmark(function, multiline_output)

#hit parser tests
hit_line = b"H temp=27 dummy T = 3044759 A=3557 R=24 D=819 C=31 E=518280026 TRAI=1 flags=0"
tr_line  = b"TRAI=1 dummy T = 3044759 NS=138"

line_parsers = [
    br"(\S+)\s*=\s*(\S+)",
    br"([^\s=]+)(?:\s*=\s*(\S+))?",  # accept words as keys w/o values?
    #\w seems faster than \S; not always ??!!
    br"(\w+)\s*=\s*(\S+)",
    br"([^\W=]+)(?:\s*=\s*(\S+))?",
    ]

@pytest.mark.benchmark(
    group="pattern_hit_line",
    #disable_gc=True,
    #warmup=True,
)
@pytest.mark.parametrize("pattern_str", line_parsers)
def test_pattern_hit(benchmark, pattern_str):
    #benchmark.group = "line_hit"
    p = re.compile(pattern_str)
    d = benchmark(lambda : defaultdict(int, p.findall(hit_line)))
    assert int(d[b"temp"]) == 27
    assert int(d[b"T"]) == 3044759

@pytest.mark.benchmark(
    group="pattern_tr_line",
    #disable_gc=True,
    #warmup=True,
)
@pytest.mark.parametrize("pattern_str", line_parsers)
def test_pattern_tr(benchmark, pattern_str):
    p = re.compile(pattern_str)
    d = benchmark(lambda : defaultdict(int, p.findall(tr_line)))
    assert int(d[b"TRAI"]) == 1
    assert int(d[b"T"]) == 3044759
