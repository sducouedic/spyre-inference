# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""collect.py: the server log lines it parses, and the Δ colour thresholds.

Run by hand: python -m pytest .claude/skills/vllm-bench-serve/scripts -o addopts=""
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402

# ruff: noqa: E501
# Two TP workers; each line as spyre_model_runner.py, spyre_attn.py and vLLM core.py log it.
SERVER_LOG = """\
INFO 10-13 10:00:00 vLLM API server version 0.0
(TP0) INFO 10-13 10:01:00 Recording 12 per-seq + 4 batched-decode attention variants for layer...
(TP0) INFO 10-13 10:02:00 Attention graph recording complete: 16 graphs in 50.000s.
(TP1) INFO 10-13 10:02:01 Attention graph recording complete: 16 graphs in 52.500s.
(TP0) INFO 10-13 10:05:00 Warmup complete in 180.000s for 6 buckets.
(TP1) INFO 10-13 10:05:01 Warmup complete in 181.000s for 6 buckets.
INFO 10-13 10:05:02 init engine (profile, create kv cache, warmup model) took 300.00 s (compilation: 240.00 s)
INFO 10-13 10:05:30 Application startup complete.
"""


def test_parse_server_log():
    assert collect.parse_server_log(SERVER_LOG) == {
        "startup_s": 330.0,
        "init_engine_s": 300.0,
        "compilation_s": 240.0,
        "model_warmup_s": 181.0,
        "buckets": 6,
        "attn_record_s": 52.5,
        "attn_graphs": 16,
        "attn_split": "12 + 4",
    }


def test_delta_classes():
    assert collect.delta(100, 101, "lower")[1] == ""
    assert collect.delta(100, 90, "lower")[1] == "d-good"
    assert collect.delta(100, 105, "lower")[1] == "d-warn"
    assert collect.delta(100, 120, "lower")[1] == "d-bad"
    assert collect.delta(100, 80, "higher")[1] == "d-bad"
    assert collect.delta(4, 4, "count") == ("", "", None)
