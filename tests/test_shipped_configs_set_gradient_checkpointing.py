"""#1862: shipped grpo, ipo, bco, reward_model and online_dpo configs set gradient checkpointing.

Since #1778 these five trainers pass ``training.gradient_checkpointing`` to TRL. With
the key unset they follow the schema default ``false``, where TRL's own config default
was ``true``, so a recipe or template that leaves it out silently trains with more
activation memory than it did before #1778. Every shipped config for these tasks sets
the key explicitly.
"""

from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TEMPLATES
from soup_cli.recipes.catalog import RECIPES
from soup_cli.templates import list_templates, load_template

_TASKS = {"grpo", "ipo", "bco", "reward_model", "online_dpo"}
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "configs"


def _shipped():
    sources = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    sources += [(f"template:{name}", load_template(name)) for name in list_templates()]
    sources += [(f"inline:{name}", body) for name, body in TEMPLATES.items()]
    sources += [
        (f"example:{path.name}", path.read_text(encoding="utf-8"))
        for path in sorted(_EXAMPLES.glob("*.yaml"))
    ]
    return [
        (label, body)
        for label, body in sources
        if (yaml.safe_load(body) or {}).get("task") in _TASKS
    ]


_SHIPPED = _shipped()
_IDS = [label for label, _ in _SHIPPED]


def test_the_walk_reaches_every_task():
    """Guards the parametrised tests below against passing on an empty walk."""
    found = {yaml.safe_load(body)["task"] for _, body in _SHIPPED}
    assert found == _TASKS


@pytest.mark.parametrize("label,body", _SHIPPED, ids=_IDS)
def test_gradient_checkpointing_is_set(label, body):
    training = yaml.safe_load(body).get("training") or {}
    assert "gradient_checkpointing" in training, f"{label} leaves it unset"


@pytest.mark.parametrize("label,body", _SHIPPED, ids=_IDS)
def test_config_still_loads(label, body):
    load_config_from_string(body)
