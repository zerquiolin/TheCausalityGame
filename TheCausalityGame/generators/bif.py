"""Discrete BIF conversion and migration of bundled legacy SCM descriptions."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np

from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
)

ALIASES = {
    "BayesianNetworkSCMNode": "TheCausalityGame.scm.nodes.bn:BayesianNetworkSCMNode",
    "EquationBasedNumericalSCMNode": "TheCausalityGame.scm.nodes.sympy:EquationBasedNumericalSCMNode",
    "EquationBasedCategoricalSCMNode": "TheCausalityGame.scm.nodes.sympy:EquationBasedCategoricalSCMNode",
    "UniformNoiseDistribution": "TheCausalityGame.scm.noise.uniform:UniformNoiseDistribution",
    "GaussianNoiseDistribution": "TheCausalityGame.scm.noise.gaussian:GaussianNoiseDistribution",
    "NoNoiseDistribution": "TheCausalityGame.scm.noise.no_noise:NoNoiseDistribution",
    "DiracNoiseDistribution": "TheCausalityGame.scm.noise.dirac:DiracNoiseDistribution",
}


def migrate_legacy_scm(
    data: dict, *, seed: int = 911, access: dict[str, str] | None = None
) -> SCMSpec:
    """Convert a legacy vars/edges asset without changing its declared mechanisms."""
    data = json.loads(json.dumps(data))
    nodes = []
    for node in data["vars"]:
        for old, new in (("class", "class_"), ("values", "domain")):
            if old in node and new not in node:
                node[new] = node.pop(old)
        if node.get("class_") in ALIASES:
            node["class_"] = ALIASES[node["class_"]]
        noise = node.get("noise_distribution")
        if noise:
            noise["class_"] = ALIASES.get(
                noise.pop("class", noise.get("class_")), noise.get("class_")
            )
            if not noise.get("params"):
                noise["params"] = {k: v for k, v in noise.items() if k != "class_"}
        node["accessibility"] = (access or {}).get(
            node["name"], node.get("accessibility", "controllable")
        )
        if node["accessibility"] == "observable":
            node["accessibility"] = "measurable"
        probabilities = node.get("probability_distribution")
        if isinstance(probabilities, list):
            node["probability_distribution"] = (
                np.asarray(probabilities).reshape(-1).tolist()
            )
        elif isinstance(probabilities, dict):
            node["probability_distribution"] = {
                key: np.asarray(value).reshape(-1).tolist()
                for key, value in probabilities.items()
            }
        nodes.append(node)
    dag = data.get("dag") or {
        "class_": "TheCausalityGame.scm.dag.core:CoreDAG",
        "edges": data.get("edges", []),
        "nodes": [node["name"] for node in nodes],
    }
    spec = SCMSpec.model_validate(
        {
            "class_": data.get("class_", "TheCausalityGame.scm.core:CoreSCM"),
            "vars": nodes,
            "dag": dag,
            "random_state": data.get("random_state")
            or random_state_to_json(np.random.RandomState(seed)),
        }
    )
    build_from_spec(spec)
    return spec


def _tokens(text: str) -> list[str]:
    return [
        json.loads(token) if token.startswith('"') else token
        for token in re.findall(r'"(?:[^"\\]|\\.)*"|[^\s,|(){};]+', text)
    ]


def _blocks(text: str, kind: str) -> list[tuple[str, str]]:
    blocks = []
    pattern = re.compile(r"\b" + kind + r"\s+([^{}]+)\{")
    position = 0
    while match := pattern.search(text, position):
        depth, end, quoted, escaped = 1, match.end(), False, False
        while end < len(text) and depth:
            ch = text[end]
            if ch == '"' and not escaped:
                quoted = not quoted
            if not quoted:
                depth += int(ch == "{") - int(ch == "}")
            escaped = ch == "\\" and not escaped
            end += 1
        if depth:
            raise ValueError("Unterminated BIF block.")
        blocks.append((match.group(1).strip(), text[match.end() : end - 1]))
        position = end
    return blocks


def load_bif(
    path: str | Path, *, seed: int = 911, access: dict[str, str] | None = None
) -> SCMSpec:
    """Read discrete BIF table/explicit-CPT syntax with declared parent ordering.

    Continuous nodes, default rows and incomplete CPTs fail explicitly. Source
    path/hash can be retained in a catalog manifest; this function returns the SCM.
    """
    text = Path(path).read_text()
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    domains = {}
    for header, body in _blocks(text, "variable"):
        names = _tokens(header)
        match = re.search(r"type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{([^{}]*)\}", body)
        if len(names) != 1 or not match or names[0] in domains:
            raise ValueError("Only unique discrete BIF variables are supported.")
        categories = _tokens(match.group(2))
        if len(categories) != int(match.group(1)) or len(set(categories)) != len(
            categories
        ):
            raise ValueError("BIF category count/uniqueness mismatch.")
        domains[names[0]] = categories
    if not domains:
        raise ValueError("BIF contains no discrete variables.")
    nodes, edges, seen = [], [], set()
    for header, body in _blocks(text, "probability"):
        labels = _tokens(header)
        name, parents = labels[0], labels[1:]
        if (
            name not in domains
            or name in seen
            or any(p not in domains for p in parents)
        ):
            raise ValueError("Unknown/duplicate BIF probability variable.")
        seen.add(name)
        combinations = list(itertools.product(*(domains[p] for p in parents)))
        table = re.search(r"\btable\s+([^;]+);", body)
        rows = {}
        if table:
            flat = [float(value) for value in _tokens(table.group(1))]
            if len(flat) != len(combinations) * len(domains[name]):
                raise ValueError("BIF flat CPT size mismatch.")
            array = np.asarray(flat).reshape(len(combinations), len(domains[name]))
            rows = {
                ",".join(combo): values.tolist()
                for combo, values in zip(combinations, array, strict=True)
            }
        else:
            if re.search(r"\bdefault\b", body):
                raise ValueError(
                    "BIF default CPT rows are not supported; provide all rows."
                )
            for combo_text, values_text in re.findall(r"\(([^()]*)\)\s*([^;]+);", body):
                key = ",".join(_tokens(combo_text))
                if key in rows:
                    raise ValueError("Duplicate BIF CPT row.")
                rows[key] = [float(value) for value in _tokens(values_text)]
        if set(rows) != {",".join(combo) for combo in combinations}:
            raise ValueError("BIF CPT is incomplete or has unknown parent assignments.")
        nodes.append(
            {
                "class_": ALIASES["BayesianNetworkSCMNode"],
                "name": name,
                "domain": domains[name],
                "parents": parents or None,
                "accessibility": (access or {}).get(name, "controllable"),
                "probability_distribution": rows if parents else rows[""],
            }
        )
        edges.extend((parent, name) for parent in parents)
    if seen != set(domains):
        raise ValueError("BIF needs one probability block per variable.")
    spec = SCMSpec.model_validate(
        {
            "class_": "TheCausalityGame.scm.core:CoreSCM",
            "vars": nodes,
            "dag": {
                "class_": "TheCausalityGame.scm.dag.core:CoreDAG",
                "nodes": list(domains),
                "edges": edges,
            },
            "random_state": random_state_to_json(np.random.RandomState(seed)),
        }
    )
    build_from_spec(spec)
    return spec
