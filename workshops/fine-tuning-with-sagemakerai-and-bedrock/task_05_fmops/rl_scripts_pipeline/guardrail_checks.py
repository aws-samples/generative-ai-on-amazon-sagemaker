"""Helpers for turning Amazon Bedrock ApplyGuardrail responses into gate metrics.

Only boto3 is needed, so the same code runs in the notebook, in a processing job
and in the inference-time wrapper.
"""

from typing import Dict, List, Optional

# Policies that count as a violation of the deployment gate. Contextual grounding
# is reported separately because it needs a grounding source, which a non-RAG
# medical assistant does not always have.
GATE_POLICIES = ("contentPolicy", "topicPolicy", "wordPolicy", "sensitiveInformationPolicy")


def blocked_policies(response: Dict) -> List[str]:
    """Return the names of the policies that BLOCKED content in an ApplyGuardrail response."""
    found = set()

    def walk(node, policy: Optional[str]):
        if isinstance(node, dict):
            if node.get("action") == "BLOCKED" and policy:
                found.add(policy)
            for key, value in node.items():
                walk(value, key if key.endswith("Policy") else policy)
        elif isinstance(node, list):
            for value in node:
                walk(value, policy)

    for assessment in response.get("assessments", []):
        walk(assessment, None)
    return sorted(found)


def apply_guardrail(client, guardrail_id: str, guardrail_version: str, text: str, source: str,
                    grounding_source: Optional[str] = None, query: Optional[str] = None) -> Dict:
    """Call ApplyGuardrail and return a compact result.

    For source="OUTPUT" with a grounding source and query, the contextual grounding
    check runs as well. Without them only the other policies are evaluated.
    """
    if source == "OUTPUT" and grounding_source and query:
        content = [
            {"text": {"text": grounding_source, "qualifiers": ["grounding_source"]}},
            {"text": {"text": query, "qualifiers": ["query"]}},
            {"text": {"text": text, "qualifiers": ["guard_content"]}},
        ]
    else:
        content = [{"text": {"text": text}}]
    response = client.apply_guardrail(
        guardrailIdentifier=guardrail_id, guardrailVersion=guardrail_version, source=source, content=content
    )
    policies = blocked_policies(response)
    return {
        "intervened": response.get("action") == "GUARDRAIL_INTERVENED",
        "blocked_policies": policies,
        "violation": any(p in GATE_POLICIES for p in policies),
        "grounding_blocked": "contextualGroundingPolicy" in policies,
        "output_text": " ".join(o.get("text", "") for o in response.get("outputs", [])),
    }


def summarize(results: List[Dict]) -> Dict:
    n = len(results)
    counts: Dict[str, int] = {}
    for r in results:
        for p in r["blocked_policies"]:
            counts[p] = counts.get(p, 0) + 1
    return {
        "n": n,
        "violations": sum(r["violation"] for r in results),
        "violation_rate": (sum(r["violation"] for r in results) / n) if n else 0.0,
        "grounding_block_rate": (sum(r["grounding_blocked"] for r in results) / n) if n else 0.0,
        "blocked_by_policy": counts,
    }
