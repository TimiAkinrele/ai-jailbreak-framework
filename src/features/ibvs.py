import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

# -------------------------
# 0) Normalisation helpers
# -------------------------
ZERO_WIDTH = "".join(["\u200b", "\u200c", "\u200d", "\ufeff"])


def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKC", t).lower()
    t = t.translate({ord(c): None for c in ZERO_WIDTH})
    t = re.sub(r"\s+", " ", t).strip()
    return t


def looks_encoded(t: str) -> bool:
    # Cheap signals for base64 / hex blobs / encoded payload artifacts.
    if re.search(r"\b(?:[a-f0-9]{32,})\b", t):  # long hex
        return True
    if re.search(r"\b(?:[a-z0-9+/]{40,}={0,2})\b", t):  # base64-ish
        return True
    if re.search(r"\b(?:rot13|uuencode|obfuscat(?:e|ed|ion))\b", t):
        return True
    return False


# -------------------------
# 1) Pattern libraries
# -------------------------
# NOTE:
# - Emphasise high-precision boundary override cues.
# - Keep generic instructional cues weak and mostly interaction-driven.

P_HIERARCHY_OVERRIDE = [
    ("ignore_prior_instructions", r"\bignore\b.{0,80}\b(previous|prior|above)\b.{0,80}\b(instructions?|rules?|polic(y|ies))\b"),
    ("forget_prior_instructions", r"\b(forget|drop)\b.{0,60}\b(previous|prior|earlier|above)\b.{0,40}\b(instructions?|rules?|polic(y|ies))\b"),
    # German override phrasing common in prompt-injection benchmarks.
    ("ignore_all_instructions_de", r"\b(?:ignoriere|ignorieren sie)\b.{0,48}\b(?:alle|die obigen|bisherigen|vorangehenden)\b.{0,48}\b(?:anweisungen|befehle|auftra(?:ge|ege))\b"),
    ("forget_all_instructions_de", r"\b(?:vergiss|vergessen sie)\b.{0,48}\b(?:alles|alle|vorherigen|bisherigen|davor)\b.{0,48}\b(?:anweisungen|befehle|auftra(?:ge|ege))?\b"),
    ("forget_everything_de", r"\bvergiss\b.{0,24}\b(?:alles|alles davor|alles davor\\.)\b"),
    ("ignore_above_de", r"\bdie obigen\b.{0,24}\b(?:ausf(?:u|ü)hrungen|anweisungen|befehle)\b.{0,24}\bignorier"),
    ("disregard_policy", r"\bdisregard\b.{0,80}\b(instructions?|rules?|polic(y|ies))\b"),
    ("do_not_follow_policy", r"\b(do not|don't|never)\b.{0,50}\b(follow|obey)\b.{0,40}\b(policy|rules?|safety|guardrails?)\b"),
    ("override_safety", r"\boverride\b.{0,60}\b(safety|guardrails?|polic(y|ies)|rules?)\b"),
    ("bypass_restrictions", r"\b(bypass|circumvent)\b.{0,60}\b(safety|filters?|restrictions?|guardrails?)\b"),
    ("priority_override", r"\b(highest|top|new)\b.{0,40}\bpriority\b.{0,40}\b(instructions?|rules?|policy|system)\b"),
    ("follow_my_instructions", r"\bfollow\b.{0,32}\b(my|these)\b.{0,32}\b(instructions?|rules?)\b.{0,32}\b(instead|only)\b"),
    ("no_restrictions", r"\b(no|without)\b.{0,24}\b(restrictions?|guardrails?|polic(y|ies))\b|\bunfiltered\b"),
]

P_ROLE_REDEFINE = [
    ("you_are_now", r"\byou are now\b"),
    ("you_are_now_de", r"\b(?:du bist|sie sind)\b.{0,16}\b(?:jetzt|nun)\b"),
    ("now_you_are_de", r"\b(?:jetzt|nun)\b.{0,12}\bbist du\b"),
    ("from_now_on_role", r"\bfrom now on\b.{0,60}\byou\b.{0,40}\b(are|will|must)\b"),
    ("act_as", r"\b(pretend to be|act as|roleplay as)\b"),
]

P_SYSTEM_SPOOF = [
    ("system_tag", r"(?:^|[\s\[(<])(?:<\s*system\s*>|\[\s*system\s*\]|system\s*:)"),
    ("developer_tag", r"(?:^|[\s\[(<])(?:<\s*developer\s*>|\[\s*developer\s*\]|developer\s*:)"),
    ("assistant_tag", r"(?:^|[\s\[(<])(?:<\s*assistant\s*>|\[\s*assistant\s*\]|assistant\s*:)"),
    ("tool_tag", r"(?:^|[\s\[(<])(?:<\s*tool\s*>|\[\s*tool\s*\]|tool\s*:)"),
]

P_TOOL_DIRECTIVE_HIGH_RISK = [
    (
        "force_tool_to_bypass",
        r"\b(use|call|invoke|run|execute)\b.{0,40}\b(tool|function|plugin|browser|api|terminal|shell)\b.{0,50}\b(bypass|override|ignore|disable|evade)\b",
    ),
    ("run_hidden_commands", r"\b(run|execute)\b.{0,40}\b(hidden|silent|background)\b.{0,40}\b(command|script)\b"),
    ("exfiltrate_via_tool", r"\b(tool|api|browser|plugin)\b.{0,50}\b(exfiltrat|steal|dump|leak)\b"),
]

P_TOOL_DIRECTIVE_GENERIC = [
    ("call_tool", r"\bcall\b.{0,50}\b(tool|function|plugin|browser|api)\b"),
    ("execute_command", r"\b(execute|run)\b.{0,50}\b(command|script|shell|terminal)\b"),
    ("force_tool_use", r"\buse\b.{0,40}\b(browser|tool|plugin|api)\b.{0,40}\b(to|for)\b"),
]

# Generic procedural language is weak on its own; keep this low precision.
P_PROCEDURAL = [
    ("step_by_step", r"\bstep[- ]?by[- ]?step\b"),
    ("ordered_steps", r"\b(first|second|third|next|then|finally)\b"),
    ("exact_instructions", r"\b(exact|detailed)\b.{0,24}\b(steps?|instructions?)\b"),
    ("how_to_sequence", r"\bhow to\b.{0,24}\b(perform|execute|do)\b"),
]

P_HARM_DOMAINS = [
    ("malware", r"\b(malware|ransomware|keylogger|stealer)\b"),
    ("phishing_creds", r"\b(phishing|credential stuffing|steal credentials?)\b"),
    ("exploit_auth", r"\b(exploit|privilege escalation|bypass auth(?:entication)?)\b"),
    ("weapons", r"\b(make|build|assemble)\b.{0,40}\b(bomb|weapon|explosive|nerve agent)\b"),
]

P_EVASION = [
    ("encoding_request", r"\b(base64|hex|encode|decode)\b"),
    ("split_chars", r"\b(character by character|with spaces between letters|insert spaces)\b"),
    ("leet_obfuscation", r"\b(leetspeak|0verride|ign0re)\b"),
    ("avoid_detection", r"\b(avoid detection|evade filters?|bypass moderation)\b"),
]

# Benign analytical / educational framing cues used for suppression.
P_BENIGN_EDU_CONTEXT = [
    ("for_research", r"\b(for|as)\b.{0,20}\b(research|study|analysis)\b"),
    ("for_class", r"\b(for|as)\b.{0,20}\b(class|course|homework|assignment)\b"),
    ("defensive_context", r"\b(defensive|safety|secure coding|threat model)\b"),
    ("historical_explanation", r"\b(history of|historically|explain why)\b"),
]

# Meta-discussion of system/developer prompts often appears in benign
# analysis material and should not be treated as an attack by itself.
P_META_SYSTEM_DISCUSSION = [
    ("mentions_system_prompt", r"\b(system prompt|developer message|assistant message)\b"),
    ("explains_prompt_injection", r"\b(prompt injection|jailbreak detection|red[- ]teaming)\b"),
    ("quoted_role_tokens", r"(?:(?:`|\"|')\s*(?:system|developer|assistant)\s*(?:`|\"|'))"),
]

# Slightly stronger weight on high-precision boundary cues.
IBVS_V2_WEIGHTS: Dict[str, float] = {
    "hierarchy_override": 3.6,
    "role_redefine": 1.2,
    "system_spoof": 3.1,
    "tool_directive": 1.8,
    "procedural": 0.4,
    "harm_domain": 0.9,
    "evasion": 2.8,
    "interaction_hierarchy_system": 2.4,
    "interaction_system_hierarchy_spoof_chain": 2.6,
    "interaction_evasion_override": 2.0,
    "interaction_tool_system": 1.0,
    "interaction_harm_evasion": 2.2,
    "interaction_harm_procedural": 0.8,
    "high_specific_risk_anchor": 1.0,
    "benign_context_suppression": 1.2,
    "meta_system_discussion_suppression": 1.0,
}


@dataclass
class IBVS2Breakdown:
    hierarchy_override: float = 0.0
    role_redefine: float = 0.0
    system_spoof: float = 0.0
    tool_directive: float = 0.0
    procedural: float = 0.0
    harm_domain: float = 0.0
    evasion: float = 0.0
    interaction_hierarchy_system: float = 0.0
    interaction_system_hierarchy_spoof_chain: float = 0.0
    interaction_evasion_override: float = 0.0
    interaction_tool_system: float = 0.0
    interaction_harm_evasion: float = 0.0
    interaction_harm_procedural: float = 0.0
    high_specific_risk_anchor: float = 0.0
    benign_context_suppression: float = 0.0
    meta_system_discussion_suppression: float = 0.0
    length_penalty: float = 0.0
    tripwire_alert: float = 0.0


IBVS_V2_NUMERIC_COLUMNS: Sequence[str] = tuple(IBVS2Breakdown.__dataclass_fields__.keys())


def _count_hits(patterns: Sequence[Tuple[str, str]], text: str) -> Tuple[int, List[str]]:
    hits: List[str] = []
    for name, pattern in patterns:
        if re.search(pattern, text):
            hits.append(name)
    return len(hits), hits


def _has_system_hierarchy_spoof_chain(text: str, max_gap: int = 140) -> bool:
    spoof = r"(?:<\s*(?:system|developer)\s*>|\[\s*(?:system|developer)\s*\]|(?:system|developer)\s*:)"
    override = r"(?:ignore|disregard|override|bypass|do not follow|don't follow|forget|drop)"
    p1 = rf"{spoof}.{{0,{max_gap}}}{override}"
    p2 = rf"{override}.{{0,{max_gap}}}{spoof}"
    return bool(re.search(p1, text) or re.search(p2, text))


def _compute_ibvs_v2(text: str) -> Tuple[float, Dict[str, float], List[str]]:
    t = normalize_text(text)
    bd = IBVS2Breakdown()
    triggered: List[str] = []

    n_override, h_override = _count_hits(P_HIERARCHY_OVERRIDE, t)
    n_role, h_role = _count_hits(P_ROLE_REDEFINE, t)
    n_system, h_system = _count_hits(P_SYSTEM_SPOOF, t)
    n_tool_risky, h_tool_risky = _count_hits(P_TOOL_DIRECTIVE_HIGH_RISK, t)
    n_tool_generic, h_tool_generic = _count_hits(P_TOOL_DIRECTIVE_GENERIC, t)
    n_proc, h_proc = _count_hits(P_PROCEDURAL, t)
    n_harm, h_harm = _count_hits(P_HARM_DOMAINS, t)
    n_evasion, h_evasion = _count_hits(P_EVASION, t)
    n_benign_ctx, h_benign_ctx = _count_hits(P_BENIGN_EDU_CONTEXT, t)
    n_meta_disc, h_meta_disc = _count_hits(P_META_SYSTEM_DISCUSSION, t)

    triggered.extend([f"hierarchy::{h}" for h in h_override])
    triggered.extend([f"role::{h}" for h in h_role])
    triggered.extend([f"system::{h}" for h in h_system])
    triggered.extend([f"tool_risky::{h}" for h in h_tool_risky])
    triggered.extend([f"tool_generic::{h}" for h in h_tool_generic])
    triggered.extend([f"procedural::{h}" for h in h_proc])
    triggered.extend([f"harm::{h}" for h in h_harm])
    triggered.extend([f"evasion::{h}" for h in h_evasion])
    triggered.extend([f"context::{h}" for h in h_benign_ctx])
    triggered.extend([f"context::{h}" for h in h_meta_disc])

    bd.hierarchy_override = n_override * IBVS_V2_WEIGHTS["hierarchy_override"]
    bd.role_redefine = n_role * IBVS_V2_WEIGHTS["role_redefine"]
    bd.system_spoof = n_system * IBVS_V2_WEIGHTS["system_spoof"]
    has_boundary_anchor = n_override > 0 or n_system > 0
    has_risk_anchor = has_boundary_anchor or n_evasion > 0 or n_harm > 0
    effective_tool_hits = n_tool_risky + (n_tool_generic if has_risk_anchor else 0)
    if n_tool_generic > 0 and not has_risk_anchor:
        triggered.append("tool_generic::suppressed_unanchored")
    bd.tool_directive = effective_tool_hits * IBVS_V2_WEIGHTS["tool_directive"]
    bd.harm_domain = n_harm * IBVS_V2_WEIGHTS["harm_domain"]

    # Procedural signals are only trusted when anchored by boundary/evasion cues.
    anchored_procedural = n_proc > 0 and (has_boundary_anchor or (n_harm > 0 and n_evasion > 0))
    if n_proc > 0 and not anchored_procedural:
        triggered.append("procedural::suppressed_unanchored")
    bd.procedural = (
        min(n_proc, 2) * IBVS_V2_WEIGHTS["procedural"]
        if anchored_procedural
        else 0.0
    )

    bd.evasion = n_evasion * IBVS_V2_WEIGHTS["evasion"]
    if looks_encoded(t):
        bd.evasion += IBVS_V2_WEIGHTS["evasion"]
        triggered.append("evasion::encoded_payload")

    # Interaction features improve expressiveness while keeping interpretability.
    if n_override > 0 and n_system > 0:
        bd.interaction_hierarchy_system = IBVS_V2_WEIGHTS["interaction_hierarchy_system"]
    if n_override > 0 and n_system > 0 and _has_system_hierarchy_spoof_chain(t):
        bd.interaction_system_hierarchy_spoof_chain = IBVS_V2_WEIGHTS["interaction_system_hierarchy_spoof_chain"]
        triggered.append("interaction::system_hierarchy_spoof_chain")
    if n_evasion > 0 and (n_override > 0 or n_system > 0 or effective_tool_hits > 0):
        bd.interaction_evasion_override = IBVS_V2_WEIGHTS["interaction_evasion_override"]
    if effective_tool_hits > 0 and n_system > 0 and (n_override > 0 or n_evasion > 0):
        bd.interaction_tool_system = IBVS_V2_WEIGHTS["interaction_tool_system"]
    if n_harm > 0 and n_evasion > 0:
        bd.interaction_harm_evasion = IBVS_V2_WEIGHTS["interaction_harm_evasion"]
        triggered.append("interaction::harm_evasion")
    if n_harm > 0 and anchored_procedural and (n_evasion > 0 or n_tool_risky > 0):
        bd.interaction_harm_procedural = IBVS_V2_WEIGHTS["interaction_harm_procedural"]
        triggered.append("interaction::harm_procedural")

    has_high_specific_anchor = bool(
        bd.interaction_system_hierarchy_spoof_chain > 0.0
        or (n_override > 0 and n_system > 0)
        or bd.interaction_harm_evasion > 0.0
    )
    if has_high_specific_anchor:
        bd.high_specific_risk_anchor = IBVS_V2_WEIGHTS["high_specific_risk_anchor"]
        triggered.append("anchor::high_specific_risk")

    benign_context = n_benign_ctx > 0 and n_override == 0 and n_evasion == 0 and n_harm == 0
    if benign_context and (n_proc > 0 or n_tool_generic > 0 or n_system > 0):
        bd.benign_context_suppression = IBVS_V2_WEIGHTS["benign_context_suppression"]
        triggered.append("context::benign_educational_suppression")

    meta_system_discussion = (
        n_meta_disc > 0
        and n_override == 0
        and n_evasion == 0
        and n_harm == 0
        and n_tool_risky == 0
    )
    if meta_system_discussion and (n_system > 0 or n_tool_generic > 0 or n_proc > 0 or n_meta_disc > 0):
        bd.meta_system_discussion_suppression = IBVS_V2_WEIGHTS["meta_system_discussion_suppression"]
        triggered.append("context::meta_system_discussion_suppression")

    n_tokens = max(len(t.split()), 1)
    if n_tokens > 320:
        bd.length_penalty = 0.6
    elif n_tokens > 220:
        bd.length_penalty = 0.4

    raw = (
        bd.hierarchy_override
        + bd.role_redefine
        + bd.system_spoof
        + bd.tool_directive
        + bd.procedural
        + bd.harm_domain
        + bd.evasion
        + bd.interaction_hierarchy_system
        + bd.interaction_system_hierarchy_spoof_chain
        + bd.interaction_evasion_override
        + bd.interaction_tool_system
        + bd.interaction_harm_evasion
        + bd.interaction_harm_procedural
        + bd.high_specific_risk_anchor
        - bd.benign_context_suppression
        - bd.meta_system_discussion_suppression
        - bd.length_penalty
    )

    # Tripwire bit: strong explicit risk cues or multiple compounding cues.
    major_signals = (
        int(n_override > 0)
        + int(n_system > 0)
        + int(n_evasion > 0)
        + int(n_tool_risky > 0)
    )
    bd.tripwire_alert = float(
        (n_override > 0 and n_system > 0)
        or bd.interaction_system_hierarchy_spoof_chain > 0.0
        or (bd.interaction_harm_evasion > 0.0 and raw >= 5.0)
        or (major_signals >= 2 and raw >= 7.0)
        or raw >= 9.0
    )

    return float(raw), asdict(bd), sorted(set(triggered))


def ibvs_v2(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Backward-compatible API:
      returns (total_score, numeric_breakdown_dict)
    """
    score, breakdown, _ = _compute_ibvs_v2(text)
    return score, breakdown


def ibvs_v2_with_triggers(text: str) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Extended API for notebook analysis:
      returns (total_score, numeric_breakdown_dict, triggered_rule_labels)
    """
    return _compute_ibvs_v2(text)


def ibvs_v2_feature_dict(text: str, prefix: str = "ibvs2_") -> Dict[str, float]:
    """
    Produce model-ready structured features from IBVS v2.
    Includes all numeric component scores plus total score.
    """
    score, breakdown = ibvs_v2(text)
    out = {f"{prefix}{k}": float(v) for k, v in breakdown.items()}
    out[f"{prefix}total"] = float(score)
    return out
