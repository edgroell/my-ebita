# AnalysisResult class for storing the results of the analysis

from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional

@dataclass
class AnalysisResult:
    summary: str = ""
    concise_rationale: str = ""
    overall_sentiment: str = ""
    management_confidence_score: int = 0
    evasiveness_score_q_a: int = 0
    key_topics: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    model_used: str = ""
    success: bool = True
    error: Optional[str] = None

    # timing fields (milliseconds)
    request_ms: Optional[float] = None
    parse_ms: Optional[float] = None
    total_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], model_used: str = "") -> "AnalysisResult":
        def _int(val: Any) -> int:
            try:
                return int(val)
            except Exception:
                return 0

        return cls(
            summary=data.get("summary", "") or "",
            concise_rationale=data.get("concise_rationale", "") or "",
            overall_sentiment=data.get("overall_sentiment", "") or "",
            management_confidence_score=_int(data.get("management_confidence_score")),
            evasiveness_score_q_a=_int(data.get("evasiveness_score_q_a")),
            key_topics=list(data.get("key_topics") or []),
            red_flags=list(data.get("red_flags") or []),
            model_used=model_used,
            success=True,
            error=None,
        )
