"""Intent Processing (IntentProcessor): natural language -> structured intent (slice type + measurable expectations).

Slice type: the repo's fine-tuned DistilBERT (Context_Model/fine_tuned_slice_intent_model) when real weights
and transformers/torch are available; otherwise the keyword classifier of routes/slice_selection.py::predict_slice_type
(extended with Korean keywords). Note: the repo only holds a Git LFS pointer for model.safetensors.
Expectations: regex extraction of latency / bandwidth / reliability / device count, with per-slice defaults.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from common import config

log = logging.getLogger("intent_processor")

KEYWORDS = {
    "URLLC": ["latency", "reliable", "real-time", "realtime", "autonomous", "surgery", "robot", "control", "v2x",
              "drone", "haptic", "mission critical", "public safety", "저지연", "초저지연", "원격수술", "수술", "자율주행",
              "로봇", "제어", "드론", "실시간", "응급", "재난", "고신뢰"],
    "mMTC": ["iot", "sensor", "meter", "device", "smart city", "tracking", "massive", "telemetry", "센서", "미터",
             "사물인터넷", "스마트시티", "검침", "단말 다수", "대규모"],
    "eMBB": ["video", "stream", "gaming", "cloud", "broadband", "4k", "8k", "vr", "ar", "concert", "fan", "download",
             "영상", "스트리밍", "게임", "방송", "대용량", "다운로드", "관람객", "고화질"],
}
DEFAULTS = {
    "URLLC": [{"kpi": "latency_ms", "op": "<=", "value": 10, "percentile": 99},
              {"kpi": "reliability_percent", "op": ">=", "value": 99.999},
              {"kpi": "bandwidth_mbps", "op": ">=", "value": 20}],
    "eMBB": [{"kpi": "bandwidth_mbps", "op": ">=", "value": 100}, {"kpi": "latency_ms", "op": "<=", "value": 50}],
    "mMTC": [{"kpi": "bandwidth_mbps", "op": ">=", "value": 5}, {"kpi": "latency_ms", "op": "<=", "value": 200},
             {"kpi": "reliability_percent", "op": ">=", "value": 99.0}],
}
PRIORITY_WORDS = {1: ["emergency", "public safety", "disaster", "응급", "재난", "긴급"],
                  2: ["surgery", "mission critical", "원격수술", "수술"],
                  4: ["best effort", "background", "low priority", "저우선"]}


class IntentProcessor:
    def __init__(self, backend: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.semantic = None
        self.model_info = {"model": "keyword-classifier", "version": "1.0"}
        self.force_keyword = False        # set by the intent-NLU lifecycle when version 1.0 is primary
        self._try_load()
        backend = backend or config.INTENT_BACKEND
        if self.model is None and backend != "keyword":
            self._try_semantic(backend)
        # P3: declarative KPI constraint shapes + optional NLI zero-shot second opinion (informational only:
        # it failed its adoption rule on calib, see datasets/intent_nlu/eval_p3.md)
        self.validator = None
        try:
            from .constraints import ConstraintValidator
            self.validator = ConstraintValidator(config.INTENT_CONSTRAINTS)
        except Exception as e:  # pragma: no cover
            log.warning("constraint rules not loaded (%s)", e)
        self.nli = None
        if config.INTENT_NLI == "info" and (config.NLI_PATH / "onnx" / "model.onnx").exists():
            try:
                from .nli_check import NliZeroShot
                self.nli = NliZeroShot(config.NLI_PATH)
            except Exception as e:  # pragma: no cover
                log.warning("NLI second opinion not loaded (%s)", e)

    def _try_load(self) -> None:
        path = config.DISTILBERT_PATH
        st = path / "model.safetensors"
        if not st.exists() or st.stat().st_size < 1_000_000:
            log.info("DistilBERT weights not present (Git LFS pointer)")
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
            self.model_info = {"model": "distilbert-slice-intent", "version": "1.0"}
        except Exception as e:  # pragma: no cover
            log.warning("DistilBERT load failed (%s)", e)

    def _try_semantic(self, backend: str) -> None:
        """Semantic kNN over datasets/intent_nlu (train + references), calibrated by calibration.json."""
        from .semantic import SemanticIntentClassifier, load_jsonl, load_references, make_encoder
        d = config.INTENT_DATA_DIR
        try:
            enc = make_encoder(config.SBERT_PATH, "sbert" if backend == "semantic" else "auto")
            clf = SemanticIntentClassifier(enc).fit(load_jsonl(d / "train.jsonl"), load_references(d / "references.json"))
        except Exception as e:
            if backend == "semantic":
                raise
            log.warning("semantic intent classifier unavailable (%s); keyword classifier", e)
            return
        cal = d / "calibration.json"
        calibrated = False
        if cal.exists():
            calibrated = clf.load_state(json.loads(cal.read_text(encoding="utf-8")).get(enc.id, {}))
        self.semantic = clf
        self.model_info = {"model": f"semantic-knn/{enc.id}", "version": "1.0", "calibrated": calibrated,
                           **({"alpha": clf.alpha} if calibrated else {})}
        # P2: TensorFlow joint intent+slot model (NumPy runtime) on the same SBERT encoder
        jp = config.JOINT_NLU_PATH
        if jp.exists() and hasattr(enc, "encode_tokens"):
            from .joint_nlu import JointIntentModel, JointNluNumpy
            try:
                jm = JointIntentModel(clf, JointNluNumpy(jp), self._regex_slots)
                st = json.loads(cal.read_text(encoding="utf-8")).get("joint", {}) if cal.exists() else {}
                if jm.load_state(st):
                    self.semantic = jm
                    self.model_info = {"model": f"joint-nlu/{jm.id}", "version": "2.0", "calibrated": True,
                                       "alpha": jm.alpha, "intentMode": jm.mode, "slotMode": jm.slot_mode}
                else:
                    log.warning("joint NLU model present but not calibrated; run tools/eval_intent.py --write-calibration")
            except Exception as e:  # pragma: no cover
                log.warning("joint NLU model not loaded (%s)", e)
        log.info("intent classifier: %s", self.model_info)

    @staticmethod
    def keyword_classify(text: str) -> tuple[str, float]:
        low = text.lower()
        def hit(w: str) -> bool:   # short ASCII words need word boundaries ("ar" must not match "smart")
            return bool(re.search(rf"\b{re.escape(w)}\b", low)) if (w.isascii() and len(w) <= 3) else w in low
        scores = {k: sum(1 for w in ws if hit(w)) for k, ws in KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "eMBB", 0.6
        total = sum(scores.values())
        return best, round(min(0.97, 0.6 + 0.37 * scores[best] / total), 2)

    def classify_detail(self, text: str) -> dict[str, Any]:
        """{label, confidence, predictionSet?, probs?, evidence?}"""
        if self.model is not None:  # pragma: no cover - needs real weights
            import torch  # type: ignore
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                p = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
            i = int(p.argmax())
            return {"label": self.model.config.id2label[i], "confidence": float(p[i])}
        if self.semantic is not None and not self.force_keyword:
            return self.semantic.predict(text)
        st, conf = self.keyword_classify(text)
        return {"label": st, "confidence": conf}

    def classify(self, text: str) -> tuple[str, float]:
        d = self.classify_detail(text)
        return d["label"], d["confidence"]

    @staticmethod
    def _num(s: str) -> float:
        return float(s.replace(",", ""))

    @classmethod
    def extract_explicit(cls, text: str) -> dict[str, float]:
        """KPI values stated in the utterance (no defaults)."""
        low = text.lower()
        out: dict[str, float] = {}
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ms|밀리초)", low)
        if m:
            out["latency_ms"] = cls._num(m.group(1))
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(gbps|gbit|mbps|mbit|kbps)", low)
        if m:
            out["bandwidth_mbps"] = cls._num(m.group(1)) * {"g": 1000.0, "m": 1.0, "k": 0.001}[m.group(2)[0]]
        m = re.search(r"(9\d(?:\.\d+)?)\s*%", low)
        if m:
            out["reliability_percent"] = cls._num(m.group(1))
        elif "five nines" in low or "5 nines" in low or "파이브 나인" in low:
            out["reliability_percent"] = 99.999
        return out

    @classmethod
    def _regex_slots(cls, text: str) -> dict[str, Any]:
        d: dict[str, Any] = dict(cls.extract_explicit(text))
        dev = cls.devices(text)
        if dev is not None:
            d["devices"] = dev
        return d

    def expectations(self, text: str, slice_type: str, explicit: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        exp: dict[str, dict[str, Any]] = {e["kpi"]: dict(e) for e in DEFAULTS[slice_type]}
        explicit = self.extract_explicit(text) if explicit is None else explicit
        for kpi, v in explicit.items():
            if kpi == "devices":
                continue
            exp[kpi] = {"kpi": kpi, "op": "<=" if kpi == "latency_ms" else ">=", "value": v}
            if kpi == "latency_ms" and slice_type == "URLLC":
                exp[kpi]["percentile"] = 99
        return list(exp.values())

    @staticmethod
    def priority(text: str, slice_type: str) -> int:
        low = text.lower()
        for p in (1, 4, 2):   # explicit 'best effort' beats an implied priority
            if any(w in low for w in PRIORITY_WORDS[p]):
                return p
        return 2 if slice_type == "URLLC" else 3

    @staticmethod
    def devices(text: str) -> Optional[int]:
        m = re.search(r"(\d[\d,]*)\s*(?:devices|sensors|meters|users|ues|대|개|명)", text.lower())
        return int(m.group(1).replace(",", "")) if m else None

    def _semantic_block(self, st: str, conf: float, det: dict[str, Any]) -> dict[str, Any]:
        sem: dict[str, Any] = {"serviceCategory": st, "classifier": {**self.model_info, "confidence": conf}}
        if "predictionSet" in det:
            sem["classifier"]["predictionSet"] = det["predictionSet"]
            sem["classifier"]["probs"] = det["probs"]
            sem["evidence"] = det["evidence"]
            # conformal set with more than one class = ambiguous at the calibrated error level -> ask the operator
            sem["needsClarification"] = len(det["predictionSet"]) > 1
        if "slots" in det:
            sem["extractedSlots"] = det["slots"]
        return sem

    def parse(self, utterance: str, intent_id: str, tenant_id: str) -> tuple[dict[str, Any], int]:
        det = self.classify_detail(utterance)
        st, conf = det["label"], det["confidence"]
        low = utterance.lower()
        slots = det.get("slots")                       # TF tagger output when the joint model is active
        devices = slots.get("devices") if slots is not None else self.devices(utterance)
        hours = re.search(r"(?:for|동안)\s*(\d+)\s*(?:h|hours|시간)", low) or re.search(r"(\d+)\s*시간\s*동안", low)
        intent: dict[str, Any] = {
            "intentId": intent_id, "tenantId": tenant_id, "utterance": utterance,
            "semantic": self._semantic_block(st, conf, det),
            "expectations": self.expectations(utterance, st, slots),
            "context": {},
            "constraints": {"isolation": "dedicated" if (st == "URLLC" or "dedicated" in low or "전용" in low) else "shared"},
            "reportPolicy": {"periodSec": 10, "onDrift": True},
        }
        if hours:
            start = datetime.now(timezone.utc)
            intent["context"]["timeWindow"] = {"start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
                                               "end": (start + timedelta(hours=int(hours.group(1)))).isoformat(
                                                   timespec="seconds").replace("+00:00", "Z")}
        cost = re.search(r"(?:cost|budget|비용|예산)[^\d]*(\d+(?:\.\d+)?)", low)
        if cost:
            intent["constraints"]["maxCostPerHour"] = float(cost.group(1))
        sem = intent["semantic"]
        explicit = slots if slots is not None else self._regex_slots(utterance)
        if self.validator is not None:
            sem["validation"] = self.validator.check(st, explicit)
            if sem["validation"]["blocking"]:
                sem["needsClarification"] = True
        if self.nli is not None:
            op = self.nli.predict(utterance)
            sem["secondOpinion"] = {"model": self.nli.id, **op, "agrees": op["label"] == st, "gating": False}
        intent["_devices"] = devices
        return intent, self.priority(utterance, st)
