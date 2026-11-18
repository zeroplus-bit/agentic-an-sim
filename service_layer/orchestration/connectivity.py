"""Connectivity Management (ConnectivityManager): end-to-end connectivity (UE groups, areas, DNN) attached to an intent."""
from __future__ import annotations

import re
from typing import Any, Optional

# named areas of the simulated city -> cells of the twin (gNB-010 .. gNB-015, two sectors each)
AREAS: dict[str, dict[str, Any]] = {
    "hospital-a": {"aliases": ["hospital a", "hospital", "병원"], "cells": ["gNB-012-1", "gNB-012-2"], "dnn": "medical"},
    "city-edge": {"aliases": ["city edge", "downtown", "city centre", "city center", "도심", "시내"],
                  "cells": ["gNB-010-1", "gNB-010-2"], "dnn": "internet"},
    "campus": {"aliases": ["campus", "university", "캠퍼스", "대학"], "cells": ["gNB-011-1", "gNB-011-2"], "dnn": "internet"},
    "stadium": {"aliases": ["stadium", "arena", "concert", "경기장", "공연장"], "cells": ["gNB-013-1", "gNB-013-2"],
                "dnn": "internet"},
    "factory": {"aliases": ["factory", "plant", "industrial", "공장", "산업단지"], "cells": ["gNB-014-1", "gNB-014-2"],
                "dnn": "industry"},
    "port": {"aliases": ["port", "harbor", "harbour", "logistics", "항만", "물류"], "cells": ["gNB-015-1", "gNB-015-2"],
             "dnn": "iot"},
}


class ConnectivityManager:
    def __init__(self):
        self.ue_groups: dict[str, dict[str, Any]] = {}

    def resolve(self, utterance: str) -> Optional[tuple[str, dict[str, Any]]]:
        low = utterance.lower()
        for name, a in AREAS.items():
            if any(al in low for al in a["aliases"]):
                return name, a
        cells = re.findall(r"gNB-\d{3}-\d", utterance, flags=re.I)
        if cells:
            return "custom", {"cells": [c.replace("gnb", "gNB") for c in cells], "dnn": "internet"}
        return None

    def attach(self, intent: dict[str, Any], utterance: str, size: Optional[int] = None) -> dict[str, Any]:
        ctx = intent.setdefault("context", {})
        found = self.resolve(utterance)
        if found and not (ctx.get("area") or {}).get("cells"):
            name, a = found
            ctx["area"] = {"type": "cellList", "cells": list(a["cells"])}
            ctx.setdefault("dnn", a["dnn"])
            group = f"{name}-devices" if name != "custom" else f"{intent['intentId']}-devices"
            ctx.setdefault("ueGroup", group)
            self.ue_groups[ctx["ueGroup"]] = {"area": name, "cells": a["cells"], "size": size or 20,
                                              "intentId": intent["intentId"]}
        return intent

    def view(self) -> dict[str, Any]:
        return {"areas": {k: {"cells": v["cells"], "dnn": v["dnn"]} for k, v in AREAS.items()}, "ueGroups": self.ue_groups}
