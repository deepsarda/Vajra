"""Common Alerting Protocol (CAP 1.2) alerts, the format NDMA's SACHET platform takes.

Prototype alerts default to status "Exercise", so nothing produced here can be mistaken
for a live warning.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

from shapely.geometry import Polygon

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
SEVERITY = {1: "Moderate", 2: "Severe", 3: "Extreme"}
COLOUR = {1: "Yellow", 2: "Orange", 3: "Red"}
TEXT = {
    "en-IN": {
        "event": "Thunderstorm with lightning",
        "headline": "{colour} alert: lightning likely in {area} within {minutes} minutes",
        "instruction": (
            "Leave open fields and water. Go inside a pucca building. Stay away from trees and poles."
        ),
    },
    "hi-IN": {
        "event": "आकाशीय बिजली के साथ आंधी",
        "headline": "{colour} अलर्ट: {area} में {minutes} मिनट में बिजली गिर सकती है",
        "instruction": "खुले खेत और पानी से दूर रहें। किसी पक्के मकान में जाएं।",
    },
}


def _cap_time(t: datetime) -> str:
    if t.tzinfo is None:
        raise ValueError("times must be timezone-aware")
    return t.isoformat(timespec="seconds")


def _cap_polygon(poly: Polygon) -> str:
    coords = list(poly.exterior.coords)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return " ".join(f"{lat:.4f},{lon:.4f}" for lon, lat in coords)


def build_cap(
    level: int,
    polygons: list[Polygon],
    area_desc: str,
    sent: datetime,
    valid_minutes: int = 80,
    lead_minutes: int = 20,
    sender: str = "vajra-prototype@example.org",
    sender_name: str = "Vajra prototype",
    status: str = "Exercise",
    languages: tuple[str, ...] = ("en-IN", "hi-IN"),
) -> str:
    """CAP 1.2 XML for one warning, with one <info> block per language."""
    if level not in SEVERITY:
        raise ValueError("level must be 1 (yellow), 2 (orange) or 3 (red)")
    ET.register_namespace("", CAP_NS)

    def sub(parent, tag, text=None):
        el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
        if text is not None:
            el.text = str(text)
        return el

    alert = ET.Element(f"{{{CAP_NS}}}alert")
    sub(alert, "identifier", f"vajra-{uuid.uuid4()}")
    sub(alert, "sender", sender)
    sub(alert, "sent", _cap_time(sent))
    sub(alert, "status", status)
    sub(alert, "msgType", "Alert")
    sub(alert, "scope", "Public")
    for lang in languages:
        words = TEXT[lang]
        info = sub(alert, "info")
        sub(info, "language", lang)
        sub(info, "category", "Met")
        sub(info, "event", words["event"])
        sub(info, "urgency", "Immediate" if lead_minutes <= 60 else "Expected")
        sub(info, "severity", SEVERITY[level])
        sub(info, "certainty", "Likely")
        sub(info, "effective", _cap_time(sent))
        sub(info, "expires", _cap_time(sent + timedelta(minutes=valid_minutes)))
        sub(info, "senderName", sender_name)
        headline = words["headline"].format(colour=COLOUR[level], area=area_desc, minutes=lead_minutes)
        sub(info, "headline", headline)
        sub(info, "instruction", words["instruction"])
        area = sub(info, "area")
        sub(area, "areaDesc", area_desc)
        for poly in polygons:
            sub(area, "polygon", _cap_polygon(poly))
    return ET.tostring(alert, encoding="unicode", xml_declaration=True)
