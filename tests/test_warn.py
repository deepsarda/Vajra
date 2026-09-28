from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from vajra.grid import Grid
from vajra.warn import build_cap, draft_polygons, exposure, risk_levels
from vajra.warn.cap import CAP_NS

GRID = Grid(nx=60, ny=60, km=2.0, lon0=82.5, lat0=24.8)


def disk_polygons():
    yy, xx = np.mgrid[0:60, 0:60]
    levels = ((xx - 30) ** 2 + (yy - 30) ** 2 <= 10**2).astype(np.int8) * 2
    return draft_polygons(levels, GRID, min_area_km2=10)


def test_more_exposure_means_a_higher_level():
    levels = risk_levels(np.full(2, 0.7), np.array([0.0, 1.0]))
    assert levels[0] < levels[1] == 3


def test_daytime_fields_are_more_exposed_than_night():
    pop, farm = np.array([2000.0]), np.array([0.9])
    assert exposure(pop, farm, 14.0) > exposure(pop, farm, 2.0)


def test_one_polygon_for_one_area():
    polys = disk_polygons()
    assert len(polys) == 1 and polys[0]["level"] == 2
    assert polys[0]["area_km2"] == pytest.approx(np.pi * 20**2, rel=0.15)  # radius 10 px = 20 km
    assert polys[0]["polygon"].centroid.y == pytest.approx(24.8 + 30 * GRID.dlat, abs=0.05)


def test_cap_alert_is_valid_and_marked_as_an_exercise():
    sent = datetime(2026, 5, 13, 13, 42, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    xml = build_cap(2, [disk_polygons()[0]["polygon"]], "Mirzapur district", sent)
    root = ET.fromstring(xml.encode("utf-8"))
    ns = {"cap": CAP_NS}
    assert root.find("cap:status", ns).text == "Exercise"
    infos = root.findall("cap:info", ns)
    assert [i.find("cap:language", ns).text for i in infos] == ["en-IN", "hi-IN"]
    assert infos[0].find("cap:severity", ns).text == "Severe"
    points = infos[0].find("cap:area/cap:polygon", ns).text.split()
    assert points[0] == points[-1] and len(points) >= 4
    with pytest.raises(ValueError):
        build_cap(2, [], "x", datetime(2026, 5, 13, 13, 42))
