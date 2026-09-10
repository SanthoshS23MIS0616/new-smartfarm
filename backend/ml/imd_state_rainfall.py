"""
imd_state_rainfall.py
=====================
IMD historical average monthly rainfall normals by Indian state,
compiled from IMD 1981-2010 climatological publications.

Also provides region-aware season detection (South India has extended
Kharif due to Northeast Monsoon, unlike North India).
"""
from __future__ import annotations

_STATE_REGION: dict[str, str] = {
    "Punjab": "north_india", "Haryana": "north_india",
    "Himachal Pradesh": "north_india", "Uttarakhand": "north_india",
    "Uttar Pradesh": "north_india", "Rajasthan": "north_india",
    "Delhi": "north_india", "Jammu and Kashmir": "north_india",
    "West Bengal": "east_india", "Bihar": "east_india",
    "Jharkhand": "east_india", "Odisha": "east_india",
    "Assam": "east_india", "Arunachal Pradesh": "east_india",
    "Meghalaya": "east_india", "Manipur": "east_india",
    "Mizoram": "east_india", "Nagaland": "east_india",
    "Tripura": "east_india", "Sikkim": "east_india",
    "Gujarat": "west_india", "Maharashtra": "west_india",
    "Goa": "west_india", "Madhya Pradesh": "west_india",
    "Chhattisgarh": "west_india",
    "Andhra Pradesh": "south_india", "Telangana": "south_india",
    "Karnataka": "south_india", "Kerala": "south_india",
    "Tamil Nadu": "south_india",
}

SEASON_BY_REGION: dict[str, dict[str, list[int]]] = {
    "north_india": {"Kharif": [6,7,8,9,10], "Rabi": [11,12,1,2,3], "Summer": [4,5]},
    "south_india": {"Kharif": [6,7,8,9,10,11], "Rabi": [12,1,2,3,4], "Summer": [5]},
    "east_india":  {"Kharif": [6,7,8,9,10,11], "Rabi": [12,1,2,3],   "Summer": [4,5]},
    "west_india":  {"Kharif": [6,7,8,9,10],    "Rabi": [11,12,1,2,3], "Summer": [4,5]},
    "default":     {"Kharif": [6,7,8,9,10],    "Rabi": [11,12,1,2,3], "Summer": [4,5]},
}

# Jan→Dec average rainfall (mm) per state from IMD 1981-2010 normals
_MONTHLY_MM: dict[str, list[float]] = {
    "Andhra Pradesh":    [5,6,10,18,40,100,130,125,160,80,40,10],
    "Arunachal Pradesh": [30,35,80,150,300,480,560,500,400,200,60,20],
    "Assam":             [15,20,55,120,240,350,420,380,280,120,25,10],
    "Bihar":             [10,12,12,8,25,120,250,280,200,60,10,5],
    "Chhattisgarh":      [8,10,14,10,20,160,330,310,200,55,10,5],
    "Goa":               [2,1,2,8,60,600,800,750,380,80,20,5],
    "Gujarat":           [5,2,2,2,10,100,280,260,120,20,5,2],
    "Haryana":           [20,18,15,8,15,40,90,100,60,10,5,10],
    "Himachal Pradesh":  [70,70,65,40,55,90,170,180,120,35,20,50],
    "Jharkhand":         [15,20,20,20,40,190,340,320,240,70,15,8],
    "Karnataka":         [5,5,10,30,70,120,150,140,180,90,30,8],
    "Kerala":            [20,20,35,80,200,650,680,580,350,250,120,40],
    "Madhya Pradesh":    [8,10,10,5,12,140,310,290,180,40,10,5],
    "Maharashtra":       [5,3,3,5,20,160,380,350,200,40,12,3],
    "Manipur":           [25,30,60,110,190,280,310,280,220,100,35,15],
    "Meghalaya":         [25,35,95,220,430,750,830,700,480,200,50,15],
    "Mizoram":           [20,25,60,130,260,390,450,400,300,130,30,10],
    "Nagaland":          [20,25,55,110,230,350,400,360,280,120,35,15],
    "Odisha":            [15,20,20,20,50,200,360,350,250,80,30,10],
    "Punjab":            [25,22,20,10,15,30,80,90,50,8,4,18],
    "Rajasthan":         [8,8,5,3,8,30,100,90,45,5,3,5],
    "Sikkim":            [30,40,80,150,310,500,560,490,380,180,50,20],
    "Tamil Nadu":        [35,20,10,15,40,45,80,90,110,180,340,150],
    "Telangana":         [5,5,8,14,28,110,160,160,160,80,20,5],
    "Tripura":           [20,25,60,120,240,380,440,390,290,130,35,10],
    "Uttar Pradesh":     [20,18,12,5,15,80,200,240,160,35,10,12],
    "Uttarakhand":       [50,50,45,30,50,130,280,310,190,40,15,35],
    "West Bengal":       [15,20,25,40,120,280,380,360,280,130,40,10],
    "Jammu and Kashmir": [55,55,65,35,30,30,55,60,35,15,15,45],
}

_INDIA_AVG: list[float] = [20,18,18,15,30,150,280,270,180,70,25,12]


def get_state_region(state: str) -> str:
    return _STATE_REGION.get(state, "default")


def get_state_monthly_rainfall(state: str, month: int) -> float:
    """IMD historical average rainfall (mm) for given state and month (1-12)."""
    data = _MONTHLY_MM.get(state, _INDIA_AVG)
    return float(data[max(0, min(11, month - 1))])


def get_state_annual_rainfall(state: str) -> float:
    return float(sum(_MONTHLY_MM.get(state, _INDIA_AVG)))


def derive_season_for_state(month: int, state: str) -> str:
    """Region-aware season: Kharif, Rabi, or Summer."""
    region = get_state_region(state)
    for season_name, months in SEASON_BY_REGION.get(region, SEASON_BY_REGION["default"]).items():
        if month in months:
            return season_name
    return "Kharif"
