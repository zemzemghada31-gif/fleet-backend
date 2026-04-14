from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import csv
import io

app = FastAPI(title="Parking LoRa API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---

class Device(BaseModel):
    id: str
    model: str
    assignment: str  # ASSIGNED, UNASSIGNED, MAINTENANCE
    last_connection: str
    status_color: str  # hex code for Flutter

class AnalyticsStats(BaseModel):
    total_distance: str
    avg_fuel_economy: str
    active_alerts: int
    distance_trend: str
    fuel_trend: str

class Vehicle(BaseModel):
    id: int
    model: str
    plate: str
    status: str
    tracker: Optional[str] = "Not Assigned"

# --- IN-MEMORY DATA ---

devices_db = [
    Device(id="X-9941-ALPHA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="2 mins ago", status_color="0xFF3B82F6"),
    Device(id="X-8820-BETA", model="Core Link Hub", assignment="UNASSIGNED", last_connection="14 hrs ago", status_color="0xFF64748B"),
    Device(id="X-1011-DELTA", model="Apex Tracker V3", assignment="MAINTENANCE", last_connection="Offline (3 days)", status_color="0xFFF59E0B"),
    Device(id="X-9950-GAMMA", model="Core Link Hub", assignment="ASSIGNED", last_connection="Just now", status_color="0xFF3B82F6"),
]

vehicles_db = [
    Vehicle(id=1, model="Tesla Model X", plate="BT-904-TX", status="ACTIVE", tracker="ST-449-ALPHA"),
    Vehicle(id=2, model="Mercedes Sprinter", plate="CA-123-VN", status="MAINTENANCE", tracker="Not Assigned"),
    Vehicle(id=3, model="Ford Transit XL", plate="TX-4409-LP", status="IDLE", tracker="ST-112-BETA"),
]

# --- ANALYTICS HELPERS ---

# Base stats per period
_BASE_STATS = {
    "last_30_days": {"distance": 284932,  "fuel": 14.2, "alerts": 24,  "dist_trend": "+12.4%", "fuel_trend": "-2.1%"},
    "quarterly":    {"distance": 892441,  "fuel": 13.8, "alerts": 61,  "dist_trend": "+8.7%",  "fuel_trend": "-3.4%"},
    "yearly":       {"distance": 3241002, "fuel": 13.1, "alerts": 187, "dist_trend": "+21.3%", "fuel_trend": "-6.2%"},
}

# Regional multipliers (fraction of total fleet activity)
_REGION_MULT = {
    "all":           1.00,
    "north_america": 0.42,
    "europe":        0.28,
    "apac":          0.18,
    "latam":         0.12,
}

# Trend data per period
_BASE_TRENDS = {
    "last_30_days": {
        "active":      [3.0, 4.0, 3.5, 5.0, 4.5, 2.0, 1.5],
        "maintenance": [1.0, 1.5, 1.0, 2.0, 1.5, 1.0, 0.5],
        "days":        ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
    },
    "quarterly": {
        "active":      [12.0, 15.0, 11.0, 18.0, 16.0, 9.0,  7.0],
        "maintenance": [3.0,  4.5,  3.0,  6.0,  5.0,  2.5,  1.5],
        "days":        ["W1",  "W2",  "W3",  "W4",  "W5",  "W6",  "W7"],
    },
    "yearly": {
        "active":      [40.0, 45.0, 38.0, 52.0, 49.0, 35.0, 28.0],
        "maintenance": [8.0,  10.0, 7.0,  14.0, 12.0, 6.0,  4.0],
        "days":        ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV", "DEC"],
    },
}


def _compute_stats(period: str, time_horizon: str, region: str) -> dict:
    base = _BASE_STATS.get(period, _BASE_STATS["last_30_days"])
    mult = _REGION_MULT.get(region, 1.0)

    distance = int(base["distance"] * mult)
    fuel      = round(base["fuel"] + (0.3 if region == "latam" else 0.0), 1)
    alerts    = max(1, int(base["alerts"] * mult))

    if time_horizon == "predictive":
        raw_dist = float(base["dist_trend"].strip("+%"))
        raw_fuel = float(base["fuel_trend"].strip("%"))
        dist_trend = f"+{raw_dist * 1.2:.1f}% (forecast)"
        fuel_trend = f"{raw_fuel * 1.1:.1f}% (forecast)"
    else:
        dist_trend = base["dist_trend"]
        fuel_trend = base["fuel_trend"]

    return {
        "total_distance": f"{distance:,} km",
        "avg_fuel_economy": f"{fuel} L/100km",
        "active_alerts": alerts,
        "distance_trend": dist_trend,
        "fuel_trend": fuel_trend,
    }


def _compute_trends(period: str, region: str) -> dict:
    base = _BASE_TRENDS.get(period, _BASE_TRENDS["last_30_days"])
    mult = _REGION_MULT.get(region, 1.0)
    return {
        "active":      [round(v * mult, 2) for v in base["active"]],
        "maintenance": [round(v * mult, 2) for v in base["maintenance"]],
        "days":        base["days"],
    }

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de Parking LoRa"}

@app.get("/api/analytics/stats", response_model=AnalyticsStats)
async def get_analytics_stats(
    period: str = "last_30_days",
    time_horizon: str = "historical",
    region: str = "all",
):
    stats = _compute_stats(period, time_horizon, region)
    return AnalyticsStats(**stats)

@app.get("/api/analytics/trends")
async def get_trends(
    period: str = "last_30_days",
    region: str = "all",
):
    return _compute_trends(period, region)

@app.get("/api/analytics/export")
async def export_analytics(
    period: str = "last_30_days",
    time_horizon: str = "historical",
    region: str = "all",
):
    stats  = _compute_stats(period, time_horizon, region)
    trends = _compute_trends(period, region)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Fleet Analytics Export"])
    writer.writerow(["Period", period, "Time Horizon", time_horizon, "Region", region])
    writer.writerow([])
    writer.writerow(["Metric", "Value", "Trend"])
    writer.writerow(["Total Distance",    stats["total_distance"],    stats["distance_trend"]])
    writer.writerow(["Avg Fuel Economy",  stats["avg_fuel_economy"],  stats["fuel_trend"]])
    writer.writerow(["Active Alerts",     stats["active_alerts"],     ""])
    writer.writerow([])
    writer.writerow(["Day", "Active Vehicles", "Maintenance Vehicles"])
    for i, day in enumerate(trends["days"]):
        writer.writerow([day, trends["active"][i], trends["maintenance"][i]])

    output.seek(0)
    filename = f"fleet_analytics_{period}_{region}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@app.get("/api/devices", response_model=List[Device])
async def get_devices():
    return devices_db

@app.post("/api/devices")
async def add_device(device: Device):
    if any(d.id == device.id for d in devices_db):
        raise HTTPException(status_code=400, detail="Device ID already exists")
    devices_db.append(device)
    return {"message": "Device registered successfully", "device": device}

@app.put("/api/devices/{device_id}")
async def update_device(device_id: str, device: Device):
    for i, d in enumerate(devices_db):
        if d.id == device_id:
            devices_db[i] = device
            return {"message": "Device updated", "device": device}
    raise HTTPException(status_code=404, detail="Device not found")

@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: str):
    for i, d in enumerate(devices_db):
        if d.id == device_id:
            devices_db.pop(i)
            return {"message": "Device deleted"}
    raise HTTPException(status_code=404, detail="Device not found")

@app.get("/api/vehicles", response_model=List[Vehicle])
def get_vehicles():
    return vehicles_db

@app.post("/api/vehicles")
def add_vehicle(vehicle: Vehicle):
    vehicles_db.append(vehicle)
    return {"message": "Vehicle added"}

@app.put("/api/vehicles/{vehicle_id}")
def update_vehicle(vehicle_id: int, vehicle: Vehicle):
    for i, v in enumerate(vehicles_db):
        if v.id == vehicle_id:
            vehicles_db[i] = vehicle
            return {"message": "Vehicle updated"}
    raise HTTPException(status_code=404, detail="Vehicle not found")

@app.delete("/api/vehicles/{vehicle_id}")
def delete_vehicle(vehicle_id: int):
    for i, v in enumerate(vehicles_db):
        if v.id == vehicle_id:
            vehicles_db.pop(i)
            return {"message": "Vehicle deleted"}
    raise HTTPException(status_code=404, detail="Vehicle not found")

# --- TRAJECTORY ENDPOINTS ---

_TRAJECTORY_DB = {
    1: {
        "vehicle": "Tesla Model X (BT-904-TX)",
        "total_distance": "1,248.5 mi",
        "avg_speed": "54.2 MPH",
        "fuel_used": "48.3 L",
        "segments": [
            {"from": "Oakland Depot", "to": "San Francisco HQ", "time": "08:30–09:45 AM", "distance": "48.2 mi", "duration": "75 min", "status": "COMPLETED"},
            {"from": "SF HQ", "to": "South Bay Hub", "time": "02:15–03:20 PM", "distance": "42.5 mi", "duration": "65 min", "status": "INCIDENT"},
        ],
        "stops": [
            {"name": "Oakland Depot", "address": "400 Hegenberger Rd, Oakland", "arrival": "08:00 AM", "duration": "30 min"},
            {"name": "San Francisco HQ", "address": "1 Market St, SF", "arrival": "09:45 AM", "duration": "85 min"},
            {"name": "South Bay Hub", "address": "200 El Camino Real, San Jose", "arrival": "03:20 PM", "duration": "55 min"},
        ],
        "events": [
            {"type": "DEPARTURE", "description": "Vehicle T-482 departed Oakland Depot", "time": "08:30 AM"},
            {"type": "SPEEDING", "description": "Speed exceeded 65 MPH on I-880", "time": "09:05 AM"},
            {"type": "ARRIVAL", "description": "Arrived at SF HQ", "time": "09:45 AM"},
            {"type": "LONG STOP", "description": "Unscheduled stop detected (45 min)", "time": "02:55 PM"},
            {"type": "ARRIVAL", "description": "Arrived at South Bay Hub", "time": "03:20 PM"},
        ],
    },
    2: {
        "vehicle": "Mercedes Sprinter (CA-123-VN)",
        "total_distance": "621.2 mi",
        "avg_speed": "41.8 MPH",
        "fuel_used": "32.1 L",
        "segments": [
            {"from": "San Jose Shop", "to": "Palo Alto Center", "time": "10:00–11:30 AM", "distance": "23.4 mi", "duration": "90 min", "status": "MAINTENANCE"},
        ],
        "stops": [
            {"name": "San Jose Shop", "address": "150 N First St, San Jose", "arrival": "09:45 AM", "duration": "15 min"},
            {"name": "Palo Alto Center", "address": "450 University Ave, Palo Alto", "arrival": "11:30 AM", "duration": "120 min"},
        ],
        "events": [
            {"type": "MAINTENANCE", "description": "Vehicle entered maintenance route", "time": "10:00 AM"},
            {"type": "ARRIVAL", "description": "Arrived at service center", "time": "11:30 AM"},
        ],
    },
    3: {
        "vehicle": "Ford Transit XL (TX-4409-LP)",
        "total_distance": "894.0 mi",
        "avg_speed": "58.6 MPH",
        "fuel_used": "55.4 L",
        "segments": [
            {"from": "SFO Airport", "to": "Downtown SF", "time": "06:15–07:00 AM", "distance": "14.2 mi", "duration": "45 min", "status": "COMPLETED"},
            {"from": "Downtown SF", "to": "Berkeley Terminal", "time": "09:30–10:15 AM", "distance": "15.8 mi", "duration": "45 min", "status": "COMPLETED"},
        ],
        "stops": [
            {"name": "SFO Airport", "address": "San Francisco International Airport", "arrival": "06:00 AM", "duration": "15 min"},
            {"name": "Downtown SF", "address": "Union Square, SF", "arrival": "07:00 AM", "duration": "150 min"},
            {"name": "Berkeley Terminal", "address": "2400 Telegraph Ave, Berkeley", "arrival": "10:15 AM", "duration": "30 min"},
        ],
        "events": [
            {"type": "DEPARTURE", "description": "Picked up cargo at SFO", "time": "06:15 AM"},
            {"type": "ARRIVAL", "description": "Delivery at Downtown SF", "time": "07:00 AM"},
            {"type": "DEPARTURE", "description": "En route to Berkeley", "time": "09:30 AM"},
            {"type": "ARRIVAL", "description": "Final delivery at Berkeley Terminal", "time": "10:15 AM"},
        ],
    },
}

@app.get("/api/trajectory/{vehicle_id}")
def get_trajectory(vehicle_id: int, date_from: str = "", date_to: str = ""):
    data = _TRAJECTORY_DB.get(vehicle_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trajectory not found for vehicle")
    return {**data, "date_from": date_from, "date_to": date_to}

@app.get("/api/trajectory/{vehicle_id}/export")
def export_trajectory(vehicle_id: int, date_from: str = "", date_to: str = ""):
    data = _TRAJECTORY_DB.get(vehicle_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trajectory not found")

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Trajectory Export", data["vehicle"]])
    w.writerow(["Period", f"{date_from} to {date_to}"])
    w.writerow([])
    w.writerow(["Total Distance", data["total_distance"]])
    w.writerow(["Average Speed", data["avg_speed"]])
    w.writerow(["Fuel Used", data["fuel_used"]])
    w.writerow([])
    w.writerow(["SEGMENTS", "From", "To", "Time", "Distance", "Duration", "Status"])
    for s in data["segments"]:
        w.writerow(["", s["from"], s["to"], s["time"], s["distance"], s["duration"], s["status"]])
    w.writerow([])
    w.writerow(["STOPS", "Name", "Address", "Arrival", "Duration"])
    for s in data["stops"]:
        w.writerow(["", s["name"], s["address"], s["arrival"], s["duration"]])
    w.writerow([])
    w.writerow(["EVENTS", "Type", "Description", "Time"])
    for e in data["events"]:
        w.writerow(["", e["type"], e["description"], e["time"]])

    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trajectory_vehicle{vehicle_id}.csv"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)