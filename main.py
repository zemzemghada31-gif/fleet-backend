from database import engine, Base
import asyncio
import math
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import csv
import io
import random
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from database import init_db, get_db, DeviceModel, VehicleModel, MaintenanceLogModel, EntryExitModel, GPSHistoryModel, UserModel, AccessRuleModel, AccessLogModel, DeliveryModel, AsyncSessionLocal
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from jose import jwt, JWTError
import bcrypt as _bcrypt
import cv2
import numpy as np
import base64

def _hash_password(pw: str) -> str:
    return _bcrypt.hashpw(pw.encode(), _bcrypt.gensalt()).decode()

def _verify_password(pw: str, hashed: str) -> bool:
    return _bcrypt.checkpw(pw.encode(), hashed.encode())

JWT_SECRET = "fleet-command-secret-key-change-in-production"
JWT_ALGO = "HS256"
JWT_EXPIRY_HOURS = 24

async def get_current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)) -> Optional[UserModel]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = await db.get(UserModel, int(user_id))
        return user
    except (JWTError, ValueError, Exception):
        return None

async def require_admin(user: UserModel = Depends(get_current_user)) -> UserModel:
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DeviceModel))
        if not result.scalars().first():
            # Ajouter beaucoup de devices (trackers GPS)
            devices = [
                # Apex Tracker V3
                DeviceModel(id="X-9941-ALPHA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="2 mins ago", status_color="0xFF3B82F6", assigned_vehicle="Mercedes-Benz Actros (BT-904-TX)", assigned_since="2025-03-10 14:30"),
                DeviceModel(id="X-1001-BETA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="5 mins ago", status_color="0xFF3B82F6", assigned_vehicle="Volvo FH (TX-4409-LP)", assigned_since="2025-04-01 09:15"),
                DeviceModel(id="X-1002-GAMMA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="12 mins ago", status_color="0xFF3B82F6", assigned_vehicle="Renault Trucks D (FR-4401-P)", assigned_since="2025-02-18 11:00"),
                DeviceModel(id="X-1003-DELTA", model="Apex Tracker V3", assignment="UNASSIGNED", last_connection="1 hr ago", status_color="0xFF64748B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-1004-EPSILON", model="Apex Tracker V3", assignment="MAINTENANCE", last_connection="3 days ago", status_color="0xFFF59E0B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-1005-ZETA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="Just now", status_color="0xFF3B82F6", assigned_vehicle="Scania G-Series (FI-202-IK)", assigned_since="2025-05-01 08:00"),
                DeviceModel(id="X-1006-THETA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="30 mins ago", status_color="0xFF3B82F6", assigned_vehicle="DAF CF (CO-7710-D)", assigned_since="2025-04-22 16:45"),
                DeviceModel(id="X-1007-IOTA", model="Apex Tracker V3", assignment="UNASSIGNED", last_connection="5 hrs ago", status_color="0xFF64748B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-1008-KAPPA", model="Apex Tracker V3", assignment="ASSIGNED", last_connection="1 min ago", status_color="0xFF3B82F6", assigned_vehicle="Iveco S-Way (HY-505-UI)", assigned_since="2025-03-28 13:20"),
                DeviceModel(id="X-1009-LAMBDA", model="Apex Tracker V3", assignment="MAINTENANCE", last_connection="1 week ago", status_color="0xFFF59E0B", assigned_vehicle="—", assigned_since="—"),
                # Core Link Hub
                DeviceModel(id="X-8820-BETA", model="Core Link Hub", assignment="UNASSIGNED", last_connection="14 hrs ago", status_color="0xFF64748B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-8821-GAMMA", model="Core Link Hub", assignment="ASSIGNED", last_connection="3 mins ago", status_color="0xFF3B82F6", assigned_vehicle="DAF XF (PY-456-RT)", assigned_since="2025-01-15 10:30"),
                DeviceModel(id="X-8822-DELTA", model="Core Link Hub", assignment="ASSIGNED", last_connection="45 mins ago", status_color="0xFF3B82F6", assigned_vehicle="Volvo FM (NN-303-LP)", assigned_since="2025-04-10 07:00"),
                DeviceModel(id="X-8823-EPSILON", model="Core Link Hub", assignment="MAINTENANCE", last_connection="2 days ago", status_color="0xFFF59E0B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-8824-ZETA", model="Core Link Hub", assignment="ASSIGNED", last_connection="10 mins ago", status_color="0xFF3B82F6", assigned_vehicle="Renault Trucks T (CI-789-YU)", assigned_since="2025-05-05 12:00"),
                DeviceModel(id="X-8825-THETA", model="Core Link Hub", assignment="UNASSIGNED", last_connection="8 hrs ago", status_color="0xFF64748B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-8826-IOTA", model="Core Link Hub", assignment="ASSIGNED", last_connection="Just now", status_color="0xFF3B82F6", assigned_vehicle="Ford F-MAX (FR-606-TY)", assigned_since="2025-04-18 09:30"),
                DeviceModel(id="X-8827-KAPPA", model="Core Link Hub", assignment="ASSIGNED", last_connection="25 mins ago", status_color="0xFF3B82F6", assigned_vehicle="MAN TGS (TY-404-ER)", assigned_since="2025-03-05 14:15"),
                # Nano Sensor X1
                DeviceModel(id="X-7701-ALPHA", model="Nano Sensor X1", assignment="ASSIGNED", last_connection="15 mins ago", status_color="0xFF3B82F6", assigned_vehicle="Mercedes-Benz Arocs (VB-101-PO)", assigned_since="2025-02-28 16:00"),
                DeviceModel(id="X-7702-BETA", model="Nano Sensor X1", assignment="UNASSIGNED", last_connection="6 hrs ago", status_color="0xFF64748B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-7703-GAMMA", model="Nano Sensor X1", assignment="ASSIGNED", last_connection="1 min ago", status_color="0xFF3B82F6", assigned_vehicle="Scania R-Series (CA-123-VN)", assigned_since="2025-04-29 11:45"),
                DeviceModel(id="X-7704-DELTA", model="Nano Sensor X1", assignment="MAINTENANCE", last_connection="4 days ago", status_color="0xFFF59E0B", assigned_vehicle="—", assigned_since="—"),
                DeviceModel(id="X-7705-EPSILON", model="Nano Sensor X1", assignment="ASSIGNED", last_connection="20 mins ago", status_color="0xFF3B82F6", assigned_vehicle="MAN TGX (ZZ-123-ZZ)", assigned_since="2025-05-08 10:00"),
                DeviceModel(id="X-7706-ZETA", model="Nano Sensor X1", assignment="ASSIGNED", last_connection="Just now", status_color="0xFF3B82F6", assigned_vehicle="Renault Trucks D (FR-4401-P)", assigned_since="2025-04-12 08:30"),
            ]
            session.add_all(devices)

            # Ajouter beaucoup de véhicules
            vehicles = [
                VehicleModel(id=1, model="Mercedes-Benz Actros", plate="BT-904-TX", status="ACTIVE", tracker="X-9941-ALPHA", lat=41.8781, lng=-87.6298, speed=68.0, fuel=92, driver="Marcus Reed", eta="4.2H TO GO", heading="NE"),
                VehicleModel(id=2, model="Scania R-Series", plate="CA-123-VN", status="MAINTENANCE", tracker="Not Assigned", lat=37.3382, lng=-121.8863, speed=0.0, fuel=44, driver="Sarah Kim", eta="OFFLOADING", heading="—"),
                VehicleModel(id=3, model="Volvo FH", plate="TX-4409-LP", status="IDLE", tracker="ST-112-BETA", lat=37.7749, lng=-122.4194, speed=0.0, fuel=88, driver="Kevin Park", eta="LOADING", heading="—"),
                VehicleModel(id=4, model="MAN TGX", plate="ZZ-123-ZZ", status="ACTIVE", tracker="Not Assigned", lat=48.8566, lng=2.3522, speed=45.0, fuel=76, driver="Pierre Dubois", eta="2.1H TO GO", heading="SE"),
                VehicleModel(id=5, model="DAF XF", plate="PY-456-RT", status="ACTIVE", tracker="ST-449-ALPHA", lat=45.7640, lng=4.8357, speed=52.0, fuel=61, driver="Marie Laurent", eta="1.8H TO GO", heading="SW"),
                VehicleModel(id=6, model="Renault Trucks T", plate="CI-789-YU", status="IDLE", tracker="ST-112-BETA", lat=43.6047, lng=1.4442, speed=0.0, fuel=95, driver="Jean Moreau", eta="—", heading="—"),
                VehicleModel(id=7, model="Mercedes-Benz Arocs", plate="VB-101-PO", status="MAINTENANCE", tracker="Not Assigned", lat=52.5200, lng=13.4050, speed=0.0, fuel=33, driver="Hans Schmidt", eta="OFFLOADING", heading="—"),
                VehicleModel(id=8, model="Scania G-Series", plate="FI-202-IK", status="ACTIVE", tracker="ST-449-ALPHA", lat=41.9028, lng=12.4964, speed=71.0, fuel=84, driver="Luigi Rossi", eta="3.5H TO GO", heading="NE"),
                VehicleModel(id=9, model="Volvo FM", plate="NN-303-LP", status="ACTIVE", tracker="ST-112-BETA", lat=35.6762, lng=139.6503, speed=48.0, fuel=67, driver="Yuki Tanaka", eta="5.1H TO GO", heading="E"),
                VehicleModel(id=10, model="MAN TGS", plate="TY-404-ER", status="IDLE", tracker="Not Assigned", lat=51.5074, lng=-0.1278, speed=0.0, fuel=91, driver="James Wilson", eta="—", heading="—"),
                VehicleModel(id=11, model="Iveco S-Way", plate="HY-505-UI", status="ACTIVE", tracker="ST-449-ALPHA", lat=37.5665, lng=126.9780, speed=55.0, fuel=72, driver="Min-Jun Kim", eta="2.9H TO GO", heading="NW"),
                VehicleModel(id=12, model="Ford F-MAX", plate="FR-606-TY", status="MAINTENANCE", tracker="Not Assigned", lat=-33.8688, lng=151.2093, speed=0.0, fuel=28, driver="Jack Thompson", eta="OFFLOADING", heading="—"),
                VehicleModel(id=13, model="DAF CF", plate="CO-7710-D", status="ACTIVE", tracker="ST-449-ALPHA", lat=39.7392, lng=-104.9903, speed=62.0, fuel=78, driver="Amanda Lee", eta="2.5H TO GO", heading="NW"),
                VehicleModel(id=14, model="Renault Trucks D", plate="FR-4401-P", status="ACTIVE", tracker="ST-112-BETA", lat=46.2276, lng=4.8126, speed=71.0, fuel=85, driver="Lucas Moreau", eta="3.0H TO GO", heading="N"),
            ]
            session.add_all(vehicles)

            # Ajouter des logs de maintenance
            logs = [
                MaintenanceLogModel(vehicle_id=1, date="OCT 12, 2023", type="ROUTINE", title="Level 2 Service: Transmission Flush & Filtration", description="System pressure normalized. Minor wear detected on coupling. Fluid analyzed: Optimal.", file="Service_Report_A402.pdf", mileage="11,200 mi"),
                MaintenanceLogModel(vehicle_id=2, date="NOV 01, 2023", type="REPAIR", title="Engine Overhaul — Cylinder Head Replacement", description="Severe overheating detected. Replaced cylinder head and gaskets. Full coolant flush performed.", file="Repair_Invoice_B208.pdf", mileage="62,300 mi"),
                MaintenanceLogModel(vehicle_id=3, date="SEP 15, 2023", type="ROUTINE", title="Oil Change & Filter Replacement", description="Standard oil change performed. All fluids topped up. New air filter installed.", file="Service_Report_C315.pdf", mileage="45,800 mi"),
                MaintenanceLogModel(vehicle_id=4, date="DEC 05, 2023", type="REPAIR", title="Brake Pad Replacement", description="Front brake pads worn below 3mm. Replaced pads and resurfaced rotors.", file="Repair_Invoice_D420.pdf", mileage="28,150 mi"),
                MaintenanceLogModel(vehicle_id=5, date="JAN 20, 2024", type="ROUTINE", title="Tire Rotation & Alignment", description="Rotated tires and performed wheel alignment. Tread depth within specs.", file=None, mileage="33,400 mi"),
                MaintenanceLogModel(vehicle_id=7, date="FEB 10, 2024", type="REPAIR", title="Turbocharger Inspection", description="Whining noise from turbo. Inspected and replaced bearings. System tested OK.", file="Repair_Invoice_E501.pdf", mileage="78,900 mi"),
                MaintenanceLogModel(vehicle_id=8, date="MAR 01, 2024", type="ROUTINE", title="Battery Test & Replacement", description="Battery health at 62%. Replaced with new AGM battery. Charging system OK.", file=None, mileage="55,200 mi"),
            ]
            session.add_all(logs)

            # Ajouter des logs d'entrée/sortie
            gates = ["Main Gate", "East Gate", "West Gate", "South Gate"]
            plates = ["BT-904-TX", "CA-123-VN", "TX-4409-LP", "ZZ-123-ZZ", "PY-456-RT", "CI-789-YU", "VB-101-PO", "FI-202-IK", "NN-303-LP", "TY-404-ER"]
            models = ["Mercedes-Benz Actros", "Scania R-Series", "Volvo FH", "MAN TGX", "DAF XF", "Renault Trucks T", "Mercedes-Benz Arocs", "Scania G-Series", "Volvo FM", "MAN TGS"]
            drivers = ["Marcus Reed", "Sarah Kim", "Kevin Park", "Pierre Dubois", "Marie Laurent", "Jean Moreau", "Hans Schmidt", "Luigi Rossi", "Yuki Tanaka", "James Wilson"]
            entries = []
            for i in range(30):
                base = datetime.now() - timedelta(days=random.randint(0, 14), hours=random.randint(0, 23))
                entry = base
                exit = entry + timedelta(hours=random.randint(1, 12)) if random.random() > 0.15 else None
                entries.append(EntryExitModel(
                    vehicle_id=random.randint(1, 12),
                    vehicle_plate=random.choice(plates),
                    vehicle_model=random.choice(models),
                    driver=random.choice(drivers),
                    entry_time=entry.strftime("%Y-%m-%d %H:%M"),
                    exit_time=exit.strftime("%Y-%m-%d %H:%M") if exit else None,
                    gate=random.choice(gates),
                    status="OUTSIDE" if exit else "INSIDE",
                    notes=random.choice(["Livraison effectuée", "Maintenance programmée", "", "Stationnement longue durée", None]),
                ))
            session.add_all(entries)
            await session.commit()
    # Seed deliveries
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DeliveryModel))
        if not result.scalars().first():
            destinations = [
                (41.8987, -87.6243, "Chicago South Yard"),
                (37.7694, -122.4862, "SF Logistics Hub"),
                (48.8566, 2.3522, "Paris Distribution"),
                (45.7640, 4.8357, "Lyon Freight Center"),
                (43.6047, 1.4442, "Toulouse Depot"),
                (52.5200, 13.4050, "Berlin Terminal"),
                (41.9028, 12.4964, "Rome Cargo"),
                (35.6762, 139.6503, "Tokyo Port"),
                (51.5074, -0.1278, "London City Hub"),
                (37.5665, 126.9780, "Seoul Logistics"),
            ]
            vehicles_for_delivery = await session.execute(
                select(VehicleModel).limit(10)
            )
            deliveries = []
            for i, v in enumerate(vehicles_for_delivery.scalars()):
                if i >= len(destinations):
                    break
                lat, lng, name = destinations[i]
                deliveries.append(DeliveryModel(
                    vehicle_id=v.id,
                    vehicle_plate=v.plate,
                    vehicle_model=v.model,
                    driver=v.driver,
                    destination_lat=lat + random.uniform(-0.01, 0.01),
                    destination_lng=lng + random.uniform(-0.01, 0.01),
                    destination_name=name,
                    status="en_route",
                    assigned_at=datetime.utcnow() - timedelta(hours=random.randint(1, 6)),
                ))
            session.add_all(deliveries)
            await session.commit()

    # Seed access rules
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AccessRuleModel))
        if not result.scalars().first():
            rules = [
                AccessRuleModel(vehicle_plate="BT-904-TX", vehicle_model="Mercedes-Benz Actros", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="BT-904-TX", vehicle_model="Mercedes-Benz Actros", allowed=True, gate="Sortie"),
                AccessRuleModel(vehicle_plate="CA-123-VN", vehicle_model="Scania R-Series", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="CA-123-VN", vehicle_model="Scania R-Series", allowed=True, gate="Sortie"),
                AccessRuleModel(vehicle_plate="TX-4409-LP", vehicle_model="Volvo FH", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="TX-4409-LP", vehicle_model="Volvo FH", allowed=True, gate="Sortie"),
                AccessRuleModel(vehicle_plate="ZZ-123-ZZ", vehicle_model="MAN TGX", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="PY-456-RT", vehicle_model="DAF XF", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="CI-789-YU", vehicle_model="Renault Trucks T", allowed=True, gate="Sortie"),
                AccessRuleModel(vehicle_plate="VB-101-PO", vehicle_model="Mercedes-Benz Arocs", allowed=False, gate="Entrée", time_start="06:00", time_end="22:00"),
                AccessRuleModel(vehicle_plate="FI-202-IK", vehicle_model="Scania G-Series", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="NN-303-LP", vehicle_model="Volvo FM", allowed=True, gate="Sortie"),
                AccessRuleModel(vehicle_plate="FR-606-TY", vehicle_model="Ford F-MAX", allowed=False, gate="Entrée"),
                AccessRuleModel(vehicle_plate="CO-7710-D", vehicle_model="DAF CF", allowed=True, gate="Entrée"),
                AccessRuleModel(vehicle_plate="CO-7710-D", vehicle_model="DAF CF", allowed=True, gate="Sortie"),
            ]
            session.add_all(rules)
            await session.commit()

    # Seed access logs
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AccessLogModel))
        if not result.scalars().first():
            gates = ["Entrée", "Sortie"]
            plates = ["BT-904-TX", "CA-123-VN", "TX-4409-LP", "ZZ-123-ZZ", "PY-456-RT",
                      "CI-789-YU", "VB-101-PO", "FI-202-IK", "NN-303-LP", "FR-606-TY"]
            actions = ["ENTRY", "EXIT"]
            logs = []
            for i in range(40):
                base = datetime.utcnow() - timedelta(
                    days=random.randint(0, 7),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )
                plate = random.choice(plates)
                granted = random.random() > 0.15
                logs.append(AccessLogModel(
                    vehicle_plate=plate,
                    action=random.choice(actions),
                    gate=random.choice(gates),
                    granted=granted,
                    reason="Accès autorisé" if granted else "Plaque non autorisée",
                    timestamp=base,
                ))
            session.add_all(logs)
            await session.commit()

    # Seed default admin user
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(UserModel).where(UserModel.email == "admin@fleet.io"))
        if not result.scalar_one_or_none():
            session.add(UserModel(
                email="admin@fleet.io",
                    password_hash=_hash_password("admin123"),
                name="Admin User",
                role="admin",
                phone="+1 (800) 555-FLEET",
            ))
            await session.commit()

    # Démarrer la simulation GPS en arrière-plan
    sim_task = asyncio.create_task(_simulate_gps_movement())
    yield
    sim_task.cancel()
    # Shutdown (if needed)

# ── Simulation GPS en arrière-plan ──

_HEADING_DELTA = {
    "N":  (0.0005, 0),
    "NE": (0.0003, 0.0004),
    "E":  (0, 0.0005),
    "SE": (-0.0003, 0.0004),
    "S":  (-0.0005, 0),
    "SW": (-0.0003, -0.0004),
    "W":  (0, -0.0005),
    "NW": (0.0003, -0.0004),
}

def _geo_distance_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Approximate Euclidean distance in degrees (~111km per degree)."""
    return ((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2) ** 0.5

_GEOFENCE_THRESHOLD_DEG = 0.005  # ~500 meters

async def _simulate_gps_movement():
    while True:
        await asyncio.sleep(3)
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(VehicleModel).where(VehicleModel.status == "ACTIVE")
                )
                vehicles = result.scalars().all()
                for v in vehicles:
                    dlat, dlng = _HEADING_DELTA.get(v.heading, (0, 0))
                    v.lat += dlat + (random.random() * 0.0002 - 0.0001)
                    v.lng += dlng + (random.random() * 0.0002 - 0.0001)
                    v.speed = max(0, v.speed + random.uniform(-3, 3))
                    v.fuel = max(0, min(100, v.fuel - (1 if random.random() < 0.3 else 0)))

                # Geo-fence check: see if any active vehicle arrived at destination
                if vehicles:
                    vehicle_ids = [v.id for v in vehicles]
                    deliveries = await session.execute(
                        select(DeliveryModel).where(
                            DeliveryModel.vehicle_id.in_(vehicle_ids),
                            DeliveryModel.status == "en_route",
                        )
                    )
                    for d in deliveries.scalars():
                        vehicle = next((v for v in vehicles if v.id == d.vehicle_id), None)
                        if vehicle is None:
                            continue
                        dist = _geo_distance_deg(vehicle.lat, vehicle.lng, d.destination_lat, d.destination_lng)
                        if dist < _GEOFENCE_THRESHOLD_DEG:
                            d.status = "arrived"
                            d.arrived_at = datetime.utcnow()
                            vehicle.eta = "OFFLOADING"
                            print(f"[LIVRAISON] {vehicle.model} ({vehicle.plate}) est arrivée à {d.destination_name}")

                await session.commit()
        except Exception:
            pass

app = FastAPI(title="Parking LoRa API", lifespan=lifespan)

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
    assignment: str
    last_connection: str
    status_color: str
    assigned_vehicle: str = "—"
    assigned_since: str = "—"

    model_config = ConfigDict(from_attributes=True)

class Vehicle(BaseModel):
    id: Optional[int] = None
    model: str
    plate: str
    status: str
    tracker: Optional[str] = "Not Assigned"
    lat: Optional[float] = 40.7128
    lng: Optional[float] = -74.0060
    speed: Optional[float] = 0.0
    fuel: Optional[int] = 80
    driver: Optional[str] = "Unassigned"
    eta: Optional[str] = "—"
    heading: Optional[str] = "—"

    model_config = ConfigDict(from_attributes=True)

class LiveVehicle(BaseModel):
    id: str
    location: str
    status: str
    speed: float
    fuel: int
    driver: str
    eta: str
    heading: str
    speed_history: Optional[list] = []
    lat: float
    lng: float

    model_config = ConfigDict(from_attributes=True)

class MaintenanceLog(BaseModel):
    id: Optional[int]
    vehicle_id: int
    date: str
    type: str
    title: str
    description: str
    file: Optional[str] = None
    mileage: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class EntryExit(BaseModel):
    id: Optional[int] = None
    vehicle_id: int
    vehicle_plate: str
    vehicle_model: str
    driver: Optional[str] = None
    entry_time: str
    exit_time: Optional[str] = None
    gate: str = "Main Gate"
    status: str = "INSIDE"
    notes: Optional[str] = None
    image_b64: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class GPSData(BaseModel):
    device_id: str
    lat: float
    lng: float
    speed: float = 0.0
    heading: str = "N"
    fuel: int = 0
    timestamp: str = ""

    model_config = ConfigDict(from_attributes=True)

class GPSDataResponse(BaseModel):
    id: int
    device_id: str
    lat: float
    lng: float
    speed: float
    heading: str
    fuel: int
    timestamp: str

    model_config = ConfigDict(from_attributes=True)

class AnalyticsStats(BaseModel):
    total_distance: str
    avg_fuel_economy: str
    active_alerts: int
    distance_trend: str
    fuel_trend: str

# --- AUTH MODELS ---

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class UserOut(BaseModel):
    id: int
    email: str
    name: str
    role: str
    phone: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: str = "operator"
    phone: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None

# --- ACCESS CONTROL MODELS ---

class AccessRuleOut(BaseModel):
    id: int
    vehicle_plate: str
    vehicle_model: Optional[str] = None
    allowed: bool = True
    gate: str = "Entrée"
    time_start: str = "00:00"
    time_end: str = "23:59"
    created_by: Optional[int] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class AccessRuleCreate(BaseModel):
    vehicle_plate: str
    vehicle_model: Optional[str] = None
    allowed: bool = True
    gate: str = "Entrée"
    time_start: str = "00:00"
    time_end: str = "23:59"

class AccessCheckRequest(BaseModel):
    vehicle_plate: str
    gate: str = "Entrée"

class AccessCheckResponse(BaseModel):
    granted: bool
    reason: str
    rule: Optional[AccessRuleOut] = None

class AccessLogOut(BaseModel):
    id: int
    vehicle_plate: str
    action: str
    gate: str
    granted: bool
    reason: Optional[str] = None
    scanned_by: Optional[int] = None
    timestamp: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class AccessLogCreate(BaseModel):
    vehicle_plate: str
    action: str = "ENTRY"
    gate: str = "Entrée"
    granted: bool = True
    reason: Optional[str] = None
    image_b64: Optional[str] = None

# --- DELIVERY MODELS ---

class DeliveryOut(BaseModel):
    id: int
    vehicle_id: int
    vehicle_plate: str
    vehicle_model: Optional[str] = None
    driver: Optional[str] = None
    destination_lat: float
    destination_lng: float
    destination_name: Optional[str] = None
    status: str = "en_route"
    eta_minutes: Optional[float] = None
    assigned_at: Optional[str] = None
    arrived_at: Optional[str] = None
    delivered_at: Optional[str] = None
    notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class DeliveryCreate(BaseModel):
    vehicle_id: int
    destination_lat: float
    destination_lng: float
    destination_name: Optional[str] = None
    notes: Optional[str] = None

# ── Delivery helpers ──

def _eta_minutes(vehicle_lat: float, vehicle_lng: float,
                  dest_lat: float, dest_lng: float,
                  speed: float) -> Optional[float]:
    if speed <= 0:
        return None
    dlat = (vehicle_lat - dest_lat) * 111.0
    mid_lat = math.radians((vehicle_lat + dest_lat) / 2)
    dlng = (vehicle_lng - dest_lng) * 111.0 * math.cos(mid_lat)
    dist_km = math.sqrt(dlat ** 2 + dlng ** 2)
    minutes = (dist_km / speed) * 60
    return round(minutes, 1)

def _delivery_to_out(d: DeliveryModel, vehicle: Optional[VehicleModel] = None) -> DeliveryOut:
    eta = None
    if d.status == "en_route" and vehicle:
        eta = _eta_minutes(vehicle.lat, vehicle.lng,
                          d.destination_lat, d.destination_lng,
                          vehicle.speed or 0)
    return DeliveryOut(
        id=d.id, vehicle_id=d.vehicle_id, vehicle_plate=d.vehicle_plate,
        vehicle_model=d.vehicle_model, driver=d.driver,
        destination_lat=d.destination_lat, destination_lng=d.destination_lng,
        destination_name=d.destination_name, status=d.status,
        eta_minutes=eta,
        assigned_at=d.assigned_at.isoformat() if d.assigned_at else None,
        arrived_at=d.arrived_at.isoformat() if d.arrived_at else None,
        delivered_at=d.delivered_at.isoformat() if d.delivered_at else None,
        notes=d.notes,
    )

# --- ANALYTICS HELPERS (Keep in memory as they are static/computed) ---

_BASE_STATS = {
    "last_30_days": {"distance": 284932,  "fuel": 14.2, "alerts": 24,  "dist_trend": "+12.4%", "fuel_trend": "-2.1%"},
    "quarterly":    {"distance": 892441,  "fuel": 13.8, "alerts": 61,  "dist_trend": "+8.7%",  "fuel_trend": "-3.4%"},
    "yearly":       {"distance": 3241002, "fuel": 13.1, "alerts": 187, "dist_trend": "+21.3%", "fuel_trend": "-6.2%"},
}

_REGION_MULT = {
    "all": 1.00, "north_america": 0.42, "europe": 0.28, "apac": 0.18, "latam": 0.12,
}

_BASE_TRENDS = {
    "last_30_days": {
        "active": [3.0, 4.0, 3.5, 5.0, 4.5, 2.0, 1.5],
        "maintenance": [1.0, 1.5, 1.0, 2.0, 1.5, 1.0, 0.5],
        "days": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
    },
    "quarterly": {
        "active": [12.0, 15.0, 11.0, 18.0, 16.0, 9.0, 7.0],
        "maintenance": [3.0, 4.5, 3.0, 6.0, 5.0, 2.5, 1.5],
        "days": ["W1", "W2", "W3", "W4", "W5", "W6", "W7"],
    },
    "yearly": {
        "active": [40.0, 45.0, 38.0, 52.0, 49.0, 35.0, 28.0],
        "maintenance": [8.0, 10.0, 7.0, 14.0, 12.0, 6.0, 4.0],
        "days": ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV", "DEC"],
    },
}

def _compute_stats(period: str, time_horizon: str, region: str) -> dict:
    base = _BASE_STATS.get(period, _BASE_STATS["last_30_days"])
    mult = _REGION_MULT.get(region, 1.0)
    distance = int(base["distance"] * mult)
    fuel = round(base["fuel"] + (0.3 if region == "latam" else 0.0), 1)
    alerts = max(1, int(base["alerts"] * mult))
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
        "active": [round(v * mult, 2) for v in base["active"]],
        "maintenance": [round(v * mult, 2) for v in base["maintenance"]],
        "days": base["days"],
    }

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de Parking LoRa avec stockage persistant"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/analytics/stats", response_model=AnalyticsStats)
async def get_analytics_stats(period: str = "last_30_days", time_horizon: str = "historical", region: str = "all"):
    return AnalyticsStats(**_compute_stats(period, time_horizon, region))

@app.get("/api/analytics/trends")
async def get_trends(period: str = "last_30_days", region: str = "all"):
    return _compute_trends(period, region)

@app.get("/api/live")
async def get_live_vehicles(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(VehicleModel))
    vehicles = result.scalars().all()
    live_data = []
    for v in vehicles:
        live_data.append({
            "id": f"{v.plate[:2].upper()}-{v.id:04d}-{v.model[:1].upper()}",
            "location": f"{v.model} - {v.plate}",
            "status": "MOVING" if v.status == "ACTIVE" else v.status,
            "speed": v.speed or 0.0,
            "fuel": v.fuel or 80,
            "driver": v.driver or "Unassigned",
            "eta": v.eta or "—",
            "heading": v.heading or "—",
            "speedHistory": [v.speed or 0.0] * 10,
            "lat": v.lat or 40.7128,
            "lng": v.lng or -74.0060,
        })
    return live_data

@app.get("/api/devices", response_model=List[Device])
async def get_devices(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DeviceModel))
    return result.scalars().all()

@app.post("/api/devices")
async def add_device(device: Device, db: AsyncSession = Depends(get_db)):
    db_device = await db.get(DeviceModel, device.id)
    if db_device:
        raise HTTPException(status_code=400, detail="Device ID already exists")
    new_device = DeviceModel(**device.model_dump())
    db.add(new_device)
    await db.commit()
    return {"message": "Device registered successfully", "device": device}

@app.put("/api/devices/{device_id}")
async def update_device(device_id: str, device: Device, db: AsyncSession = Depends(get_db)):
    db_device = await db.get(DeviceModel, device_id)
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    for key, value in device.model_dump().items():
        setattr(db_device, key, value)
    await db.commit()
    return {"message": "Device updated", "device": device}

@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: str, db: AsyncSession = Depends(get_db)):
    db_device = await db.get(DeviceModel, device_id)
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    await db.delete(db_device)
    await db.commit()
    return {"message": "Device deleted"}

@app.get("/api/vehicles", response_model=List[Vehicle])
async def get_vehicles(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(VehicleModel))
    return result.scalars().all()

@app.post("/api/vehicles")
async def add_vehicle(vehicle: Vehicle, db: AsyncSession = Depends(get_db)):
    data = vehicle.model_dump(exclude={"id"})
    new_vehicle = VehicleModel(**data)
    db.add(new_vehicle)
    await db.commit()
    return {"message": "Vehicle added"}

@app.put("/api/vehicles/{vehicle_id}")
async def update_vehicle(vehicle_id: int, vehicle: Vehicle, db: AsyncSession = Depends(get_db)):
    db_vehicle = await db.get(VehicleModel, vehicle_id)
    if not db_vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    for key, value in vehicle.dict().items():
        setattr(db_vehicle, key, value)
    await db.commit()
    return {"message": "Vehicle updated"}

@app.delete("/api/vehicles/{vehicle_id}")
async def delete_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    db_vehicle = await db.get(VehicleModel, vehicle_id)
    if not db_vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    await db.delete(db_vehicle)
    await db.commit()
    return {"message": "Vehicle deleted"}

@app.get("/api/maintenance/{vehicle_id}/logs", response_model=List[MaintenanceLog])
async def get_maintenance_logs(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MaintenanceLogModel).where(MaintenanceLogModel.vehicle_id == vehicle_id))
    return result.scalars().all()

@app.post("/api/maintenance/{vehicle_id}/logs", status_code=201)
async def add_maintenance_log(vehicle_id: int, log: MaintenanceLog, db: AsyncSession = Depends(get_db)):
    new_log = MaintenanceLogModel(**log.model_dump(exclude={"id"}))
    new_log.vehicle_id = vehicle_id
    db.add(new_log)
    await db.commit()
    await db.refresh(new_log)
    return new_log

# --- Maintenance diagnostics endpoint ---
@app.get("/api/maintenance/{vehicle_id}/diagnostics")
async def get_diagnostics(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    # Simuler des données de diagnostic (à remplacer par de vraies données plus tard)
    import random
    return {
        "battery_health": random.randint(60, 100),
        "next_service_days": random.randint(1, 90),
        "next_service_type": random.choice(["Brake Fluid Replacement", "Engine Inspection", "Tire Rotation"]),
        "fuel_consumption": round(random.uniform(14.0, 25.0), 1),
        "brake_pad_life": random.randint(10, 100),
        "coolant_temp": random.randint(90, 120),
        "thermostat_temp": f"{random.randint(180, 220)}°",
        "thermostat_trend": random.choice(["+0.2% Stability", "-0.1% Stable", "+1.8% Rising"]),
        "thermostat_spots": [round(random.uniform(2.0, 7.0), 2), round(random.uniform(2.0, 7.0), 2), round(random.uniform(2.0, 7.0), 2), round(random.uniform(2.0, 7.0), 2), round(random.uniform(2.0, 7.0), 2)],
        "dtc_codes": random.choice([
            [{"code": "P0420", "description": "Catalyst Efficiency Below Threshold", "detected": f"{random.randint(1, 5)}h ago", "location": "Block 1", "severity": random.choice(["WARNING", "CRITICAL"])}],
            [{"code": "P0300", "description": "Random/Multiple Cylinder Misfire", "detected": f"{random.randint(1, 5)}h ago", "location": "All Cylinders", "severity": "CRITICAL"}],
            []
        ]),
        "predictive": {"probability": random.randint(10, 95), "component": random.choice(["fuel pump", "engine cooling system", "transmission"]), "miles_remaining": random.randint(100, 2500)},
    }

# --- Maintenance parts endpoint ---
@app.get("/api/maintenance/{vehicle_id}/parts")
async def get_parts(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    # Simuler des données de pièces de rechange
    import random
    parts = []
    part_names = [
        ("Brake Pads (Front)", "BP-2023-F", random.choice(["LOW_STOCK", "IN_STOCK", "ORDER_NOW"])),
        ("Engine Air Filter", "AF-1102-X", random.choice(["IN_STOCK", "LOW_STOCK"])),
        ("Transmission Fluid", "TF-884-SYN", random.choice(["ORDER_NOW", "IN_STOCK"])),
        ("Oil Filter", "OF-5567", random.choice(["IN_STOCK", "LOW_STOCK"])),
        ("Spark Plugs", "SP-440-X", random.choice(["IN_STOCK", "ORDER_NOW"])),
    ]
    for i, (name, pn, status) in enumerate(part_names, 1):
        parts.append({
            "id": i,
            "name": name,
            "partNumber": pn,
            "status": status,
            "quantity": random.randint(0, 10),
            "lastReplaced": random.choice(["JAN 2023", "MAR 2023", "OCT 2022"]),
            "nextReplacement": random.choice(["DEC 2023", "MAR 2024", "JUN 2024"]),
        })
    return parts

# --- Entry/Exit endpoints ---
@app.get("/api/entry-exit", response_model=List[EntryExit])
async def get_entry_exit(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(EntryExitModel).order_by(EntryExitModel.id.desc()))
    return result.scalars().all()

@app.get("/api/entry-exit/{entry_id}", response_model=EntryExit)
async def get_entry_exit_by_id(entry_id: int, db: AsyncSession = Depends(get_db)):
    entry = await db.get(EntryExitModel, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry

@app.post("/api/entry-exit")
async def create_entry_exit(entry: EntryExit, db: AsyncSession = Depends(get_db)):
    new_entry = EntryExitModel(**entry.model_dump(exclude={"id"}))
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return new_entry

@app.put("/api/entry-exit/{entry_id}/exit")
async def record_exit(entry_id: int, exit_time: str, db: AsyncSession = Depends(get_db)):
    db_entry = await db.get(EntryExitModel, entry_id)
    if not db_entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    db_entry.exit_time = exit_time
    db_entry.status = "OUTSIDE"
    await db.commit()
    return {"message": "Exit recorded", "entry": db_entry}

# ── GPS Endpoints ──

@app.post("/api/gps/update")
async def receive_gps(data: GPSData, db: AsyncSession = Depends(get_db)):
    vehicle = await db.execute(
        select(VehicleModel).where(VehicleModel.tracker == data.device_id)
    )
    v = vehicle.scalar_one_or_none()
    if v:
        v.lat = data.lat
        v.lng = data.lng
        v.speed = data.speed
        v.heading = data.heading
        v.fuel = data.fuel if data.fuel else v.fuel

    device = await db.get(DeviceModel, data.device_id)
    if device:
        device.last_connection = "Just now"

    history = GPSHistoryModel(
        device_id=data.device_id,
        lat=data.lat,
        lng=data.lng,
        speed=data.speed,
        heading=data.heading,
        fuel=data.fuel,
        timestamp=datetime.utcnow()
    )
    db.add(history)
    await db.commit()
    return {"status": "ok", "vehicle_id": v.id if v else None}

@app.get("/api/gps/history/{device_id}", response_model=List[GPSDataResponse])
async def get_gps_history(device_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GPSHistoryModel)
        .where(GPSHistoryModel.device_id == device_id)
        .order_by(GPSHistoryModel.timestamp.desc())
        .limit(100)
    )
    return result.scalars().all()

@app.get("/api/gps/vehicle/{vehicle_id}/history", response_model=List[GPSDataResponse])
async def get_vehicle_gps_history(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    vehicle = await db.get(VehicleModel, vehicle_id)
    if not vehicle or not vehicle.tracker or vehicle.tracker == "Not Assigned":
        return []
    result = await db.execute(
        select(GPSHistoryModel)
        .where(GPSHistoryModel.device_id == vehicle.tracker)
        .order_by(GPSHistoryModel.timestamp.desc())
        .limit(100)
    )
    return result.scalars().all()

@app.get("/api/gps/ingress")
async def gps_ingress_get(
    id: str,
    lat: float,
    lng: float,
    speed: float = 0,
    heading: str = "N",
    fuel: int = 0,
    db: AsyncSession = Depends(get_db)
):
    data = GPSData(device_id=id, lat=lat, lng=lng, speed=speed, heading=heading, fuel=fuel)
    return await receive_gps(data, db)

# ── AUTH ENDPOINTS ──

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel).where(UserModel.email == data.email))
    user = result.scalar_one_or_none()
    if not user or not _verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    token = jwt.encode({
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }, JWT_SECRET, algorithm=JWT_ALGO)
    return TokenResponse(
        access_token=token,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "phone": user.phone or "",
        }
    )

@app.get("/api/auth/me")
async def get_me(user: UserModel = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "phone": user.phone or "",
        "is_active": user.is_active,
    }

# ── USER MANAGEMENT (admin only) ──

@app.get("/api/users", response_model=List[UserOut])
async def get_users(admin: UserModel = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel))
    users = result.scalars().all()
    return [UserOut(
        id=u.id, email=u.email, name=u.name, role=u.role,
        phone=u.phone, is_active=u.is_active,
        created_at=u.created_at.isoformat() if u.created_at else None,
    ) for u in users]

@app.post("/api/users", status_code=201)
async def create_user(data: UserCreate, admin: UserModel = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(UserModel).where(UserModel.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already exists")
    user = UserModel(
        email=data.email,
        password_hash=_hash_password(data.password),
        name=data.name,
        role=data.role,
        phone=data.phone,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"message": "User created", "id": user.id}

@app.put("/api/users/{user_id}")
async def update_user(user_id: int, data: UserUpdate, admin: UserModel = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = await db.get(UserModel, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for key, value in data.model_dump(exclude_none=True).items():
        setattr(user, key, value)
    await db.commit()
    return {"message": "User updated"}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, admin: UserModel = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = await db.get(UserModel, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    await db.delete(user)
    await db.commit()
    return {"message": "User deleted"}

# ── ACCESS RULES ──

@app.get("/api/access/rules", response_model=List[AccessRuleOut])
async def get_access_rules(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AccessRuleModel).order_by(AccessRuleModel.id.desc()))
    rules = result.scalars().all()
    return [AccessRuleOut(
        id=r.id, vehicle_plate=r.vehicle_plate, vehicle_model=r.vehicle_model,
        allowed=r.allowed, gate=r.gate, time_start=r.time_start, time_end=r.time_end,
        created_by=r.created_by,
        created_at=r.created_at.isoformat() if r.created_at else None,
    ) for r in rules]

@app.post("/api/access/rules", status_code=201)
async def create_access_rule(data: AccessRuleCreate, user: UserModel = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rule = AccessRuleModel(
        vehicle_plate=data.vehicle_plate.upper(),
        vehicle_model=data.vehicle_model,
        allowed=data.allowed,
        gate=data.gate,
        time_start=data.time_start,
        time_end=data.time_end,
        created_by=user.id if user else None,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return {"message": "Access rule created", "id": rule.id}

@app.put("/api/access/rules/{rule_id}")
async def update_access_rule(rule_id: int, data: AccessRuleCreate, user: UserModel = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rule = await db.get(AccessRuleModel, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    for key, value in data.model_dump().items():
        if key == "vehicle_plate":
            setattr(rule, key, value.upper())
        else:
            setattr(rule, key, value)
    await db.commit()
    return {"message": "Rule updated"}

@app.delete("/api/access/rules/{rule_id}")
async def delete_access_rule(rule_id: int, user: UserModel = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rule = await db.get(AccessRuleModel, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    await db.delete(rule)
    await db.commit()
    return {"message": "Rule deleted"}

@app.post("/api/access/check", response_model=AccessCheckResponse)
async def check_access(data: AccessCheckRequest, db: AsyncSession = Depends(get_db)):
    plate = data.vehicle_plate.upper()
    result = await db.execute(
        select(AccessRuleModel).where(
            AccessRuleModel.vehicle_plate == plate,
            AccessRuleModel.gate == data.gate,
        )
    )
    rule = result.scalar_one_or_none()
    if rule:
        return AccessCheckResponse(
            granted=rule.allowed,
            reason="Access granted" if rule.allowed else "Access denied by rule",
            rule=AccessRuleOut(
                id=rule.id, vehicle_plate=rule.vehicle_plate,
                vehicle_model=rule.vehicle_model, allowed=rule.allowed,
                gate=rule.gate, time_start=rule.time_start, time_end=rule.time_end,
                created_by=rule.created_by,
                created_at=rule.created_at.isoformat() if rule.created_at else None,
            ),
        )
    # Check if it's a known vehicle
    vehicle = await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate)
    )
    v = vehicle.scalar_one_or_none()
    if v:
        return AccessCheckResponse(granted=True, reason="Known vehicle, no specific rule")
    return AccessCheckResponse(granted=False, reason="Unknown vehicle, no access rule found")

# ── ACCESS LOGS ──

@app.post("/api/access/log", status_code=201)
async def log_access(data: AccessLogCreate, user: UserModel = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    log = AccessLogModel(
        vehicle_plate=data.vehicle_plate.upper(),
        action=data.action,
        gate=data.gate,
        granted=data.granted,
        reason=data.reason,
        scanned_by=user.id if user else None,
        image_b64=data.image_b64,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    # If granted entry, also create an entry-exit record
    if data.granted and data.action == "ENTRY":
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        vehicle = await db.execute(
            select(VehicleModel).where(VehicleModel.plate == data.vehicle_plate.upper())
        )
        v = vehicle.scalar_one_or_none()
        entry = EntryExitModel(
            vehicle_id=v.id if v else 0,
            vehicle_plate=data.vehicle_plate.upper(),
            vehicle_model=v.model if v else "Unknown",
            driver=v.driver if v else None,
            entry_time=now,
            gate=data.gate,
            status="INSIDE",
        )
        db.add(entry)
        await db.commit()

    return {"message": "Access logged", "id": log.id}

@app.get("/api/access/logs", response_model=List[AccessLogOut])
async def get_access_logs(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AccessLogModel).order_by(AccessLogModel.id.desc()).limit(limit)
    )
    logs = result.scalars().all()
    return [AccessLogOut(
        id=l.id, vehicle_plate=l.vehicle_plate, action=l.action,
        gate=l.gate, granted=l.granted, reason=l.reason,
        scanned_by=l.scanned_by,
        timestamp=l.timestamp.isoformat() if l.timestamp else None,
    ) for l in logs]

# ── DELIVERY ENDPOINTS ──

@app.get("/api/deliveries", response_model=List[DeliveryOut])
async def get_deliveries(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DeliveryModel).order_by(DeliveryModel.id.desc())
    )
    deliveries = result.scalars().all()
    # Pre-load vehicles for ETA calculation
    vehicle_ids = [d.vehicle_id for d in deliveries]
    vehicles_map = {}
    if vehicle_ids:
        v_result = await db.execute(
            select(VehicleModel).where(VehicleModel.id.in_(vehicle_ids))
        )
        for v in v_result.scalars():
            vehicles_map[v.id] = v
    return [_delivery_to_out(d, vehicles_map.get(d.vehicle_id)) for d in deliveries]

@app.post("/api/deliveries", status_code=201)
async def create_delivery(data: DeliveryCreate, user: UserModel = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    vehicle = await db.get(VehicleModel, data.vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    delivery = DeliveryModel(
        vehicle_id=vehicle.id,
        vehicle_plate=vehicle.plate,
        vehicle_model=vehicle.model,
        driver=vehicle.driver,
        destination_lat=data.destination_lat,
        destination_lng=data.destination_lng,
        destination_name=data.destination_name,
        status="en_route",
        notes=data.notes,
    )
    db.add(delivery)
    await db.commit()
    await db.refresh(delivery)
    return {"message": "Delivery created", "id": delivery.id}

@app.get("/api/deliveries/recent-arrivals", response_model=List[DeliveryOut])
async def get_recent_arrivals(minutes: int = 10, db: AsyncSession = Depends(get_db)):
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    result = await db.execute(
        select(DeliveryModel).where(
            DeliveryModel.status == "arrived",
            DeliveryModel.arrived_at >= cutoff,
        ).order_by(DeliveryModel.arrived_at.desc())
    )
    arrivals = result.scalars().all()
    vehicle_ids = [d.vehicle_id for d in arrivals]
    vehicles_map = {}
    if vehicle_ids:
        v_result = await db.execute(
            select(VehicleModel).where(VehicleModel.id.in_(vehicle_ids))
        )
        for v in v_result.scalars():
            vehicles_map[v.id] = v
    return [_delivery_to_out(d, vehicles_map.get(d.vehicle_id)) for d in arrivals]

@app.get("/api/deliveries/{delivery_id}", response_model=DeliveryOut)
async def get_delivery(delivery_id: int, db: AsyncSession = Depends(get_db)):
    d = await db.get(DeliveryModel, delivery_id)
    if not d:
        raise HTTPException(status_code=404, detail="Delivery not found")
    vehicle = await db.get(VehicleModel, d.vehicle_id)
    return _delivery_to_out(d, vehicle)

@app.put("/api/deliveries/{delivery_id}/confirm")
async def confirm_delivery(delivery_id: int, user: UserModel = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    d = await db.get(DeliveryModel, delivery_id)
    if not d:
        raise HTTPException(status_code=404, detail="Delivery not found")
    if d.status != "arrived":
        raise HTTPException(status_code=400, detail="Delivery must be in 'arrived' status to confirm")
    d.status = "delivered"
    d.delivered_at = datetime.utcnow()
    await db.commit()
    return {"message": "Livraison confirmée", "id": d.id}

# ── YOLO / WEBCAM ENDPOINTS ──

from pydantic import BaseModel

from yolo_service import yolo_service as _yolo_svc

class GateScanRequest(BaseModel):
    gate: str = "Entrée"

@app.post("/api/yolo/scan")
async def yolo_scan(gate_req: GateScanRequest = GateScanRequest(), db: AsyncSession = Depends(get_db)):
    _yolo_svc.start_camera()
    result = _yolo_svc.scan_once()
    plate = result.get("plate")
    image_b64 = result.get("image_b64")
    error = result.get("error")

    if error:
        raise HTTPException(status_code=503, detail=error)

    if not plate:
        return {
            "plate": None,
            "granted": False,
            "reason": "Aucune plaque détectée",
            "image_b64": image_b64,
        }

    gate = gate_req.gate
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    if gate == "Entrée":
        access = await _check_access_for_plate(plate, db)
        await _log_entry(plate, access, image_b64, now, db)
        return {
            "plate": plate,
            "granted": access["granted"],
            "reason": access["reason"],
            "image_b64": image_b64,
            "action": "ENTRY",
        }
    else:
        entry_id = await _log_exit(plate, image_b64, now, db)
        return {
            "plate": plate,
            "granted": True,
            "reason": "Sortie enregistrée",
            "image_b64": image_b64,
            "action": "EXIT",
            "entry_id": entry_id,
        }

@app.get("/api/yolo/status")
async def yolo_status():
    return _yolo_svc.get_status()

@app.post("/api/yolo/start-monitoring")
async def yolo_start_monitoring():
    _yolo_svc.start_camera()
    _yolo_svc.start_monitoring()
    return {"status": "monitoring_started"}

@app.post("/api/yolo/stop-monitoring")
async def yolo_stop_monitoring():
    _yolo_svc.stop_monitoring()
    _yolo_svc.stop_camera()
    return {"status": "monitoring_stopped"}


# ── DEDICATED CAPTURE ENDPOINT ──────────────────────────────────────────────

@app.post("/api/gate/capture", status_code=status.HTTP_201_CREATED)
async def gate_capture(
    gate: str = Form(...),
    image: UploadFile = File(...),
    plate: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    if gate not in ("Entrée", "Sortie"):
        raise HTTPException(status_code=400, detail="Gate must be 'Entrée' or 'Sortie'")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Image vide")

    image_b64 = base64.b64encode(contents).decode()

    detected_plate = plate
    if not detected_plate:
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            p, _ = _yolo_svc.detect_plate_from_frame(frame)
            if p:
                detected_plate = p

    if not detected_plate:
        return {
            "plate": None,
            "granted": False,
            "reason": "Aucune plaque détectée dans l'image",
            "image_b64": image_b64,
        }

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    if gate == "Entrée":
        access = await _check_access_for_plate(detected_plate, db)
        await _log_entry(detected_plate, access, image_b64, now, db)
        return {
            "plate": detected_plate,
            "granted": access["granted"],
            "reason": access["reason"],
            "image_b64": image_b64,
            "action": "ENTRY",
        }
    else:
        entry_id = await _log_exit(detected_plate, image_b64, now, db)
        return {
            "plate": detected_plate,
            "granted": True,
            "reason": "Sortie enregistrée",
            "image_b64": image_b64,
            "action": "EXIT",
            "entry_id": entry_id,
        }


async def _check_access_for_plate(plate: str, db: AsyncSession) -> dict:
    from sqlalchemy import select
    result = await db.execute(
        select(AccessRuleModel).where(
            AccessRuleModel.vehicle_plate == plate.upper(),
            AccessRuleModel.gate == "Entrée",
        )
    )
    rule = result.scalar_one_or_none()
    if rule:
        return {
            "granted": rule.allowed,
            "reason": "Accès autorisé" if rule.allowed else "Accès refusé par règle",
        }
    vehicle = await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate.upper())
    )
    v = vehicle.scalar_one_or_none()
    if v:
        return {"granted": True, "reason": "Véhicule connu, accès autorisé"}
    return {"granted": False, "reason": "Plaque inconnue, accès refusé"}


async def _log_entry(plate: str, access: dict, image_b64: Optional[str], now: str, db: AsyncSession):
    log = AccessLogModel(
        vehicle_plate=plate.upper(),
        action="ENTRY",
        gate="Entrée",
        granted=access["granted"],
        reason=access["reason"],
        image_b64=image_b64,
    )
    db.add(log)

    if access["granted"]:
        vehicle = await db.execute(
            select(VehicleModel).where(VehicleModel.plate == plate.upper())
        )
        v = vehicle.scalar_one_or_none()
        entry = EntryExitModel(
            vehicle_id=v.id if v else 0,
            vehicle_plate=plate.upper(),
            vehicle_model=v.model if v else "Inconnu",
            driver=v.driver if v else None,
            entry_time=now,
            gate="Entrée",
            status="INSIDE",
            image_b64=image_b64,
        )
        db.add(entry)
    await db.commit()


async def _log_exit(plate: str, image_b64: Optional[str], now: str, db: AsyncSession) -> Optional[int]:
    vehicle = await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate.upper())
    )
    v = vehicle.scalar_one_or_none()

    result = await db.execute(
        select(EntryExitModel).where(
            EntryExitModel.vehicle_plate == plate.upper(),
            EntryExitModel.status == "INSIDE",
        ).order_by(EntryExitModel.id.desc()).limit(1)
    )
    entry = result.scalar_one_or_none()
    entry_id = None

    if entry:
        entry.exit_time = now
        entry.status = "OUTSIDE"
        entry.image_b64 = image_b64
        entry_id = entry.id

    log = AccessLogModel(
        vehicle_plate=plate.upper(),
        action="EXIT",
        gate="Sortie",
        granted=True,
        reason="Sortie enregistrée" if entry else "Aucune entrée trouvée",
        image_b64=image_b64,
    )
    db.add(log)
    await db.commit()
    return entry_id


@app.get("/api/entry-exit/{entry_id}/image")
async def get_entry_exit_image(entry_id: int, db: AsyncSession = Depends(get_db)):
    entry = await db.get(EntryExitModel, entry_id)
    if not entry or not entry.image_b64:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"image_b64": entry.image_b64, "vehicle_plate": entry.vehicle_plate}


@app.get("/api/entry-exit/with-images")
async def get_entry_exit_with_images(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EntryExitModel).where(
            EntryExitModel.image_b64.isnot(None)
        ).order_by(EntryExitModel.id.desc()).limit(limit)
    )
    entries = result.scalars().all()
    return [{
        "id": e.id,
        "vehicle_plate": e.vehicle_plate,
        "vehicle_model": e.vehicle_model,
        "entry_time": e.entry_time,
        "exit_time": e.exit_time,
        "gate": e.gate,
        "status": e.status,
        "has_image": e.image_b64 is not None,
    } for e in entries]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
