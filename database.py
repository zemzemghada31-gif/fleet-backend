from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, BigInteger, DateTime, Text, text
from datetime import datetime

DATABASE_URL = "mysql+aiomysql://lora_user:lora_pass@localhost/parking_lora"
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
class Base(DeclarativeBase):
    pass
class DeviceModel(Base):
    __tablename__ = "devices"
    id = Column(String(100), primary_key=True, index=True)
    model = Column(String(100))
    assignment = Column(String(50))
    last_connection = Column(String(50))
    status_color = Column(String(50))
    assigned_vehicle = Column(String(100), default="—")
    assigned_since = Column(String(50), default="—")
class VehicleModel(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(100))
    plate = Column(String(20))
    status = Column(String(50))
    tracker = Column(String(100), default="Not Assigned")
    lat = Column(Float, default=40.7128)
    lng = Column(Float, default=-74.0060)
    speed = Column(Float, default=0.0)
    fuel = Column(Integer, default=80)
    driver = Column(String(100), default="Unassigned")
    eta = Column(String(50), default="—")
    heading = Column(String(10), default="—")
class MaintenanceLogModel(Base):
    __tablename__ = "maintenance_logs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vehicle_id = Column(Integer, index=True)
    date = Column(String(50))
    type = Column(String(50))
    title = Column(String(255))
    description = Column(String(500))
    file = Column(String(255), nullable=True)
    mileage = Column(String(50), nullable=True)
class EntryExitModel(Base):
    __tablename__ = "entry_exit"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vehicle_id = Column(Integer, index=True)
    vehicle_plate = Column(String(20))
    vehicle_model = Column(String(100))
    driver = Column(String(100), nullable=True)
    entry_time = Column(String(50))
    exit_time = Column(String(50), nullable=True)
    gate = Column(String(50), default="Main Gate")
    status = Column(String(20), default="INSIDE")
    notes = Column(String(500), nullable=True)
    image_b64 = Column(Text, nullable=True)
class GPSHistoryModel(Base):
    __tablename__ = "gps_history"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    device_id = Column(String(255))
    lat = Column(Float)
    lng = Column(Float)
    speed = Column(Float)
    heading = Column(String(10))
    fuel = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))
    name = Column(String(255))
    role = Column(String(50), default="operator")
    phone = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AccessRuleModel(Base):
    __tablename__ = "access_rules"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_plate = Column(String(50), index=True)
    vehicle_model = Column(String(100), nullable=True)
    allowed = Column(Boolean, default=True)
    gate = Column(String(50), default="Entrée")
    time_start = Column(String(5), default="00:00")
    time_end = Column(String(5), default="23:59")
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AccessLogModel(Base):
    __tablename__ = "access_logs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vehicle_plate = Column(String(50), index=True)
    action = Column(String(10))  # ENTRY / EXIT
    gate = Column(String(50))
    granted = Column(Boolean)
    reason = Column(String(255), nullable=True)
    scanned_by = Column(Integer, nullable=True)
    image_b64 = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DeliveryModel(Base):
    __tablename__ = "deliveries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, index=True)
    vehicle_plate = Column(String(50))
    vehicle_model = Column(String(100), nullable=True)
    driver = Column(String(100), nullable=True)
    destination_lat = Column(Float)
    destination_lng = Column(Float)
    destination_name = Column(String(255), nullable=True)
    status = Column(String(20), default="en_route")  # en_route / arrived / delivered
    assigned_at = Column(DateTime, default=datetime.utcnow)
    arrived_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    notes = Column(String(500), nullable=True)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for col, dtype in [("assigned_vehicle", "VARCHAR(100) DEFAULT '---'"),
                           ("assigned_since", "VARCHAR(50) DEFAULT '---'")]:
            try:
                await conn.execute(text(f"ALTER TABLE devices ADD COLUMN {col} {dtype}"))
            except Exception:
                pass
        try:
            await conn.execute(text("ALTER TABLE entry_exit ADD COLUMN image_b64 TEXT"))
        except Exception:
            pass
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session