from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT
from .database import engine, SessionLocal
from .models import Base
from .routes import router as main_router
from .admin_routes import router as admin_router
from .services import import_products_from_excel
from .models import Medicine

app = FastAPI()

# 🔥 ADD THIS CORS BLOCK TO ALLOW REACT TO CONNECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Restrict to frontend origin so allow_credentials=True works
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(main_router)  # Chat + core routes
app.include_router(admin_router, prefix="/admin", tags=["admin"])

# Create tables
Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"message": "Pharmacy AI running 🚀"}

def seed_dangerous_drugs(db):
    dangerous_drugs = [
        {
            "name": "Oxycodone", 
            "price": 85.00, 
            "package_size": "10 mg tablets", 
            "description": "A potent Schedule II opioid agonist pain medication used to treat moderate to severe pain. Warning: Highly addictive! Explicitly requires a verified doctor's prescription and OCR scanner clearance.", 
            "stock": 25, 
            "prescription_required": True, 
            "max_safe_dosage": 2
        },
        {
            "name": "Adderall", 
            "price": 120.00, 
            "package_size": "20 mg capsules", 
            "description": "Amphetamine-based stimulant used to treat ADHD and narcolepsy. Warning: Controlled substance. Requires a valid handwritten or digital prescription before dispensing.", 
            "stock": 10, 
            "prescription_required": True, 
            "max_safe_dosage": 2
        }   
    ]
    for d in dangerous_drugs:
        if not db.query(Medicine).filter(Medicine.name == d["name"]).first():
            med = Medicine(**d)
            db.add(med)
    db.commit()

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    import_products_from_excel(db)
    seed_dangerous_drugs(db)
    db.close()