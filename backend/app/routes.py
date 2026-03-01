from fastapi import APIRouter, Depends, Query, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from collections import Counter
from typing import List
import os
import time
import uuid
import base64
import base64
from groq import Groq
import fitz  # PyMuPDF

from pydantic import BaseModel
from .database import get_db
from .models import Medicine, Order, RefillAlert, Prescription, Patient, SystemLog
from .services import (
    predict_refill,
    scan_and_generate_refill_alerts,
)
from .agents.orchestrator import run_pharmacy_agent
from .agents.safety_agent import run_safety_checks

# ✅ ONLY ONE ROUTER
router = APIRouter()

# =====================================================
# 🔐 AUTH & REGISTRATION (Email/Password)
# =====================================================
import uuid

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/auth/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    # Check if user exists
    existing = db.query(Patient).filter(Patient.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # In a real app we would hash the password, e.g. bcrypt
    hashed_pwd = data.password + "_hashed" 

    new_patient = Patient(
        id=str(uuid.uuid4()),
        name=data.name,
        email=data.email,
        hashed_password=hashed_pwd,
        is_verified=True
    )
    db.add(new_patient)
    db.commit()
    
    return {
        "status": "success",
        "message": "Registration successful",
        "patient": {"id": new_patient.id, "email": new_patient.email, "name": new_patient.name}
    }

@router.post("/auth/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    
    patient = db.query(Patient).filter(Patient.email == data.email).first()
    
    if not patient:
        raise HTTPException(status_code=400, detail="Invalid email or password")
        
    hashed_pwd = data.password + "_hashed"
    
    if patient.hashed_password != hashed_pwd:
        raise HTTPException(status_code=400, detail="Invalid email or password")
        
    return {
        "status": "success",
        "message": "Login successful",
        "patient": {"id": patient.id, "email": patient.email, "name": patient.name}
    }


# =====================================================
class ChatRequest(BaseModel):
    user_id: str
    message: str


@router.post("/chat")
def chat(data: ChatRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    response = run_pharmacy_agent(db, data.user_id, data.message)
    end_time = time.time()
    
    trace_len = len(response.get("trace", []))
    status = "Verified" if response.get("type") not in ["error", "safety_block"] else "Blocked"
    db.add(SystemLog(
        trace_id=f"RX-{str(uuid.uuid4())[:8].upper()}",
        agent_count=trace_len if trace_len > 0 else 1,
        execution_time=round(end_time - start_time, 2),
        status=status
    ))
    db.commit()
    return response


from pydantic import BaseModel

class QuantityRequest(BaseModel):
    user_id: str
    medicine: str
    quantity: int


from pydantic import BaseModel

class QuantityRequest(BaseModel):
    user_id: str
    medicine: str
    quantity: int


@router.post("/chat/quantity")
def continue_order(data: QuantityRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    response = run_pharmacy_agent(
        db,
        data.user_id,
        f"Order {data.quantity} packs of {data.medicine}"
    )
    end_time = time.time()
    
    trace_len = len(response.get("trace", []))
    status = "Verified" if response.get("type") not in ["error", "safety_block"] else "Blocked"
    db.add(SystemLog(
        trace_id=f"RX-{str(uuid.uuid4())[:8].upper()}",
        agent_count=trace_len if trace_len > 0 else 1,
        execution_time=round(end_time - start_time, 2),
        status=status
    ))
    db.commit()
    return response
# =====================================================
# 📦 ADMIN ROUTING: REFILL STOCK
# =====================================================
class RefillStockRequest(BaseModel):
    medicine_name: str
    amount: int

@router.post("/admin/refill-stock")
def refill_stock(data: RefillStockRequest, db: Session = Depends(get_db)):
    med = db.query(Medicine).filter(Medicine.name == data.medicine_name).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medicine not found")
    med.stock += data.amount
    db.commit()
    return {"status": "success", "message": f"Added {data.amount} to {data.medicine_name}", "new_stock": med.stock}
# =====================================================
@router.get("/search")
def search_medicines(query: str = Query(..., min_length=2), db: Session = Depends(get_db)):

    results = db.query(Medicine).filter(
        or_(
            Medicine.name.ilike(f"%{query}%"),
            Medicine.description.ilike(f"%{query}%")
        )
    ).limit(5).all()

    return [
        {
            "id": med.id,
            "name": med.name,
            "price": med.price,
            "stock": med.stock,
            "prescription_required": med.prescription_required
        }
        for med in results
    ]


# =====================================================
# 📦 PRODUCTS (STORE FRONT)
# =====================================================
@router.get("/products")
def get_products(db: Session = Depends(get_db)):

    medicines = db.query(Medicine).all()

    return [
        {
            "id": m.id,
            "name": m.name,
            "price": m.price,
            "stock": m.stock,
            "prescription_required": m.prescription_required,
            "max_safe_dosage": m.max_safe_dosage
        }
        for m in medicines
    ]


# =====================================================
# 📦 FINALIZE CHECKOUT (CONFIRM ORDER)
# =====================================================

from pydantic import BaseModel

class CartItem(BaseModel):
    name: str
    quantity: int
    confirmed_overdose: bool = False

class CheckoutRequest(BaseModel):
    patient_id: str
    phone: Optional[str] = None
    items: List[CartItem]

@router.post("/finalize-checkout")
def finalize_checkout(data: CheckoutRequest, db: Session = Depends(get_db)):

    for item in data.items:

        medicine = db.query(Medicine).filter(Medicine.name == item.name).first()

        if not medicine:
            raise HTTPException(status_code=404, detail=f"{item.name} not found")

        # 1️⃣ Check stock and Auto-Restock
        if medicine.stock < item.quantity:
            print(f"📦 [RESTOCK] {medicine.name} is insufficient for order. Automatically ordering 100 units from Retailer...")
            medicine.stock += 100
            db.commit()
            if medicine.stock < item.quantity:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for {item.name}")

        # 2️⃣ Prescription check
        if medicine.prescription_required:
            has_rx = db.query(Prescription).filter(
                Prescription.patient_id == data.patient_id,
                Prescription.medicine_name == medicine.name,
                Prescription.approved == True
            ).first()
            if not has_rx:
                raise HTTPException(status_code=403, detail=f"{item.name} requires prescription. Please upload one first.")

        # 3️⃣ Strict Safety Overdosage check
        max_safe = medicine.max_safe_dosage or 10
        if item.quantity > max_safe * 3:
            raise HTTPException(status_code=400, detail=f"Quantity {item.quantity} for {item.name} is dangerously unsafe and completely blocked by Backend.")
            
        if item.quantity > max_safe and not item.confirmed_overdose:
            raise HTTPException(status_code=400, detail=f"Quantity {item.quantity} exceeds safe dosage of {max_safe}. Explicit confirmation required.")

        safety = run_safety_checks(db, data.patient_id, medicine.name)

        if safety["status"] == "blocked":
            raise HTTPException(status_code=403, detail="Safety rule blocked this purchase")

        # 4️⃣ Deduct stock
        medicine.stock -= item.quantity

        # 5️⃣ Create order record
        new_order = Order(
            patient_id=data.patient_id,
            product_name=medicine.name,
            quantity=item.quantity,
            dosage_frequency=1
        )

        db.add(new_order)

    db.commit()

    # 6️⃣ Send WhatsApp Confirmation if phone provided
    whatsapp_status = "Skipped"
    if data.phone and len(data.phone.strip()) > 5:
        import urllib.parse
        try:
            # We use an open/free WhatsApp messaging API for dev demonstrations
            # CallMeBot allows sending messages by sending a simple GET request
            clean_phone = "".join(c for c in data.phone if c.isdigit() or c == "+")
            if not clean_phone.startswith("+"):
                clean_phone = "+" + clean_phone 
                
            item_list_str = "\\n".join([f"- {i.quantity}x {i.name}" for i in data.items])
            total_items = sum(i.quantity for i in data.items)
            msg = f"*PharmaAgent: Order Confirmed!* ✅\\n\\nYour prescription order for {total_items} items has been successfully processed and audited.\\n\\n*Items:*\\n{item_list_str}\\n\\nThanks for using PharmaAgent!"
            
            # Use Ultramsg / CallMeBot or generic free API form
            # Because this is a hackathon/demo, we use an open HTTP WhatsApp gateway simulator if an actual twilio/meta key isn't provided
            # Using CallMeBot developer API as a placeholder trigger
            encoded_msg = urllib.parse.quote(msg)
            
            # NOTE: For CallMeBot to work, the user needs to get their free API key from the bot, 
            # so we'll simulate the successful HTTP request logic here.
            wa_url = f"https://api.callmebot.com/whatsapp.php?phone={clean_phone}&text={encoded_msg}&apikey=YOUR_API_KEY"
            
            # Fire and forget (in a real app, use celery/background tasks)
            requests.get(wa_url, timeout=3)
            whatsapp_status = "Attempted"
            print(f"✅ WhatsApp Confirmation triggered for {clean_phone}")
            
        except Exception as e:
            print("WhatsApp Integration Error:", e)
            whatsapp_status = f"Failed: {str(e)}"

    return {
        "status": "success",
        "message": "Checkout completed safely",
        "whatsapp_status": whatsapp_status
    }


# =====================================================
# 📊 USER ORDER HISTORY
# =====================================================
@router.get("/user/orders/{user_id}")
def get_user_orders(user_id: str, db: Session = Depends(get_db)):

    orders = db.query(Order).filter(
        Order.patient_id == user_id
    ).order_by(Order.purchase_date.desc()).all()

    return [
        {
            "product": order.product_name,
            "quantity": order.quantity,
            "purchase_date": order.purchase_date
        }
        for order in orders
    ]


# =====================================================
# 🔔 REFILL SYSTEM
# =====================================================
@router.get("/admin/refill/{user_id}")
def refill_alert(user_id: str, db: Session = Depends(get_db)):
    return predict_refill(db, user_id)


@router.post("/admin/scan-refills")
def scan_refills(db: Session = Depends(get_db)):
    alerts = scan_and_generate_refill_alerts(db)
    return {
        "message": "Refill scan completed",
        "alerts_generated": alerts
    }


@router.get("/admin/refill-alerts")
def get_refill_alerts(db: Session = Depends(get_db)):

    alerts = db.query(RefillAlert).all()

    return [
        {
            "patient_id": a.patient_id,
            "medicine": a.medicine_name,
            "expected_run_out": a.expected_run_out.strftime("%Y-%m-%d")
        }
        for a in alerts
    ]


class EmailTestRequest(BaseModel):
    user_id: str
    medicine_name: str

@router.post("/admin/test-refill-email")
def test_refill_email(data: EmailTestRequest, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == data.user_id).first()
    email = patient.email if patient else data.user_id
    
    msg = f"📧 [MOCK EMAIL to {email}] Hello! Your {data.medicine_name} is running low. Please log in to PharmaAgent to repurchase."
    print(msg)
    
    return {
        "status": "success",
        "message": f"Sent mock email to {email} for {data.medicine_name}"
    }

# =====================================================
# 📦 INVENTORY SYSTEM
# =====================================================
@router.get("/admin/inventory")
def get_inventory(db: Session = Depends(get_db)):

    medicines = db.query(Medicine).all()

    return [
        {
            "id": med.id,
            "name": med.name,
            "stock": med.stock,
            "price": med.price
        }
        for med in medicines
    ]


@router.get("/admin/low-stock")
def low_stock(threshold: int = 10, db: Session = Depends(get_db)):

    medicines = db.query(Medicine).filter(Medicine.stock <= threshold).all()

    return [
        {
            "name": med.name,
            "stock": med.stock
        }
        for med in medicines
    ]


@router.get("/debug/stock/{product_name}")
def debug_stock(product_name: str, db: Session = Depends(get_db)):

    product = db.query(Medicine).filter(
        Medicine.name.ilike(f"%{product_name}%")
    ).first()

    if not product:
        return {"status": "not_found"}

    return {
        "product": product.name,
        "current_stock": product.stock
    }


# =====================================================
# 📄 PRESCRIPTION UPLOAD
# =====================================================
UPLOAD_DIR = "uploaded_prescriptions"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/upload-prescription/{user_id}/{medicine_name}")
async def upload_prescription(
    user_id: str,
    medicine_name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    file_location = f"{UPLOAD_DIR}/{user_id}_{medicine_name}_{file.filename}"

    content = await file.read()
    with open(file_location, "wb") as f:
        f.write(content)
        
    extracted_text = ""
    approved = True

    import requests
    try:
        if file.content_type in ["image/jpeg", "image/png", "image/jpg", "application/pdf"]:
            # 1. Extract Text using OCR Space Engine 2 (Highly Trained Handwritten OCR)
            with open(file_location, 'rb') as f:
                r = requests.post(
                    'https://api.ocr.space/parse/image',
                    files={'file': f},
                    data={
                        'apikey': 'helloworld',
                        'language': 'eng',
                        'OCREngine': '2',
                        'scale': 'true'
                    }
                )
            ocr_result = r.json()
            
            if ocr_result.get("IsErroredOnProcessing"):
                raise Exception(f"OCR Parsing Error: {ocr_result.get('ErrorMessage')}")
                
            parsed_results = ocr_result.get("ParsedResults", [])
            extracted_text = "\n".join([p.get("ParsedText", "") for p in parsed_results]).strip()
            
            if not extracted_text:
                raise Exception("No text could be extracted from this image/pdf.")

            # 2. Semantic Verification using Llama 3 8B Text Model
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            prompt = f"You are a medical verification system. I have extracted the following raw OCR text from a medical prescription:\n\n\"{extracted_text}\"\n\nDoes this prescription explicitly prescribe or mention the medicine: '{medicine_name}'? Please account for slight OCR misspellings. Answer ONLY with 'YES' or 'NO'."
            
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            ai_decision = completion.choices[0].message.content.strip().upper()
            
            if "YES" not in ai_decision and medicine_name != "Unknown":
                approved = False  # Strictly reject if not found
                print(f"Rx Check Failed: '{medicine_name}' rejected by LLM based on OCR text: {extracted_text}")
            else:
                approved = True
                print(f"Rx Check Passed or Generic Upload.")
    except Exception as e:
        print("OCR Vision Error:", e)
        raise HTTPException(status_code=400, detail=f"OCR Engine Error: {str(e)}")

    prescription = Prescription(
        patient_id=user_id,
        medicine_name=medicine_name,
        file_path=file_location,
        extracted_text=extracted_text,
        approved=approved
    )

    db.add(prescription)
    db.commit()

    if not approved:
        raise HTTPException(status_code=400, detail="Prescription rejected. The specified medicine was not clearly identified in the handwritten text.")

    return {
        "message": "Prescription analyzed and verified successfully.",
        "file_path": file_location,
        "extracted_text": extracted_text
    }

# =====================================================
# 📚 ADMIN TRACES API
# =====================================================
@router.get("/admin/traces")
def get_system_logs(db: Session = Depends(get_db)):
    logs = db.query(SystemLog).order_by(SystemLog.created_at.desc()).limit(15).all()
    return [
        {
            "id": log.id,
            "trace_id": log.trace_id,
            "agent_count": log.agent_count,
            "execution_time": log.execution_time,
            "status": log.status,
            "created_at": log.created_at.isoformat()
        }
        for log in logs
    ]


# =====================================================
# 🚚 WAREHOUSE WEBHOOK
# =====================================================
@router.post("/webhook/warehouse")
def warehouse_webhook(payload: dict):
    print("📦 Warehouse received order:", payload)
    return {"status": "warehouse_notified"}