from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import os
import traceback

from database import get_db, User, Base, engine
from auth import hash_password, verify_password, create_access_token, decode_access_token

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df_houses = pd.read_csv("housing_cleaned.csv")
df_users = pd.read_csv("survey_cleaned.csv")

if os.path.exists("images"):
    app.mount("/images", StaticFiles(directory="images"), name="images")

weights = {
    'neighborhood': 0.35,
    'monthly_budget': 0.25,
    'security': 0.20,
    'wifi': 0.10,
    'room_type': 0.10,
}

# ── SCHEMAS ───────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    full_name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserPreferences(BaseModel):
    primary_neighborhood: str
    room_type: str
    monthly_budget: int
    security_importance: int
    wifi_required: float

# ── AUTH ENDPOINTS ────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    try:
        existing = db.query(User).filter(User.email == user.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        new_user = User(
            full_name=user.full_name,
            email=user.email,
            hashed_password=hash_password(user.password)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        token = create_access_token(data={"sub": new_user.email})
        return {
            "status": "success",
            "message": "Account created successfully",
            "token": token,
            "user": {
                "full_name": new_user.full_name,
                "email": new_user.email
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        if not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        token = create_access_token(data={"sub": db_user.email})
        return {
            "status": "success",
            "message": "Login successful",
            "token": token,
            "user": {
                "full_name": db_user.full_name,
                "email": db_user.email
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── RECOMMENDATION ENDPOINT ───────────────────────────────────────────────────

def score_listing(user, house):
    score = 0

    if house['availability'] == 0:
        return 0

    if user['primary_neighborhood'] == house['neighborhood']:
        score += 0.35

    if house['monthly_rent'] <= user['monthly_budget']:
        diff_below = user['monthly_budget'] - house['monthly_rent']
        percent_below = diff_below / user['monthly_budget']
        if percent_below <= 0.10:
            score += 0.25
        elif percent_below <= 0.30:
            score += 0.15
        else:
            return 0
    else:
        diff_above = house['monthly_rent'] - user['monthly_budget']
        tolerance = user['monthly_budget'] * 0.20
        if diff_above <= tolerance:
            score += 0.10
        else:
            return 0

    if user['security_importance'] == 2:
        if house['security_score'] >= 4:
            score += 0.20
        elif house['security_score'] == 3:
            score += 0.10
    elif user['security_importance'] == 1:
        if house['security_score'] >= 3:
            score += 0.20

    if user['wifi_required'] == 1 and house['wifi_available'] == 1:
        score += 0.10
    elif user['wifi_required'] == 0.5 and house['wifi_available'] == 1:
        score += 0.05
    elif user['wifi_required'] == 0:
        score += 0.10

    if user['room_type'] == house['room_type']:
        score += 0.10

    return round(score * 100, 1)

def get_top3_recommendations(user, df_houses):
    scores = []

    for _, house in df_houses.iterrows():
        s = score_listing(user, house)
        if s > 0:
            scores.append({
                'listing_id':     house['listing_id'],
                'neighborhood':   house['neighborhood'],
                'city':           house['city'],
                'room_type':      house['room_type'],
                'monthly_rent':   int(house['monthly_rent']),
                'wifi_available': 'Yes' if house['wifi_available'] == 1 else 'No',
                'security_score': int(house['security_score']),
                'latitude':       float(house['latitude']),
                'longitude':      float(house['longitude']),
                'photo_count':    int(house['photo_count']),
                'description':    house['description'],
                'agent_contact':  '+237654546089',
                'match_score':    s
            })

    if not scores:
        return []

    results = sorted(scores, key=lambda x: x['match_score'], reverse=True)
    return results[:3]

@app.get("/")
def home():
    return {"message": "Affordable Housing Recommendation API is running!"}

@app.post("/recommend")
def recommend(preferences: UserPreferences):
    user = {
        'primary_neighborhood': preferences.primary_neighborhood,
        'room_type':            preferences.room_type,
        'monthly_budget':       preferences.monthly_budget,
        'security_importance':  preferences.security_importance,
        'wifi_required':        preferences.wifi_required,
    }

    recommendations = get_top3_recommendations(user, df_houses)

    if not recommendations:
        return {"status": "no_results", "recommendations": []}

    return {"status": "success", "recommendations": recommendations}