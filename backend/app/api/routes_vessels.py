from __future__ import annotations

from fastapi import APIRouter

from app.core.vessel_profiles import default_vessel_profile_id, get_vessel_profile, list_vessel_profiles


router = APIRouter(tags=["vessels"])


@router.get("/vessels/profiles")
def list_profiles() -> dict:
    return {
        "default_profile_id": default_vessel_profile_id(),
        "profiles": list_vessel_profiles(),
    }


@router.get("/vessels/profiles/{profile_id}")
def get_profile(profile_id: str) -> dict:
    return {"profile": get_vessel_profile(profile_id).to_json()}
