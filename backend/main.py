# -*- coding: utf-8 -*-
"""
FastAPI Backend for Football Prediction Mobile App
Serves predictions via REST API for Android app

@author: Football Prediction Project
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_functions.advanced_model_ensemble import AdvancedEnsemblePredictor, FeatureEngineeringEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Football Prediction API",
    description="Mobile-friendly REST API for Premier League predictions",
    version="1.0.0"
)

# Configure CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for mobile app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATA MODELS ====================

class MatchFeatures(BaseModel):
    """Input features for a single match prediction"""
    av_shots_diff: float
    av_shots_inside_box_diff: float
    av_fouls_diff: float
    av_corners_diff: float
    av_possession_diff: float
    av_pass_accuracy_diff: float
    av_goal_difference: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "av_shots_diff": 2.5,
                "av_shots_inside_box_diff": 1.2,
                "av_fouls_diff": -1.0,
                "av_corners_diff": 0.5,
                "av_possession_diff": 5.0,
                "av_pass_accuracy_diff": 2.5,
                "av_goal_difference": 0.8
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    matches: List[Dict] = None
    fixture_ids: Optional[List[int]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "matches": [
                    {
                        "home_team": "Manchester United",
                        "away_team": "Liverpool",
                        "av_shots_diff": 2.5,
                        "av_shots_inside_box_diff": 1.2,
                        "av_fouls_diff": -1.0,
                        "av_corners_diff": 0.5,
                        "av_possession_diff": 5.0,
                        "av_pass_accuracy_diff": 2.5,
                        "av_goal_difference": 0.8
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response"""
    home_team: str
    away_team: str
    away_win_probability: float
    draw_probability: float
    home_win_probability: float
    predicted_outcome: str
    confidence: float
    fixture_date: Optional[str] = None
    fixture_id: Optional[int] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_matches: int
    predictions: List[PredictionResponse]
    timestamp: str
    model_version: str


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    accuracy: float
    last_updated: str
    algorithm: str
    features_required: int
    classes: List[str]


# ==================== GLOBAL STATE ====================

class PredictionService:
    """Service class to handle predictions"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_info = {
            "name": "Advanced Ensemble Predictor",
            "version": "1.0.0",
            "algorithm": "VotingClassifier (RF + GB + KNN)",
            "features_required": 7,
            "classes": ["Away Win", "Draw", "Home Win"],
            "last_updated": datetime.now().isoformat()
        }
    
    def load_model(self, model_path: str = None):
        """Load pre-trained model"""
        try:
            if model_path is None:
                model_path = Path(__file__).parent.parent / "ml_model_build_random_forest" / "ml_models" / "random_forest_model_10.pk1"
            
            if Path(model_path).exists():
                logger.info(f"Loading model from {model_path}")
                self.model = AdvancedEnsemblePredictor()
                self.model.load_model(str(model_path))
                self.is_loaded = True
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {model_path}. Creating new model...")
                self.model = AdvancedEnsemblePredictor()
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = AdvancedEnsemblePredictor()
            self.is_loaded = False
    
    def predict(self, features: List[float]) -> Dict:
        """
        Make prediction for a single match
        
        Parameters:
        -----------
        features : List[float]
            7 features in order: [shots_diff, shots_box_diff, fouls_diff, 
                                 corners_diff, possession_diff, accuracy_diff, 
                                 goal_diff]
        
        Returns:
        --------
        Dict : Prediction probabilities and outcome
        """
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        try:
            probabilities = self.model.predict_probabilities(X)[0]
            prediction = self.model.predict(X)[0]
            
            # Map prediction to outcome
            outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            
            return {
                "away_win": float(probabilities[0]) * 100,
                "draw": float(probabilities[1]) * 100,
                "home_win": float(probabilities[2]) * 100,
                "predicted_outcome": outcome_map[prediction],
                "confidence": float(np.max(probabilities)) * 100
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise


# Initialize prediction service
prediction_service = PredictionService()


# ==================== ROUTES ====================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Football Prediction API...")
    prediction_service.load_model()
    logger.info("API startup complete")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Football Prediction API",
        "version": "1.0.0",
        "description": "Mobile-friendly REST API for Premier League match predictions",
        "endpoints": {
            "predict": "/predict",
            "batch-predict": "/predict/batch",
            "model-info": "/model/info",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": prediction_service.is_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single_match(
    home_team: str,
    away_team: str,
    features: MatchFeatures
):
    """
    Predict outcome for a single match
    
    Example request:
    ```
    POST /predict?home_team=Manchester%20United&away_team=Liverpool
    {
        "av_shots_diff": 2.5,
        "av_shots_inside_box_diff": 1.2,
        "av_fouls_diff": -1.0,
        "av_corners_diff": 0.5,
        "av_possession_diff": 5.0,
        "av_pass_accuracy_diff": 2.5,
        "av_goal_difference": 0.8
    }
    ```
    
    Returns:
    ```
    {
        "home_team": "Manchester United",
        "away_team": "Liverpool",
        "away_win_probability": 25.5,
        "draw_probability": 20.3,
        "home_win_probability": 54.2,
        "predicted_outcome": "Home Win",
        "confidence": 54.2
    }
    ```
    """
    
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract features in correct order
        feature_list = [
            features.av_shots_diff,
            features.av_shots_inside_box_diff,
            features.av_fouls_diff,
            features.av_corners_diff,
            features.av_possession_diff,
            features.av_pass_accuracy_diff,
            features.av_goal_difference
        ]
        
        # Get prediction
        result = prediction_service.predict(feature_list)
        
        return PredictionResponse(
            home_team=home_team,
            away_team=away_team,
            away_win_probability=round(result["away_win"], 2),
            draw_probability=round(result["draw"], 2),
            home_win_probability=round(result["home_win"], 2),
            predicted_outcome=result["predicted_outcome"],
            confidence=round(result["confidence"], 2)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch_matches(request: BatchPredictionRequest):
    """
    Predict outcomes for multiple matches in batch
    
    Example request:
    ```
    POST /predict/batch
    {
        "matches": [
            {
                "home_team": "Manchester United",
                "away_team": "Liverpool",
                "av_shots_diff": 2.5,
                "av_shots_inside_box_diff": 1.2,
                "av_fouls_diff": -1.0,
                "av_corners_diff": 0.5,
                "av_possession_diff": 5.0,
                "av_pass_accuracy_diff": 2.5,
                "av_goal_difference": 0.8
            },
            {
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "av_shots_diff": 1.5,
                "av_shots_inside_box_diff": 0.8,
                "av_fouls_diff": 0.5,
                "av_corners_diff": -0.3,
                "av_possession_diff": 3.0,
                "av_pass_accuracy_diff": 1.5,
                "av_goal_difference": 0.3
            }
        ]
    }
    ```
    """
    
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.matches or len(request.matches) == 0:
        raise HTTPException(status_code=400, detail="No matches provided")
    
    predictions = []
    
    try:
        for i, match in enumerate(request.matches):
            feature_list = [
                match.get("av_shots_diff", 0),
                match.get("av_shots_inside_box_diff", 0),
                match.get("av_fouls_diff", 0),
                match.get("av_corners_diff", 0),
                match.get("av_possession_diff", 0),
                match.get("av_pass_accuracy_diff", 0),
                match.get("av_goal_difference", 0)
            ]
            
            result = prediction_service.predict(feature_list)
            
            predictions.append(
                PredictionResponse(
                    home_team=match.get("home_team", f"Team_{i}"),
                    away_team=match.get("away_team", f"Team_{i+1}"),
                    away_win_probability=round(result["away_win"], 2),
                    draw_probability=round(result["draw"], 2),
                    home_win_probability=round(result["home_win"], 2),
                    predicted_outcome=result["predicted_outcome"],
                    confidence=round(result["confidence"], 2),
                    fixture_id=match.get("fixture_id")
                )
            )
        
        return BatchPredictionResponse(
            total_matches=len(predictions),
            predictions=predictions,
            timestamp=datetime.now().isoformat(),
            model_version=prediction_service.model_info["version"]
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    
    return ModelInfo(
        model_name=prediction_service.model_info["name"],
        version=prediction_service.model_info["version"],
        accuracy=0.515,  # Update with actual accuracy from training
        last_updated=prediction_service.model_info["last_updated"],
        algorithm=prediction_service.model_info["algorithm"],
        features_required=prediction_service.model_info["features_required"],
        classes=prediction_service.model_info["classes"]
    )


@app.get("/model/features", tags=["Model"])
async def get_features_info():
    """Get information about required features"""
    
    features_info = {
        "total_features": 7,
        "features": [
            {
                "name": "av_shots_diff",
                "description": "Average shots difference (team - opponent)",
                "type": "float",
                "example": 2.5
            },
            {
                "name": "av_shots_inside_box_diff",
                "description": "Average shots inside box difference",
                "type": "float",
                "example": 1.2
            },
            {
                "name": "av_fouls_diff",
                "description": "Average fouls difference",
                "type": "float",
                "example": -1.0
            },
            {
                "name": "av_corners_diff",
                "description": "Average corners difference",
                "type": "float",
                "example": 0.5
            },
            {
                "name": "av_possession_diff",
                "description": "Possession percentage difference",
                "type": "float",
                "example": 5.0
            },
            {
                "name": "av_pass_accuracy_diff",
                "description": "Pass accuracy percentage difference",
                "type": "float",
                "example": 2.5
            },
            {
                "name": "av_goal_difference",
                "description": "Average goals scored difference",
                "type": "float",
                "example": 0.8
            }
        ]
    }
    
    return features_info


# ==================== ERROR HANDLERS ====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    logger.error(f"ValueError: {str(exc)}")
    return {
        "detail": str(exc),
        "error_type": "ValueError"
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "detail": "Internal server error",
        "error_type": type(exc).__name__
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
