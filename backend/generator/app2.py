import torch
import torch.nn as nn
import cv2
import numpy as np
import timm
from torchvision import transforms
import json
import os
import gc
from collections import defaultdict, Counter
import time
import whisper
from rapidfuzz import fuzz
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import shutil
import subprocess
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
import os
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
MONGODB_URL = "mongodb://localhost:27017"  # Change to your MongoDB URL
client = AsyncIOMotorClient(MONGODB_URL)
db = client["football_highlights"]
matches_collection = db["matches"]

# Base directories
BASE_MEDIA_DIR = Path("media")
BASE_MEDIA_DIR.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model1_path': 'models/best_swin_small_model_CALF.pth',
    'model1_name': 'swin_small_patch4_window7_224',
    'model1_type': 'enhanced',
    'model2_path': 'models/best_video_swin_model_20_epoch.pth',
    'model2_name': 'swin_small_patch4_window7_224',
    'model2_type': 'video',
    'num_classes': 5,
    'num_frames': 60,
    'img_size': 224,
    'dropout_model1': 0.4,
    'dropout_model2': 0.3,
    'whisper_model': 'base',
    'video_path': 'try.mp4',  # Will be overridden
    'output_final_json': 'final_verified_events.json',  # Will be overridden
    'stride': 45,
    'confidence_threshold': 0.55,
    'min_model_agreement': 1,
    'audio_verification_required': True,
    'audio_can_override': True,
    'audio_can_add_events': True,
    'min_audio_confidence_to_add': 0.90,
    'audio_match_window': 30.0,
    'fuzzy_match_threshold': 75,
    'min_keyword_matches': 5,
    'min_high_confidence_matches': 1,
    'tag_replays': True,
    'min_event_gap': 45.0,
    'temporal_grouping_window': 4.0,
    'prevent_trim_overlap': True,
    'min_clip_gap': 2.0,
    'high_confidence_bypass': 0.90,
    'trim_window_before': 10.0,
    'trim_window_after': 10.0,
    'prefer_earliest_detection': True,
    'replay_exclusion_window': 30.0,
    'enable_sequence_validation': True,
    'max_consecutive_gap': 5.0,
    'use_audio_for_sequence_resolution': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    'batch_size': 4,
    'frame_skip': 3,
    'class_names': ['foul', 'goal', 'freekick', 'penalty', 'corner']
}

# ============================================================================
# PROGRESS TRACKING
# ============================================================================
class ProgressTracker:
    def __init__(self):
        self.progress = {}
        self.websockets = {}
    
    def update(self, match_id: str, status: str, progress: float, message: str = ""):
        self.progress[match_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def broadcast(self, match_id: str):
        if match_id in self.websockets:
            for ws in self.websockets[match_id]:
                try:
                    await ws.send_json(self.progress[match_id])
                except:
                    pass

progress_tracker = ProgressTracker()

# ============================================================================
# CHUNKED VIDEO PROCESSOR
# ============================================================================
class ChunkedVideoProcessor:
    """Process long videos in manageable chunks"""
    
    def __init__(self, config, chunk_duration_minutes=50):
        self.config = config
        self.chunk_duration = chunk_duration_minutes * 60  # Convert to seconds
        self.max_memory_mb = 4000  # Maximum memory per chunk (4GB)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata without loading frames"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'width': width,
            'height': height
        }
    
    def calculate_chunks(self, duration: float) -> List[Tuple[float, float]]:
        """Calculate chunk time ranges"""
        chunks = []
        current_time = 0
        
        while current_time < duration:
            end_time = min(current_time + self.chunk_duration, duration)
            chunks.append((current_time, end_time))
            current_time = end_time
        
        print(f"\n{'='*60}")
        print(f"VIDEO CHUNKING STRATEGY")
        print(f"{'='*60}")
        print(f"Total duration: {duration/60:.1f} minutes")
        print(f"Chunk size: {self.chunk_duration/60:.1f} minutes")
        print(f"Number of chunks: {len(chunks)}")
        print(f"{'='*60}\n")
        
        return chunks
    
    def process_chunk(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: float,
        chunk_idx: int,
        total_chunks: int,
        detector: 'RobustMultiModalDetector'
    ) -> Tuple[List, List]:
        """Process a single video chunk"""
        
        print(f"\n{'='*60}")
        print(f"PROCESSING CHUNK {chunk_idx + 1}/{total_chunks}")
        print(f"Time range: {start_time/60:.1f}m - {end_time/60:.1f}m")
        print(f"{'='*60}")
        
        # Monitor memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory before chunk: {mem_before:.1f} MB")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Seek to start position
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        model1_events = []
        model2_events = []
        
        frame_buffer = []
        frame_idx_buffer = []
        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply frame skip
            if (current_frame - start_frame) % self.config['frame_skip'] == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.config['img_size'], self.config['img_size']))
                frame_buffer.append(frame)
                frame_idx_buffer.append(current_frame)
            
            # Process when buffer is full
            if len(frame_buffer) >= self.config['num_frames'] * 3:
                events_m1, events_m2 = detector._process_frame_chunk(
                    frame_buffer, frame_idx_buffer, fps
                )
                model1_events.extend(events_m1)
                model2_events.extend(events_m2)
                
                # Keep overlap
                overlap = self.config['num_frames'] // 2
                frame_buffer = frame_buffer[-overlap:]
                frame_idx_buffer = frame_idx_buffer[-overlap:]
                
                # Memory check
                mem_current = process.memory_info().rss / 1024 / 1024
                if mem_current > self.max_memory_mb:
                    print(f"⚠️ High memory usage: {mem_current:.1f} MB - forcing cleanup")
                    gc.collect()
                    if detector.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            current_frame += 1
            
            # Progress update
            chunk_progress = ((current_frame - start_frame) / (end_frame - start_frame)) * 100
            if int(chunk_progress) % 10 == 0:
                print(f"Chunk {chunk_idx + 1} progress: {chunk_progress:.1f}%", end='\r')
        
        # Process remaining frames
        if len(frame_buffer) >= self.config['num_frames']:
            events_m1, events_m2 = detector._process_frame_chunk(
                frame_buffer, frame_idx_buffer, fps
            )
            model1_events.extend(events_m1)
            model2_events.extend(events_m2)
        
        cap.release()
        
        # Aggressive cleanup
        del frame_buffer, frame_idx_buffer
        gc.collect()
        if detector.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"\nMemory after chunk: {mem_after:.1f} MB (Δ {mem_after - mem_before:+.1f} MB)")
        print(f"✓ Chunk {chunk_idx + 1} complete: M1={len(model1_events)}, M2={len(model2_events)}")
        print(f"{'='*60}\n")
        
        return model1_events, model2_events
    
    def merge_chunk_events(self, all_events: List[List], overlap_time: float = 5.0) -> List:
        """Merge events from multiple chunks, removing duplicates at boundaries"""
        if not all_events:
            return []
        
        merged = []
        
        for chunk_events in all_events:
            if not merged:
                merged.extend(chunk_events)
                continue
            
            # Check for overlapping events at chunk boundary
            for event in chunk_events:
                # Skip if too close to last event from previous chunk
                is_duplicate = False
                for existing in merged[-10:]:  # Only check recent events
                    if (event['event_name'] == existing['event_name'] and
                        abs(event['timestamp'] - existing['timestamp']) < overlap_time):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged.append(event)
        
        return merged

# ============================================================================
# FOOTBALL EVENT SEQUENCE RULES (UPDATED WITH CAUSALITY PROTECTION)
# ============================================================================
SEQUENCE_RULES = {
    'foul': {
        'can_follow': ['foul', 'penalty', 'freekick', 'goal', 'corner'],
        'can_cause': ['penalty', 'freekick'],  # Foul CAUSES these
        'description': 'Foul can cause penalty/freekick; audio should not change foul to these',
        'preserves_identity': True,
        'can_be_consecutive': True,
        'consecutive_events': ['penalty', 'freekick'],
        'time_constraints': {
            'penalty': 10.0,
            'freekick': 10.0
        }
    },
    'penalty': {
        'can_follow': ['goal', 'foul', 'penalty', 'freekick', 'corner'],
        'can_cause': ['goal'],  # Penalty CAUSES goal
        'description': 'Penalty can cause goal; audio should not change penalty to goal',
        'preserves_identity': True,
        'can_be_consecutive': True,
        'consecutive_events': ['goal', 'foul'],
        'time_constraints': {
            'goal': 30.0,
            'foul': None
        }
    },
    'freekick': {
        'can_follow': ['goal', 'foul', 'penalty', 'freekick', 'corner'],
        'can_cause': ['goal'],  # Freekick CAUSES goal
        'description': 'Freekick can cause goal; audio should not change freekick to goal',
        'preserves_identity': True,
        'can_be_consecutive': True,
        'consecutive_events': ['goal'],
        'time_constraints': {
            'goal': 30.0
        }
    },
    'goal': {
        'can_follow': ['foul', 'penalty', 'freekick', 'goal', 'corner'],
        'can_cause': [],  # Goal doesn't cause other events
        'description': 'Goal can be followed by any event',
        'preserves_identity': True,
        'can_be_consecutive': False,
        'consecutive_events': [],
        'time_constraints': {}
    },
    'corner': {
        'can_follow': ['goal', 'foul', 'penalty', 'freekick', 'corner'],
        'can_cause': ['goal'],  # Corner can cause goal
        'description': 'Corner can cause goal; audio should not change corner to goal',
        'preserves_identity': True,
        'can_be_consecutive': False,
        'consecutive_events': ['goal'],
        'time_constraints': {
            'goal': 30.0
        }
    }
}

# ============================================================================
# ENHANCED COMMENTARY DICTIONARY WITH WEIGHTS + REPLAY DETECTION
# ============================================================================
COMMENTARY_DICT = {
    "goal": {
        "high_confidence": [
            "goal!", "he scores!", "back of the net!", "finds the net",
            "it's a goal", "scores", "what a goal", "brilliant goal",
            "into the net", "scores a goal", "goal scored"
        ],
        "medium_confidence": [
            "that's in!", "what a finish!", "incredible strike!",
            "they take the lead", "equalizer!", "amazing goal",
            "beautiful goal", "clinical finish"
        ],
        "low_confidence": [
            "shot", "attempt", "striker", "forward"
        ]
    },
    "penalty": {
        "high_confidence": [
            "penalty", "spot kick", "penalty kick", "from the penalty spot",
            "points to the spot", "penalty awarded", "penalty given",
            "penalty decision", "referee points", "penalty box"
        ],
        "medium_confidence": [
            "fouled in the box", "foul in the area", "penalty appeal",
            "penalty shout", "twelve yards", "from the spot"
        ],
        "low_confidence": [
            "in the box", "area", "penalty area"
        ]
    },
    "freekick": {
        "high_confidence": [
            "free kick", "freekick", "set piece", "direct free kick",
            "free kick awarded", "takes the free kick", "free kick taken",
            "dangerous free kick"
        ],
        "medium_confidence": [
            "foul", "referee awards", "dangerous position",
            "free kick opportunity", "set piece opportunity"
        ],
        "low_confidence": [
            "stopped", "halted", "whistle"
        ]
    },
    "corner": {
        "high_confidence": [
            "corner", "corner kick", "from the corner", "corner flag",
            "corner awarded", "corner given", "corner taken",
            "corner ball"
        ],
        "medium_confidence": [
            "in-swinging", "out-swinger", "corner opportunity",
            "short corner", "corner routine"
        ],
        "low_confidence": [
            "wide", "out of play"
        ]
    },
    "foul": {
        "high_confidence": [
            "foul", "foul committed", "yellow card", "red card",
            "booking", "referee blows", "foul play", "free kick for foul",
            "cautioned", "sent off"
        ],
        "medium_confidence": [
            "illegal tackle", "bad challenge", "reckless",
            "dangerous play", "tackle", "carded"
        ],
        "low_confidence": [
            "challenge", "contact", "goes down"
        ]
    }
}
REPLAY_KEYWORDS = [
    "replay", "slow motion", "another look", "different angle", "let's see that again"
]

# ============================================================================
# EVENT-LEVEL CONFIDENCE RULES
# ============================================================================
EVENT_CONFIDENCE_RULES = {
    'goal': {
        'ultra_high_threshold': 0.90,
        'requires_audio': False,
        'min_audio_matches': 4,  # Lower requirement
        'min_high_confidence_matches': 2,  # Lower requirement
        'description': 'Goals with 90%+ confidence added directly'
    },
    'corner': {
        'ultra_high_threshold': 0.90,
        'requires_audio': False,
        'min_audio_matches': 4,  # Lower requirement
        'min_high_confidence_matches': 2,  # Lower requirement
        'description': 'Corners with 90%+ confidence added directly'
    },
    'foul': {
        'ultra_high_threshold': 0.90,
        'requires_audio': True,  # Always needs audio
        'min_audio_matches': 5,
        'min_high_confidence_matches': 2,
        'description': 'Fouls always require audio verification'
    },
    'freekick': {
        'ultra_high_threshold': 0.90,
        'requires_audio': True,  # Always needs audio
        'min_audio_matches': 5,
        'min_high_confidence_matches': 2,
        'description': 'Freekicks always require audio verification'
    },
    'penalty': {
        'ultra_high_threshold': 0.90,
        'requires_audio': False,
        'min_audio_matches': 4,
        'min_high_confidence_matches': 2,
        'description': 'Penalties with 90%+ confidence added directly'
    }
}

# ============================================================================
# SEQUENCE VALIDATOR
# ============================================================================
class FootballSequenceValidator:
    """Validates and prunes events based on football game rules"""

    def __init__(self, config):
        self.config = config
        self.rules = SEQUENCE_RULES

    def validate_sequence(self, events, transcription, audio_analyzer):
        """
        Validate entire event sequence and prune invalid transitions.
        Ensures valid sequences like foul->freekick->goal are preserved.
        """
        if not self.config['enable_sequence_validation']:
            return events

        if len(events) <= 1:
            return events

        print(f"\n{'='*60}")
        print("SEQUENCE VALIDATION & PRUNING")
        print(f"{'='*60}")
        print(f"Input events: {len(events)}")

        validated_events = []
        pruned_count = 0

        i = 0
        while i < len(events):
            current_event = events[i]

            # First event always valid
            if i == 0:
                validated_events.append(current_event)
                print(f" ✓ Event {len(validated_events)}: {current_event['final_event']} at {current_event['timestamp']}s [FIRST EVENT]")
                i += 1
                continue

            previous_event = validated_events[-1]
            time_gap = current_event['timestamp'] - previous_event['timestamp']

            # Check if current event can follow previous event
            is_valid, reason = self._is_valid_transition(
                previous_event, current_event, transcription, audio_analyzer
            )

            if is_valid:
                # Special case: Check for freekick->goal sequence
                if (current_event['final_event'] == 'goal' and
                    previous_event['final_event'] == 'freekick' and
                    time_gap <= self.config['max_consecutive_gap']):
                    # Ensure both events are kept if audio supports both
                    audio_analysis = audio_analyzer.analyze_audio_for_event(
                        current_event['timestamp'],
                        transcription,
                        window=self.config['audio_match_window']
                    )
                    if audio_analysis and audio_analysis['event_type'] == 'goal':
                        # Check if freekick is also supported in the same window
                        prev_audio = audio_analyzer.analyze_audio_for_event(
                            previous_event['timestamp'],
                            transcription,
                            window=self.config['audio_match_window']
                        )
                        if prev_audio and prev_audio['event_type'] == 'freekick':
                            validated_events.append(current_event)
                            print(f" ✓ Event {len(validated_events)}: {current_event['final_event']} at {current_event['timestamp']}s [VALID: Freekick->Goal sequence confirmed]")
                        else:
                            # If no freekick support, keep goal only
                            validated_events.append(current_event)
                            print(f" ✓ Event {len(validated_events)}: {current_event['final_event']} at {current_event['timestamp']}s [VALID: Goal confirmed, no freekick support]")
                    else:
                        pruned_count += 1
                        print(f" ✗ Event {i+1}: {current_event['final_event']} at {current_event['timestamp']}s [PRUNED: No audio support for goal]")
                else:
                    validated_events.append(current_event)
                    print(f" ✓ Event {len(validated_events)}: {current_event['final_event']} at {current_event['timestamp']}s [VALID: {reason}]")
            else:
                pruned_count += 1
                print(f" ✗ Event {i+1}: {current_event['final_event']} at {current_event['timestamp']}s [PRUNED: {reason}]")

            i += 1

        print(f"\n✓ Validation complete")
        print(f"✓ Events after validation: {len(validated_events)}")
        print(f"✓ Events pruned: {pruned_count}")
        print(f"{'='*60}\n")

        return validated_events

    def _is_valid_transition(self, prev_event, curr_event, transcription, audio_analyzer):
        """
        Check if current event can validly follow previous event WITH TIME CONSTRAINTS.
        Returns (is_valid, reason)
        """
        prev_type = prev_event['final_event']
        curr_type = curr_event['final_event']
        time_gap = curr_event['timestamp'] - prev_event['timestamp']

        # Get allowed following events for previous event type
        allowed_following = self.rules[prev_type]['can_follow']

        # Check if current event is in allowed list
        if curr_type not in allowed_following:
            return False, f"{prev_type} cannot be followed by {curr_type}"

        # Check TIME CONSTRAINTS for this transition
        time_constraints = self.rules[prev_type].get('time_constraints', {})
        if curr_type in time_constraints:
            max_gap = time_constraints[curr_type]
            if max_gap is not None and time_gap > max_gap:
                return True, f"Valid sequence: {prev_type}->{curr_type} (gap: {time_gap:.1f}s, beyond immediate window)"

        # Check if events are too close (possible duplicate/misclassification)
        if time_gap < 2.0 and curr_type == prev_type:
            return False, f"Duplicate {curr_type} within 2s"

        # Special handling for consecutive events (foul->penalty, freekick->goal, etc.)
        if curr_type in self.rules[prev_type].get('consecutive_events', []):
            if time_gap <= self.config['max_consecutive_gap']:
                # Use audio to verify if both events are real
                if self.config['use_audio_for_sequence_resolution'] and transcription:
                    return self._audio_verify_consecutive_events(
                        prev_event, curr_event, transcription, audio_analyzer
                    )
                else:
                    return True, f"Valid consecutive: {prev_type}->{curr_type}"

        # Standard validation passed
        return True, f"Valid sequence: {prev_type}->{curr_type} (gap: {time_gap:.1f}s)"

    def _audio_verify_consecutive_events(self, prev_event, curr_event, transcription, audio_analyzer):
        """
        Use audio analysis to verify if consecutive events are both real
        or if one should be pruned.
        """
        prev_type = prev_event['final_event']
        curr_type = curr_event['final_event']

        # Analyze audio for both events
        prev_audio = audio_analyzer.analyze_audio_for_event(
            prev_event['timestamp'], transcription, window=self.config['audio_match_window']
        )
        curr_audio = audio_analyzer.analyze_audio_for_event(
            curr_event['timestamp'], transcription, window=self.config['audio_match_window']
        )

        # If both have strong audio support, keep both
        if prev_audio and curr_audio:
            if (prev_audio['high_confidence_matches'] >= 2 and
                curr_audio['high_confidence_matches'] >= 2):
                return True, f"Both events audio-verified: {prev_type}->{curr_type}"

        # If current event has no audio support but previous does
        if prev_audio and not curr_audio:
            if prev_audio['high_confidence_matches'] >= 2:
                return False, f"Current event lacks audio support (likely duplicate)"

        # If previous has weak audio but current has strong audio
        if curr_audio and (not prev_audio or prev_audio.get('total_matches', 0) < 3):
            if curr_audio['high_confidence_matches'] >= 2:
                return True, f"Current event has stronger audio evidence"

        # Default: allow if gap is reasonable
        time_gap = curr_event['timestamp'] - prev_event['timestamp']
        if time_gap >= 2.0:
            return True, f"Sufficient time gap: {time_gap:.1f}s"

        return False, f"Insufficient evidence for consecutive events"

    def get_sequence_explanation(self, event_type):
        """Get human-readable explanation of sequence rules for an event"""
        if event_type in self.rules:
            rule = self.rules[event_type]
            return f"{event_type.upper()}: {rule['description']}"
        return f"{event_type}: No specific rules"

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
class ContextAwareTemporalModule(nn.Module):
    def __init__(self, feature_dim, num_frames, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.pyramid_1 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.pyramid_2 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.pyramid_3 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.pyramid_4 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 4, kernel_size=11, padding=5),
            nn.BatchNorm1d(feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        p1 = self.pyramid_1(x_t)
        p2 = self.pyramid_2(x_t)
        p3 = self.pyramid_3(x_t)
        p4 = self.pyramid_4(x_t)
        pyramid_out = torch.cat([p1, p2, p3, p4], dim=1)
        fused = self.fusion(pyramid_out)
        return fused.transpose(1, 2)

class EnhancedSwinSmallTransformer(nn.Module):
    def __init__(self, model_name, num_classes, num_frames=60, dropout=0.4):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable=True)
        self.feature_dim = self.backbone.num_features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.context_temporal = ContextAwareTemporalModule(self.feature_dim, num_frames, dropout)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim), nn.GELU()
        )
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.feature_dim // 2,
                            num_layers=2, batch_first=True, bidirectional=True,
                            dropout=dropout * 0.5 if dropout > 0 else 0)
        attention_heads = 8
        while self.feature_dim % attention_heads != 0:
            attention_heads -= 1
        self.temporal_attention = nn.MultiheadAttention(embed_dim=self.feature_dim,
                                                       num_heads=attention_heads,
                                                       dropout=dropout * 0.5, batch_first=True)
        self.frame_weights = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(self.feature_dim // 4, 1), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim * 2), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(self.feature_dim * 2, self.feature_dim), nn.GELU(), nn.Dropout(dropout * 0.6),
            nn.Linear(self.feature_dim, self.feature_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.4),
            nn.Linear(self.feature_dim // 2, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        if len(features.shape) == 4:
            if features.shape[1] == self.feature_dim:
                features = self.spatial_pool(features).squeeze(-1).squeeze(-1)
            else:
                features = features.mean(dim=[1, 2])
        features = features.view(batch_size, num_frames, self.feature_dim)
        context_features = self.context_temporal(features)
        temp_conv_out = self.temporal_conv(context_features.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(temp_conv_out)
        attended_features, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        combined = features + context_features + temp_conv_out + lstm_out + attended_features
        frame_importance = self.frame_weights(combined)
        weighted_features = combined * frame_importance
        avg_pool = weighted_features.mean(dim=1)
        max_pool, _ = weighted_features.max(dim=1)
        video_features = avg_pool + 0.3 * max_pool
        return self.classifier(video_features)

class VideoSwinTransformer(nn.Module):
    def __init__(self, model_name, num_classes, num_frames=60, dropout=0.3):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        self.feature_dim = self.backbone.num_features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5)
        )
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.feature_dim // 2,
                            num_layers=2, batch_first=True, bidirectional=True,
                            dropout=dropout * 0.5 if dropout > 0 else 0)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8,
                                                       dropout=dropout * 0.5, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(self.feature_dim // 2, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = features.mean(dim=[1, 2])
        features = features.view(batch_size, num_frames, -1)
        temp_conv_out = self.temporal_conv(features.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(temp_conv_out)
        attended_features, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        combined_features = lstm_out + attended_features
        video_features = combined_features.mean(dim=1)
        return self.classifier(video_features)

# ============================================================================
# AUDIO ANALYSIS MODULE
# ============================================================================
class RobustCommentaryAnalyzer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        print(f"\n{'='*60}")
        print("AUDIO ANALYSIS SETUP")
        print(f"{'='*60}")
        print(f"Loading Whisper model: {config['whisper_model']}")
        try:
            self.model = whisper.load_model(config['whisper_model'], device=self.device)
            print(f"✓ Whisper loaded | Fuzzy threshold: {config['fuzzy_match_threshold']}%")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{config['whisper_model']}': {str(e)}")
        print(f"✓ Audio window: ±{config['audio_match_window']}s")
        print(f"✓ Min keyword matches required: {config['min_keyword_matches']}")
        print(f"✓ Min high-conf matches required: {config['min_high_confidence_matches']}")
        print(f"{'='*60}\n")

    def transcribe_video(self, video_path):
        # Existing transcription logic (unchanged)
        print(f"\n{'='*60}")
        print("TRANSCRIBING AUDIO")
        print(f"{'='*60}")
        import time
        start_time = time.time()
        try:
            result = self.model.transcribe(video_path, language='en', verbose=False, word_timestamps=True)
            elapsed = time.time() - start_time
            print(f"✓ Complete in {elapsed:.1f}s | Segments: {len(result['segments'])}")
            print(f"{'='*60}\n")
            transcription_data = {
                'language': result.get('language', 'unknown'),
                'full_text': result['text'].strip(),
                'segments': []
            }
            for segment in result['segments']:
                transcription_data['segments'].append({
                    'start': round(segment['start'], 2),
                    'end': round(segment['end'], 2),
                    'text': segment['text'].strip(),
                    'confidence': round(1 - segment.get('no_speech_prob', 0), 4)
                })
            return transcription_data
        except Exception as e:
            print(f"⚠ Error: {e}")
            return None

    def analyze_audio_for_event(self, timestamp, transcription, window=15.0):
        if not transcription or not transcription['segments']:
            return None

        search_start = max(0, timestamp - window)
        search_end = timestamp + window
        relevant_segments = []
        for segment in transcription['segments']:
            if segment['start'] <= search_end and segment['end'] >= search_start:
                relevant_segments.append(segment)

        if not relevant_segments:
            print(f"No audio segments found for timestamp {timestamp}s")
            return None

        replay_info = self._detect_replay(relevant_segments)
        if replay_info and replay_info['confidence'] > 0.75:
            print(f"Replay detected at {timestamp}s with confidence {replay_info['confidence']:.2f}")
            return None  # Skip events during strong replay detection

        event_scores = {}
        event_matches = {event: [] for event in CONFIG['class_names']}
        for event_type in CONFIG['class_names']:
            scores = {'high': 0, 'medium': 0, 'low': 0}
            for segment in relevant_segments:
                text_lower = segment['text'].lower()
                # Stricter matching for high-confidence keywords
                for keyword in COMMENTARY_DICT[event_type]['high_confidence']:
                    score = fuzz.partial_ratio(keyword.lower(), text_lower)
                    if score >= self.config['fuzzy_match_threshold']:
                        scores['high'] += 1
                        event_matches[event_type].append({
                            'keyword': keyword,
                            'score': score,
                            'text': segment['text'],
                            'level': 'high',
                            'time_offset': abs(segment['start'] - timestamp),
                            'timestamp': segment['start']
                        })
                for keyword in COMMENTARY_DICT[event_type]['medium_confidence']:
                    score = fuzz.partial_ratio(keyword.lower(), text_lower)
                    if score >= self.config['fuzzy_match_threshold'] + 5:
                        scores['medium'] += 1
                        event_matches[event_type].append({
                            'keyword': keyword,
                            'score': score,
                            'text': segment['text'],
                            'level': 'medium',
                            'time_offset': abs(segment['start'] - timestamp),
                            'timestamp': segment['start']
                        })
                for keyword in COMMENTARY_DICT[event_type]['low_confidence']:
                    score = fuzz.partial_ratio(keyword.lower(), text_lower)
                    if score >= self.config['fuzzy_match_threshold'] + 15:
                        scores['low'] += 1
            total_score = scores['high'] * 3 + scores['medium'] * 2 + scores['low'] * 1
            total_matches = scores['high'] + scores['medium'] + scores['low']
            event_scores[event_type] = {
                'total_score': total_score,
                'total_matches': total_matches,
                'high_matches': scores['high'],
                'medium_matches': scores['medium'],
                'low_matches': scores['low'],
                'matches': event_matches[event_type]
            }

        best_event = max(event_scores.items(), key=lambda x: x[1]['total_score'])
        if (best_event[1]['total_matches'] < self.config['min_keyword_matches'] or
            best_event[1]['high_matches'] < self.config['min_high_confidence_matches']):
            print(f"Insufficient audio evidence for {best_event[0]} at {timestamp}s: "
                  f"{best_event[1]['total_matches']} matches, {best_event[1]['high_matches']} high")
            return None

        max_possible_score = len(relevant_segments) * 3
        audio_confidence = min(best_event[1]['total_score'] / max(max_possible_score, 1), 1.0)
        result = {
            'event_type': best_event[0],
            'audio_confidence': round(audio_confidence, 4),
            'total_matches': best_event[1]['total_matches'],
            'high_confidence_matches': best_event[1]['high_matches'],
            'medium_confidence_matches': best_event[1]['medium_matches'],
            'low_confidence_matches': best_event[1]['low_matches'],
            'all_scores': {k: v['total_score'] for k, v in event_scores.items()},
            'matched_segments': relevant_segments,
            'best_matches': sorted(best_event[1]['matches'],
                                   key=lambda x: (x['level'] == 'high', x['score']),
                                   reverse=True)[:5],
            'is_replay': False
        }

        if replay_info:
            result['is_replay'] = True
            result['replay_confidence'] = replay_info['confidence']
            result['replay_keywords'] = replay_info['keywords']
            result['replay_segments'] = replay_info['segments']

        return result

    def _detect_replay(self, segments):
        replay_matches = []
        for segment in segments:
            text_lower = segment['text'].lower()
            for keyword in REPLAY_KEYWORDS:
                score = fuzz.partial_ratio(keyword.lower(), text_lower)
                if score >= 85:  # Stricter threshold for replays
                    replay_matches.append({
                        'keyword': keyword,
                        'score': score,
                        'text': segment['text'],
                        'timestamp': segment['start']
                    })
        if len(replay_matches) >= 1:
            confidence = max([m['score'] for m in replay_matches]) / 100.0
            return {
                'confidence': confidence,
                'keywords': [m['keyword'] for m in replay_matches],
                'segments': [m['text'] for m in replay_matches]
            }
        return None

# ============================================================================
# MULTIMODAL FUSION ENGINE (UPDATED)
# ============================================================================
# UPDATED: Remove sequence validation, strict audio verification
class MultimodalFusionEngine:
    def __init__(self, config):
        self.config = config
        # Remove sequence validator - no longer needed
        # self.sequence_validator = FootballSequenceValidator(config)

    def fuse_detections(self, model1_events, model2_events, transcription, audio_analyzer):
        print(f"\n{'='*60}")
        print("MULTIMODAL FUSION & VERIFICATION (EVENT-LEVEL RULES)")
        print(f"{'='*60}")
        print(f"Model 1 detections: {len(model1_events)}")
        print(f"Model 2 detections: {len(model2_events)}")

        merged_events = self._merge_model_detections(model1_events, model2_events)
        print(f"Merged video detections: {len(merged_events)}")

        verified_events = []
        corrected_count = 0
        rejected_count = 0
        replay_tagged_count = 0
        audio_overridden_count = 0
        high_confidence_bypassed = 0
        ultra_high_confidence_direct = 0
        event_level_bypassed = 0
        foul_to_penalty_protected = 0
        foul_to_goal_removed = 0  # NEW COUNTER

        for event in merged_events:
            predicted_event = event['predicted_event']
            video_conf = event['video_confidence']

            # Get event-specific rules
            event_rules = EVENT_CONFIDENCE_RULES.get(predicted_event, {
                'ultra_high_threshold': 0.90,
                'requires_audio': True,
                'min_audio_matches': 5,
                'min_high_confidence_matches': 2
            })

            # NEW: Event-level ultra high confidence check
            if video_conf >= event_rules['ultra_high_threshold']:
                if not event_rules['requires_audio']:
                    # Events like goal, corner, penalty can be added directly at 90%+
                    event['final_event'] = predicted_event
                    event['final_confidence'] = video_conf
                    event['audio_analysis'] = None
                    event['verification_status'] = 'event_level_ultra_high_confidence'
                    event['is_replay'] = False
                    verified_events.append(event)
                    event_level_bypassed += 1
                    print(f" ✓ Event-level bypass: {predicted_event.upper()} at {event['timestamp']}s "
                          f"(conf: {video_conf:.2f}) [NO AUDIO REQUIRED]")
                    continue
                else:
                    # Events like foul, freekick need audio even at 90%+
                    print(f" ⚠️ High-conf {predicted_event.upper()} at {event['timestamp']}s "
                          f"(conf: {video_conf:.2f}) - checking audio (event rule)")

            # High-confidence events (0.85-0.89) - check audio
            if video_conf >= self.config['high_confidence_bypass']:
                audio_analysis = None
                if transcription:
                    audio_analysis = audio_analyzer.analyze_audio_for_event(
                        event['timestamp'],
                        transcription,
                        window=self.config['audio_match_window']
                    )

                if audio_analysis:
                    audio_event = audio_analysis.get('event_type', 'unknown')

                    # NEW: Remove foul if audio detects goal
                    if predicted_event == 'foul' and audio_event == 'goal':
                        foul_to_goal_removed += 1
                        rejected_count += 1
                        print(f" ❌ REMOVED FOUL at {event['timestamp']}s "
                              f"(audio detected goal - foul removed)")
                        continue

                    # NEW: Protect foul from being changed to penalty
                    if predicted_event == 'foul' and audio_event == 'penalty':
                        event['final_event'] = 'foul'
                        event['final_confidence'] = video_conf
                        event['audio_analysis'] = audio_analysis
                        event['verification_status'] = 'foul_protected_from_penalty'
                        event['audio_suggested'] = 'penalty'
                        event['is_replay'] = audio_analysis.get('is_replay', False)
                        verified_events.append(event)
                        foul_to_penalty_protected += 1
                        print(f" 🛡️ Protected FOUL at {event['timestamp']}s "
                              f"(audio suggested penalty but keeping foul)")
                        continue

                    # Check if audio strongly contradicts video
                    if (audio_event != predicted_event and
                        audio_analysis.get('high_confidence_matches', 0) >= 3 and
                        audio_analysis.get('total_matches', 0) >= 5):
                        # Audio overrides even high-confidence video
                        event['final_event'] = audio_event
                        event['corrected_from'] = predicted_event
                        event['correction_reason'] = 'audio_override_strong_evidence'
                        event['final_confidence'] = audio_analysis.get('audio_confidence', 0.5)
                        event['audio_analysis'] = audio_analysis
                        event['is_replay'] = audio_analysis.get('is_replay', False)
                        verified_events.append(event)
                        audio_overridden_count += 1
                        print(f" 🔄 Audio override: {event['corrected_from']} → {event['final_event']} at {event['timestamp']}s "
                              f"(video conf: {video_conf:.2f}, audio matches: {audio_analysis.get('total_matches', 0)})")
                    else:
                        # High confidence with audio agreement or weak audio
                        event['final_event'] = predicted_event
                        event['final_confidence'] = video_conf
                        event['audio_analysis'] = audio_analysis
                        event['verification_status'] = 'high_confidence_with_audio'
                        event['is_replay'] = audio_analysis.get('is_replay', False)
                        verified_events.append(event)
                        high_confidence_bypassed += 1
                        print(f" ✓ High-conf with audio: {predicted_event} at {event['timestamp']}s "
                              f"(conf: {video_conf:.2f})")
                else:
                    # High confidence but NO audio - reject
                    rejected_count += 1
                    print(f" ✗ Rejected (no audio): {predicted_event} at {event['timestamp']}s "
                          f"(conf: {video_conf:.2f})")
                continue

            # Skip low confidence detections early
            if video_conf < self.config['confidence_threshold']:
                print(f" ✗ Skipping low confidence at {event['timestamp']}s")
                rejected_count += 1
                continue

            # AUDIO VERIFICATION with event-specific requirements
            audio_analysis = None
            if transcription:
                audio_analysis = audio_analyzer.analyze_audio_for_event(
                    event['timestamp'],
                    transcription,
                    window=self.config['audio_match_window']
                )

            if not audio_analysis:
                # NO AUDIO = REJECT EVENT
                rejected_count += 1
                print(f" ✗ Rejected (no audio evidence): {predicted_event} at {event['timestamp']}s")
                continue

            # Use event-specific minimum requirements
            required_matches = event_rules['min_audio_matches']
            required_high_conf = event_rules['min_high_confidence_matches']

            # Check minimum match requirements
            if audio_analysis.get('total_matches', 0) < required_matches:
                rejected_count += 1
                print(f" ✗ Insufficient matches ({audio_analysis.get('total_matches', 0)}/{required_matches}) "
                      f"for {predicted_event} at {event['timestamp']}s")
                continue

            if audio_analysis.get('high_confidence_matches', 0) < required_high_conf:
                rejected_count += 1
                print(f" ✗ Insufficient high-conf matches "
                      f"({audio_analysis.get('high_confidence_matches', 0)}/{required_high_conf}) "
                      f"for {predicted_event} at {event['timestamp']}s")
                continue

            # Check for replay
            is_replay = audio_analysis.get('is_replay', False)
            if is_replay and audio_analysis.get('replay_confidence', 0) > 0.75:
                print(f" 🚫 Skipping strong replay at {event['timestamp']}s")
                rejected_count += 1
                continue

            original_event = predicted_event
            audio_event = audio_analysis.get('event_type', 'unknown')

            # NEW: Remove foul if audio detects goal (for normal confidence events too)
            if original_event == 'foul' and audio_event == 'goal':
                foul_to_goal_removed += 1
                rejected_count += 1
                print(f" ❌ REMOVED FOUL at {event['timestamp']}s "
                      f"(audio detected goal - foul removed)")
                continue

            # NEW: Protect foul from being changed to penalty (for normal confidence events too)
            if original_event == 'foul' and audio_event == 'penalty':
                event['final_event'] = 'foul'
                event['final_confidence'] = video_conf
                event['audio_analysis'] = audio_analysis
                event['verification_status'] = 'foul_protected_from_penalty'
                event['audio_suggested'] = 'penalty'
                event['is_replay'] = is_replay
                verified_events.append(event)
                foul_to_penalty_protected += 1
                print(f" 🛡️ Protected FOUL at {event['timestamp']}s "
                      f"(audio suggested penalty but keeping foul)")
                continue

            # AUDIO MATCHING with event-specific rules
            if audio_event == original_event:
                # Perfect match - keep event
                event['final_event'] = original_event
                event['confidence_boost'] = 'audio_agreement'
                event['final_confidence'] = min(video_conf * 1.2, 1.0)
                event['audio_analysis'] = audio_analysis
                event['is_replay'] = is_replay
                verified_events.append(event)
                status = "Replay + " if is_replay else ""
                print(f" ✓ {status}Verified: {original_event} at {event['timestamp']}s "
                      f"({audio_analysis.get('total_matches', 0)} matches, {audio_analysis.get('high_confidence_matches', 0)} high)")

            elif audio_analysis.get('high_confidence_matches', 0) >= 2:
                # Audio says something different with strong evidence - CHANGE EVENT TYPE
                event['final_event'] = audio_event
                event['corrected_from'] = original_event
                event['correction_reason'] = 'audio_correction_strong_evidence'
                event['final_confidence'] = audio_analysis.get('audio_confidence', 0.5)
                event['audio_analysis'] = audio_analysis
                event['is_replay'] = is_replay
                verified_events.append(event)
                corrected_count += 1
                status = "Replay + " if is_replay else ""
                print(f" 🔄 {status}Corrected: {original_event} → {audio_event} at {event['timestamp']}s "
                      f"(audio matches: {audio_analysis.get('total_matches', 0)}, high: {audio_analysis.get('high_confidence_matches', 0)})")

            else:
                # Weak audio evidence - REJECT EVENT
                rejected_count += 1
                print(f" ✗ Rejected (weak audio, mismatch): video={original_event}, audio={audio_event} "
                      f"at {event['timestamp']}s")

        # Add audio-only events
        if self.config['audio_can_add_events'] and transcription:
            audio_only_events = self._find_audio_only_events(
                verified_events, transcription, audio_analyzer
            )
            print(f"Audio-only events found: {len(audio_only_events)}")
            verified_events.extend(audio_only_events)

        # Filter out any None values before processing
        verified_events = [e for e in verified_events if e is not None]
        
        # Filter replays and duplicates
        verified_events = self._filter_replays_and_select_primary(verified_events)

        # Prevent trim overlaps
        if self.config.get('prevent_trim_overlap', True):
            verified_events = self._prevent_trim_overlaps(verified_events)

        print(f"✓ Final verified events: {len(verified_events)}")
        print(f"✓ Event-level bypassed (goal/corner/penalty 90%+): {event_level_bypassed}")
        print(f"🛡️ Foul protected from penalty: {foul_to_penalty_protected}")
        print(f"❌ Foul removed (audio=goal): {foul_to_goal_removed}")  # NEW
        print(f"✓ High-conf with audio check: {high_confidence_bypassed}")
        print(f"✓ Corrected by audio: {corrected_count}")
        print(f"✓ Audio overridden: {audio_overridden_count}")
        print(f"✗ Rejected (no/weak audio): {rejected_count}")
        print(f"🔄 Replays tagged: {replay_tagged_count}")
        print(f"{'='*60}\n")

        return verified_events

    def _filter_replays_and_select_primary(self, events):
        """Filter replays and remove duplicates - KEEP ONLY EVENTS WITH STRONG AUDIO"""
        if not events:
            return []

        # Filter out None values from the events list
        events = [e for e in events if e is not None]
        
        if not events:
            return []

        print(f"\n{'='*60}")
        print("FILTERING REPLAYS & DUPLICATES (STRICT MODE)")
        print(f"{'='*60}")

        events.sort(key=lambda x: x['timestamp'])
        filtered_events = []

        for event in events:
            # Safety check - skip if event is None or invalid
            if not event or not isinstance(event, dict):
                print(f" ⚠️ Skipping invalid event: {event}")
                continue
                
            is_replay = event.get('is_replay', False)

            # Skip strong replays
            if is_replay and event.get('replay_info', {}).get('confidence', 0) > 0.75:
                print(f" 🚫 Skipping strong replay at {event['timestamp']}s")
                continue

            # Check for duplicate events (same type, close in time)
            duplicate = False
            replace_index = -1
            
            for idx, existing in enumerate(filtered_events):
                # Safety check for existing event
                if not existing or not isinstance(existing, dict):
                    continue
                    
                time_diff = abs(event['timestamp'] - existing['timestamp'])

                if (event['final_event'] == existing['final_event'] and
                    time_diff < self.config['min_event_gap']):

                    # Keep the one with better audio evidence
                    # audio_analysis may be None; guard by falling back to empty dict
                    event_audio_score = (event.get('audio_analysis') or {}).get('total_matches', 0)
                    existing_audio_score = (existing.get('audio_analysis') or {}).get('total_matches', 0)

                    if event_audio_score > existing_audio_score:
                        replace_index = idx
                        print(f" ⚠ Replacing with better audio: "
                              f"{existing['timestamp']}s → {event['timestamp']}s "
                              f"(audio: {existing_audio_score} → {event_audio_score})")
                        break
                    else:
                        duplicate = True
                        print(f" 🚫 Skipping duplicate (weaker audio): {event['timestamp']}s")
                        break

            # Replace if we found a better event
            if replace_index >= 0:
                filtered_events[replace_index] = event
            elif not duplicate:
                # Remove replay flag for primary events
                if is_replay:
                    event['is_replay'] = False
                    if 'replay_info' in event:
                        del event['replay_info']
                    print(f" ✓ Accepting event at {event['timestamp']}s")
                filtered_events.append(event)

        removed_count = len(events) - len(filtered_events)
        if removed_count > 0:
            print(f"✓ Removed {removed_count} replay/duplicate events")
        print(f"✓ Final event count: {len(filtered_events)}")
        print(f"{'='*60}\n")

        return filtered_events

    def _merge_model_detections(self, model1_events, model2_events):
        """Merge detections from both models"""
        merged = []
        time_threshold = 3.0
        all_events = model1_events + model2_events
        all_events.sort(key=lambda x: x['timestamp'])
        used = set()

        for i, event1 in enumerate(all_events):
            if i in used:
                continue
            group = [event1]
            group_indices = [i]

            for j, event2 in enumerate(all_events[i+1:], start=i+1):
                if abs(event2['timestamp'] - event1['timestamp']) <= time_threshold:
                    group.append(event2)
                    group_indices.append(j)
                elif event2['timestamp'] - event1['timestamp'] > time_threshold:
                    break

            used.update(group_indices)

            event_types = [e['event_name'] for e in group]
            event_counter = Counter(event_types)
            most_common_event = event_counter.most_common(1)[0][0]

            group_sorted = sorted(group, key=lambda x: x['timestamp'])
            middle_idx = len(group_sorted) // 2
            middle_event = group_sorted[middle_idx]
            selected_timestamp = middle_event['timestamp']
            selected_frame = middle_event['frame']

            merged_event = {
                'timestamp': round(selected_timestamp, 2),
                'predicted_event': most_common_event,
                'video_confidence': round(np.mean([e['confidence'] for e in group]), 4),
                'model_agreement': len([e for e in group if e['event_name'] == most_common_event]) / len(group),
                'num_models_detected': len(group),
                'all_predictions': event_types,
                'frame': selected_frame
            }
            merged.append(merged_event)

        return merged

    def _find_audio_only_events(self, existing_events, transcription, audio_analyzer):
        """Find events detected only by audio with STRICT requirements"""
        if not transcription:
            return []

        audio_events = []
        existing_timestamps = [e['timestamp'] for e in existing_events]

        # Scan through transcription segments
        for i in range(0, len(transcription['segments']), 5):
            segment_group = transcription['segments'][i:i+10]
            if not segment_group:
                continue

            timestamp = segment_group[len(segment_group)//2]['start']

            # Skip if too close to existing event
            if any(abs(timestamp - t) < self.config['min_event_gap'] for t in existing_timestamps):
                continue

            audio_analysis = audio_analyzer.analyze_audio_for_event(
                timestamp, transcription, window=self.config['audio_match_window']
            )

            if not audio_analysis:
                continue

            # Check for replay
            is_replay = audio_analysis.get('is_replay', False)
            if is_replay and audio_analysis.get('replay_confidence', 0) > 0.75:
                continue

            # STRICT requirements for audio-only events (higher than normal)
            if (audio_analysis['high_confidence_matches'] >= self.config['min_high_confidence_matches'] + 1 and
                audio_analysis['total_matches'] >= self.config['min_keyword_matches'] + 2 and
                audio_analysis['audio_confidence'] >= self.config['min_audio_confidence_to_add']):

                audio_event = {
                    'timestamp': round(timestamp, 2),
                    'predicted_event': 'none_detected',
                    'video_confidence': 0.0,
                    'final_event': audio_analysis['event_type'],
                    'final_confidence': audio_analysis['audio_confidence'],
                    'source': 'audio_only',
                    'audio_analysis': audio_analysis,
                    'num_models_detected': 0,
                    'model_agreement': 0.0,
                    'frame': int(timestamp * 30),
                    'is_replay': False
                }

                audio_events.append(audio_event)
                existing_timestamps.append(timestamp)
                print(f" + Audio-only: {audio_analysis['event_type']} at {timestamp}s "
                      f"(matches: {audio_analysis['total_matches']}, high: {audio_analysis['high_confidence_matches']}, "
                      f"conf: {audio_analysis['audio_confidence']:.2f})")

        return audio_events

    def _prevent_trim_overlaps(self, events):
        """Prevent trim window overlaps between consecutive events"""
        if not events or len(events) <= 1:
            return events

        print(f"\n{'='*60}")
        print("PREVENTING TRIM WINDOW OVERLAPS")
        print(f"{'='*60}")

        events.sort(key=lambda x: x['timestamp'])
        adjusted_count = 0

        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]

            current_end = current['timestamp'] + current.get('custom_trim_after', self.config['trim_window_after'])
            next_start = max(0, next_event['timestamp'] - next_event.get('custom_trim_before', self.config['trim_window_before']))

            if current_end > next_start:
                overlap = current_end - next_start
                midpoint = (current_end + next_start) / 2

                if 'trim_adjusted' not in current:
                    current['trim_adjusted'] = True
                    current['original_trim_after'] = self.config['trim_window_after']

                if 'trim_adjusted' not in next_event:
                    next_event['trim_adjusted'] = True
                    next_event['original_trim_before'] = self.config['trim_window_before']

                new_current_after = midpoint - current['timestamp']
                new_next_before = next_event['timestamp'] - midpoint

                min_gap = self.config.get('min_clip_gap', 2.0)
                new_current_after = max(new_current_after - min_gap/2, 3.0)
                new_next_before = max(new_next_before - min_gap/2, 3.0)

                current['custom_trim_after'] = new_current_after
                next_event['custom_trim_before'] = new_next_before

                adjusted_count += 1
                print(f" ⚠️ Adjusted overlap between events at {current['timestamp']}s and {next_event['timestamp']}s")
                print(f" Overlap: {overlap:.1f}s → Gap: {min_gap:.1f}s")

        if adjusted_count > 0:
            print(f"✓ Adjusted {adjusted_count} overlapping trim windows")
        else:
            print(f"✓ No overlapping trim windows detected")

        print(f"{'='*60}\n")

        return events

# ============================================================================
# MAIN DETECTOR
# ============================================================================
class RobustMultiModalDetector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        print(f"{'='*60}")
        print("LOADING MODELS")
        print(f"{'='*60}")
        print(f"Loading Model 1 (Enhanced Swin)...")
        self.model1 = EnhancedSwinSmallTransformer(
            config['model1_name'], config['num_classes'],
            config['num_frames'], config['dropout_model1']
        ).to(self.device)
        checkpoint1 = torch.load(config['model1_path'], map_location=self.device)
        self.model1.load_state_dict(checkpoint1['model_state_dict'])
        self.model1.eval()
        print("✓ Model 1 loaded")
        print(f"Loading Model 2 (Video Swin)...")
        self.model2 = VideoSwinTransformer(
            config['model2_name'], config['num_classes'],
            config['num_frames'], config['dropout_model2']
        ).to(self.device)
        checkpoint2 = torch.load(config['model2_path'], map_location=self.device)
        self.model2.load_state_dict(checkpoint2['model_state_dict'])
        self.model2.eval()
        print("✓ Model 2 loaded")
        print(f"{'='*60}\n")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.audio_analyzer = RobustCommentaryAnalyzer(config)
        self.fusion_engine = MultimodalFusionEngine(config)

    def load_frames_generator(self, video_path):
        """Generator to load frames efficiently without storing all in memory"""
        cap = cv2.VideoCapture(video_path)
        actual_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if actual_frame % self.config['frame_skip'] == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.config['img_size'], self.config['img_size']))
                yield frame, actual_frame
            actual_frame += 1
        cap.release()

    def preprocess_batch(self, frames):
        processed = []
        for frame in frames:
            frame_tensor = self.to_tensor(frame)
            frame_tensor = self.normalize(frame_tensor)
            processed.append(frame_tensor)
        return torch.stack(processed)

    def process_video_with_both_models(self, video_path):
        print(f"\n{'='*60}")
        print("PROCESSING VIDEO WITH BOTH MODELS")
        print(f"{'='*60}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Video FPS: {fps:.2f} | Total frames: {total_frames}")
        model1_events = []
        model2_events = []
        chunk_size = self.config['num_frames'] * 3
        frame_buffer = []
        frame_idx_buffer = []
        print(f"\nRunning inference (chunk size: {chunk_size} frames)...\n")
        for frame, frame_idx in self.load_frames_generator(video_path):
            frame_buffer.append(frame)
            frame_idx_buffer.append(frame_idx)
            if len(frame_buffer) >= chunk_size:
                events_m1, events_m2 = self._process_frame_chunk(
                    frame_buffer, frame_idx_buffer, fps
                )
                model1_events.extend(events_m1)
                model2_events.extend(events_m2)
                overlap = self.config['num_frames'] // 2
                frame_buffer = frame_buffer[-overlap:]
                frame_idx_buffer = frame_idx_buffer[-overlap:]
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% | M1: {len(model1_events)} | M2: {len(model2_events)}", end='\r')
                if len(model1_events) % 50 == 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
        if len(frame_buffer) >= self.config['num_frames']:
            events_m1, events_m2 = self._process_frame_chunk(
                frame_buffer, frame_idx_buffer, fps
            )
            model1_events.extend(events_m1)
            model2_events.extend(events_m2)
        print(f"\n✓ Video processing complete")
        del frame_buffer, frame_idx_buffer
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return model1_events, model2_events, fps

    def _process_frame_chunk(self, frames, frame_indices, fps):
        """Process a chunk of frames efficiently"""
        model1_events = []
        model2_events = []
        with torch.no_grad():
            current_frame = 0
            while current_frame + self.config['num_frames'] <= len(frames):
                window_frames = frames[current_frame:current_frame + self.config['num_frames']]
                video_tensor = self.preprocess_batch(window_frames).unsqueeze(0).to(self.device)
                middle_idx = current_frame + self.config['num_frames'] // 2
                if middle_idx >= len(frame_indices):
                    middle_idx = len(frame_indices) - 1
                actual_frame = frame_indices[middle_idx]
                timestamp = actual_frame / fps
                with torch.amp.autocast('cuda', enabled=self.config['mixed_precision']):
                    logits1 = self.model1(video_tensor)
                    logits2 = self.model2(video_tensor)
                probs1 = torch.softmax(logits1, dim=1)
                conf1, pred1 = torch.max(probs1, dim=1)
                probs2 = torch.softmax(logits2, dim=1)
                conf2, pred2 = torch.max(probs2, dim=1)
                if conf1.item() >= self.config['confidence_threshold']:
                    model1_events.append({
                        'timestamp': round(timestamp, 2),
                        'event_name': self.config['class_names'][pred1.item()],
                        'confidence': round(conf1.item(), 4),
                        'frame': int(actual_frame),
                        'model': 'model1'
                    })
                if conf2.item() >= self.config['confidence_threshold']:
                    model2_events.append({
                        'timestamp': round(timestamp, 2),
                        'event_name': self.config['class_names'][pred2.item()],
                        'confidence': round(conf2.item(), 4),
                        'frame': int(actual_frame),
                        'model': 'model2'
                    })
                current_frame += self.config['stride']
        return model1_events, model2_events

    def save_raw_detections(self, model1_events, model2_events, fps):
        """Save raw model detections to separate JSON for verification"""
        raw_output = {
            'video_info': {
                'path': self.config['video_path'],
                'fps': round(fps, 2)
            },
            'model1_detections': {
                'total': len(model1_events),
                'events': sorted(model1_events, key=lambda x: x['timestamp'])
            },
            'model2_detections': {
                'total': len(model2_events),
                'events': sorted(model2_events, key=lambda x: x['timestamp'])
            },
            'event_counts': {
                'model1': dict(Counter([e['event_name'] for e in model1_events])),
                'model2': dict(Counter([e['event_name'] for e in model2_events]))
            }
        }

        raw_json_path = self.config['output_final_json'].replace('.json', '_raw_detections.json')
        with open(raw_json_path, 'w') as f:
            json.dump(raw_output, f, indent=2)

        print(f"✓ Raw detections saved to: {raw_json_path}")
        return raw_json_path

    def save_final_results(self, verified_events, fps):
        """Save final results WITHOUT sequence information"""
        verified_events.sort(key=lambda x: x['timestamp'])
        final_output = {
            'video_info': {
                'path': self.config['video_path'],
                'fps': round(fps, 2)
            },
            'fusion_settings': {
                'models_used': 2,
                'audio_verification': 'STRICT_REQUIRED',
                'audio_can_override': self.config['audio_can_override'],
                'audio_can_add_events': self.config['audio_can_add_events'],
                'min_audio_confidence_to_add': self.config['min_audio_confidence_to_add'],
                'audio_match_window': self.config['audio_match_window'],
                'min_keyword_matches': self.config['min_keyword_matches'],
                'min_high_confidence_matches': self.config['min_high_confidence_matches'],
                'min_event_gap': self.config['min_event_gap'],
                'trim_window_before': self.config['trim_window_before'],
                'trim_window_after': self.config['trim_window_after'],
                'sequence_validation_enabled': False  # DISABLED
            },
            'summary': {
                'total_verified_events': len(verified_events),
                'audio_corrected': len([e for e in verified_events if 'corrected_from' in e]),
                'audio_only': len([e for e in verified_events if e.get('source') == 'audio_only']),
                'video_only': 0,  # No video-only events in strict mode
                'sequence_validated': False
            },
            'verified_events': []
        }

        for i, event in enumerate(verified_events, 1):
            mins = int(event['timestamp'] // 60)
            secs = int(event['timestamp'] % 60)
            trim_start = max(0, event['timestamp'] - event.get('custom_trim_before', self.config['trim_window_before']))
            trim_end = event['timestamp'] + event.get('custom_trim_after', self.config['trim_window_after'])
            event_output = {
                'id': i,
                'timestamp': event['timestamp'],
                'time_formatted': f"{mins:02d}:{secs:02d}",
                'event_type': event['final_event'],
                'confidence': round(event['final_confidence'], 4),
                'frame': event['frame'],
                'is_replay': event.get('is_replay', False),
                'trim_window': {
                    'start': round(trim_start, 2),
                    'end': round(trim_end, 2),
                    'duration': round(trim_end - trim_start, 2),
                    'adjusted': event.get('trim_adjusted', False)
                },
                'source_info': {
                    'num_models_detected': event.get('num_models_detected', 0),
                    'model_agreement': round(event.get('model_agreement', 0), 2),
                    'original_prediction': event.get('predicted_event'),
                }
            }
            if 'corrected_from' in event:
                event_output['correction'] = {
                    'original': event['corrected_from'],
                    'corrected_to': event['final_event'],
                    'reason': event['correction_reason']
                }

            if event.get('audio_analysis'):
                audio = event['audio_analysis']
                event_output['audio_verification'] = {
                    'verified': True,
                    'audio_confidence': audio['audio_confidence'],
                    'high_confidence_matches': audio['high_confidence_matches'],
                    'total_matches': audio['total_matches'],
                    'top_keywords': [m['keyword'] for m in audio.get('best_matches', [])[:3]]
                }
            else:
                event_output['audio_verification'] = {'verified': False}

            # REMOVED: sequence_info - no longer tracking sequences

            final_output['verified_events'].append(event_output)

        event_counts = defaultdict(int)
        for event in verified_events:
            event_counts[event['final_event']] += 1

        final_output['event_statistics'] = dict(event_counts)

        with open(self.config['output_final_json'], 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"\n{'='*60}")
        print("FINAL VERIFIED EVENTS (STRICT AUDIO MODE)")
        print(f"{'='*60}")

        if verified_events:
            for i, event in enumerate(final_output['verified_events']):
                if event.get('source_info', {}).get('num_models_detected', 0) == 0:
                    status_icon = "🔊"
                else:
                    status_icon = "🎥"

                correction_str = ""
                if 'correction' in event:
                    correction_str = f" [Corrected: {event['correction']['original']} → {event['correction']['corrected_to']}]"

                print(f"{status_icon} {event['id']}. [{event['time_formatted']}] {event['event_type'].upper()} "
                      f"(conf: {event['confidence']:.1%}){correction_str}")
                print(f" Trim: {event['trim_window']['start']:.1f}s - {event['trim_window']['end']:.1f}s")

                if event['audio_verification']['verified']:
                    audio = event['audio_verification']
                    print(f" Audio: {audio['total_matches']} matches ({audio['high_confidence_matches']} high)")
                    if audio.get('top_keywords'):
                        print(f" Keywords: {', '.join(audio['top_keywords'][:3])}")
                print()
        else:
            print("No verified events found")

        print(f"{'='*60}")
        print("STATISTICS")
        print(f"{'='*60}")
        print(f"Total events: {final_output['summary']['total_verified_events']}")
        print(f"Audio corrected: {final_output['summary']['audio_corrected']}")
        print(f"Audio-only events: {final_output['summary']['audio_only']}")
        print(f"Sequence validation: DISABLED (strict audio mode)")
        print(f"\nEvent distribution:")
        for event_type, count in final_output['event_statistics'].items():
            print(f" {event_type}: {count}")
        print(f"\n✓ Saved to: {self.config['output_final_json']}")
        print(f"{'='*60}\n")

    def process_video_chunked(
        self, 
        video_path: str,
        match_id: str = None,
        progress_callback = None
    ):
        """Process video in chunks for long videos"""
        
        # Create chunked processor
        chunked_processor = ChunkedVideoProcessor(self.config, chunk_duration_minutes=50)
        
        # Get video info
        video_info = chunked_processor.get_video_info(video_path)
        print(f"\n{'='*60}")
        print(f"VIDEO INFORMATION")
        print(f"{'='*60}")
        print(f"Duration: {video_info['duration']/60:.1f} minutes")
        print(f"FPS: {video_info['fps']:.2f}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"Total frames: {video_info['total_frames']:,}")
        print(f"{'='*60}\n")
        
        # Decide processing strategy
        if video_info['duration'] < 3000:  # Less than 50 minutes
            print("✓ Video is short - using standard processing")
            return self.process_video_with_both_models(video_path)
        
        # Use chunked processing for long videos
        print("✓ Video is long - using chunked processing")
        chunks = chunked_processor.calculate_chunks(video_info['duration'])
        
        all_model1_events = []
        all_model2_events = []
        
        for idx, (start_time, end_time) in enumerate(chunks):
            # Update progress
            if progress_callback:
                progress = (idx / len(chunks)) * 70  # Reserve 70% for video processing
                progress_callback(
                    status="processing",
                    progress=progress,
                    message=f"Processing chunk {idx + 1}/{len(chunks)}"
                )
            
            # Process chunk
            m1_events, m2_events = chunked_processor.process_chunk(
                video_path, start_time, end_time, idx, len(chunks), self
            )
            
            all_model1_events.append(m1_events)
            all_model2_events.append(m2_events)
            
            # Save intermediate results
            if match_id and idx % 2 == 0:  # Save every 2 chunks
                self._save_intermediate_results(
                    match_id, all_model1_events, all_model2_events, idx, len(chunks)
                )
        
        # Merge all chunk events
        print(f"\n{'='*60}")
        print("MERGING CHUNK RESULTS")
        print(f"{'='*60}")
        
        model1_events = chunked_processor.merge_chunk_events(all_model1_events)
        model2_events = chunked_processor.merge_chunk_events(all_model2_events)
        
        print(f"✓ Total Model 1 events: {len(model1_events)}")
        print(f"✓ Total Model 2 events: {len(model2_events)}")
        print(f"{'='*60}\n")
        
        return model1_events, model2_events, video_info['fps']
    
    def _save_intermediate_results(self, match_id, m1_events, m2_events, chunk_idx, total_chunks):
        """Save intermediate results to prevent data loss"""
        checkpoint_path = BASE_MEDIA_DIR / match_id / f"checkpoint_{chunk_idx}.json"
        
        checkpoint_data = {
            'chunk': chunk_idx,
            'total_chunks': total_chunks,
            'model1_events': sum(len(e) for e in m1_events),
            'model2_events': sum(len(e) for e in m2_events),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"✓ Checkpoint saved: {checkpoint_path.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def validate_paths(config):
    """Validate all required file paths."""
    # Only the video path is strictly required at runtime for an uploaded match.
    # Model files are optional during development; if missing we warn instead of raising.
    if not os.path.exists(config.get('video_path', '')):
        raise FileNotFoundError(f"Required video file not found: {config.get('video_path')}")

    # Warn about model files if they are missing (don't fail hard)
    for model_key in ('model1_path', 'model2_path'):
        model_path = config.get(model_key)
        if model_path and not os.path.exists(model_path):
            print(f"⚠️ Warning: model file not found: {model_path} (key: {model_key})")

    print("✓ Video path validated (models may be missing — see warnings if any)")

# UPDATED: Main execution - remove sequence validation
def run_robust_detection():
    print("="*60)
    print("EVENT-LEVEL CONFIDENCE RULES DETECTION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f" Model 1: {CONFIG['model1_path'].split('/')[-1]}")
    print(f" Model 2: {CONFIG['model2_path'].split('/')[-1]}")
    print(f" Video: {CONFIG['video_path'].split('/')[-1]}")
    print(f" Audio window: ±{CONFIG['audio_match_window']}s")
    print(f" Frame skip: {CONFIG['frame_skip']} | Stride: {CONFIG['stride']}")

    print(f"\nEvent-Level Rules:")
    for event_type, rules in EVENT_CONFIDENCE_RULES.items():
        audio_req = "❌ NO AUDIO REQUIRED" if not rules['requires_audio'] else "✅ AUDIO REQUIRED"
        print(f" {event_type.upper()}: {rules['ultra_high_threshold']*100:.0f}%+ conf → {audio_req}")
        print(f" Min matches: {rules['min_audio_matches']}, Min high-conf: {rules['min_high_confidence_matches']}")

    print("="*60)

    validate_paths(CONFIG)
    detector = RobustMultiModalDetector(CONFIG)

    model1_events, model2_events, fps = detector.process_video_with_both_models(CONFIG['video_path'])

    # Save raw detections
    detector.save_raw_detections(model1_events, model2_events, fps)

    transcription = detector.audio_analyzer.transcribe_video(CONFIG['video_path'])

    verified_events = detector.fusion_engine.fuse_detections(
        model1_events, model2_events, transcription, detector.audio_analyzer
    )

    detector.save_final_results(verified_events, fps)

    print(f"{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Model 1 raw detections: {len(model1_events)}")
    print(f"✓ Model 2 raw detections: {len(model2_events)}")
    print(f"✓ Final verified events: {len(verified_events)}")
    print(f"✓ Mode: EVENT-LEVEL RULES")
    print(f"✓ Output: {CONFIG['output_final_json']}")
    print(f"{'='*60}\n")

    return verified_events

# Helper function to convert ObjectId to string
def match_helper(match) -> dict:
    created_at = match.get("created_at")
    updated_at = match.get("updated_at")

    # Ensure datetimes are serialized as ISO strings for JSON responses
    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()
    if hasattr(updated_at, "isoformat"):
        updated_at = updated_at.isoformat()

    return {
        "id": str(match["_id"]),
        "match_id": match["match_id"],
        "title": match["title"],
        "date": match.get("date"),
        "description": match.get("description", ""),
        "video_path": match.get("video_path"),
        "poster_path": match.get("poster_path"),
        "status": match.get("status"),
        "main_highlights": match.get("main_highlights"),
        "event_clips": match.get("event_clips", {}),
        "analysis_data": match.get("analysis_data"),
        "created_at": created_at,
        "updated_at": updated_at
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/api/matches/upload")
async def upload_match(
    title: str = Form(...),
    date: str = Form(...),
    description: Optional[str] = Form(None),
    video: UploadFile = File(...),
    poster: Optional[UploadFile] = File(None)
):
    """
    Upload a new match video and metadata
    """
    try:
        # Generate unique match ID
        match_id = f"match_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

        # Create match-specific directory
        match_dir = BASE_MEDIA_DIR / match_id
        match_dir.mkdir(parents=True, exist_ok=True)

        # Save video file
        video_filename = f"original_video{Path(video.filename).suffix}"
        video_path = match_dir / video_filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Save poster if provided
        poster_path = None
        if poster:
            poster_filename = f"poster{Path(poster.filename).suffix}"
            poster_path = match_dir / poster_filename
            with open(poster_path, "wb") as buffer:
                shutil.copyfileobj(poster.file, buffer)

        # Create match document
        match_doc = {
            "match_id": match_id,
            "title": title,
            "date": date,
            "description": description or "",
            "video_path": str(video_path),
            "poster_path": str(poster_path) if poster_path else None,
            "status": "uploaded",  # uploaded -> processing -> completed -> failed
            "main_highlights": None,
            "event_clips": {},
            "analysis_data": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Insert into MongoDB
        result = await matches_collection.insert_one(match_doc)
        match_doc["_id"] = result.inserted_id

        return JSONResponse(content={
            "success": True,
            "message": "Match uploaded successfully",
            "match": match_helper(match_doc)
        }, status_code=201)

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.websocket("/ws/progress/{match_id}")
async def websocket_progress(websocket: WebSocket, match_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    
    if match_id not in progress_tracker.websockets:
        progress_tracker.websockets[match_id] = []
    progress_tracker.websockets[match_id].append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        progress_tracker.websockets[match_id].remove(websocket)

@app.post("/api/matches/{match_id}/analyze")
async def analyze_match(match_id: str):
    """
    Trigger AI analysis for a specific match (OPTIMIZED FOR LONG VIDEOS)
    """
    try:
        # Find match in database
        match = await matches_collection.find_one({"match_id": match_id})
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        # Update status to processing
        await matches_collection.update_one(
            {"match_id": match_id},
            {"$set": {"status": "processing", "updated_at": datetime.utcnow()}}
        )
        
        # Create a queue for progress updates
        import queue
        progress_queue = queue.Queue()
        
        # Run analysis in thread pool to avoid blocking
        def run_analysis():
            try:
                video_path = match["video_path"]
                match_dir = Path(video_path).parent
                clips_dir = match_dir / "clips"
                clips_dir.mkdir(exist_ok=True)

                # Update CONFIG
                CONFIG['video_path'] = video_path
                CONFIG['output_final_json'] = str(match_dir / "analysis_result.json")

                # Initialize detector
                detector = RobustMultiModalDetector(CONFIG)

                # Progress callback that puts updates in queue
                def sync_progress_update(**kwargs):
                    progress_queue.put(kwargs)

                # If raw detections already exist, reuse them to avoid re-running video inference
                raw_json_path = match_dir / CONFIG['output_final_json'].replace('.json', '_raw_detections.json')
                if raw_json_path.exists():
                    try:
                        with open(raw_json_path, 'r') as f:
                            raw = json.load(f)
                        model1_events = raw.get('model1_detections', {}).get('events', [])
                        model2_events = raw.get('model2_detections', {}).get('events', [])
                        fps = raw.get('video_info', {}).get('fps', 30)
                        progress_queue.put({"status": "processing", "progress": 20, "message": "Loaded raw detections - skipping video inference"})
                    except Exception as e:
                        # If loading fails, fall back to full processing
                        progress_queue.put({"status": "processing", "progress": 5, "message": "Analyzing video structure"})
                        model1_events, model2_events, fps = detector.process_video_chunked(
                            video_path,
                            match_id=match_id,
                            progress_callback=sync_progress_update
                        )
                else:
                    progress_queue.put({"status": "processing", "progress": 5, "message": "Analyzing video structure"})
                    # Use chunked processing for long videos
                    model1_events, model2_events, fps = detector.process_video_chunked(
                        video_path,
                        match_id=match_id,
                        progress_callback=sync_progress_update
                    )
                
                # Save raw detections
                progress_queue.put({"status": "processing", "progress": 75, "message": "Saving detections"})
                detector.save_raw_detections(model1_events, model2_events, fps)
                
                # Audio transcription
                progress_queue.put({"status": "processing", "progress": 80, "message": "Transcribing audio"})
                transcription = detector.audio_analyzer.transcribe_video(video_path)
                
                # Fusion and verification
                progress_queue.put({"status": "processing", "progress": 85, "message": "Verifying events"})
                verified_events = detector.fusion_engine.fuse_detections(
                    model1_events, model2_events, transcription, detector.audio_analyzer
                )
                
                # Save final results
                detector.save_final_results(verified_events, fps)
                
                # Load analysis results
                with open(CONFIG['output_final_json'], "r") as f:
                    analysis_data = json.load(f)
                
                # Create clips
                progress_queue.put({"status": "processing", "progress": 90, "message": "Creating clips"})
                event_clips = {}
                clip_counter = {}
                ts_paths = []

                for event in analysis_data['verified_events']:
                    event_type = event['event_type']
                    clip_counter[event_type] = clip_counter.get(event_type, 0) + 1

                    clip_name = f"highlight_{event_type}{clip_counter[event_type]}.mp4"
                    output_path = clips_dir / clip_name

                    start = event['trim_window']['start']
                    end = event['trim_window']['end']

                    # Create clip
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", video_path,
                        "-ss", str(start), "-to", str(end),
                        "-c:v", "libx264", "-c:a", "aac",
                        "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-y", str(output_path)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # Create TS intermediate
                    ts_path = output_path.with_suffix('.ts')
                    cmd_ts = [
                        "ffmpeg", "-loglevel", "error",
                        "-i", str(output_path),
                        "-c", "copy", "-bsf:v", "h264_mp4toannexb",
                        "-f", "mpegts", "-y", str(ts_path)
                    ]
                    subprocess.run(cmd_ts, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    if event_type not in event_clips:
                        event_clips[event_type] = []
                    event_clips[event_type].append(str(output_path))
                    ts_paths.append(str(ts_path))

                # Create main highlights
                progress_queue.put({"status": "processing", "progress": 95, "message": "Creating main highlights"})
                main_highlights_path = None
                if ts_paths:
                    merge_list_path = match_dir / "merge_ts.txt"
                    with open(merge_list_path, "w") as f:
                        for ts in ts_paths:
                            abs_path = Path(ts).absolute().as_posix()
                            f.write(f"file '{abs_path}'\n")

                    main_highlights_path = clips_dir / "main_highlights.mp4"
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-f", "concat", "-safe", "0",
                        "-i", str(merge_list_path),
                        "-c", "copy", "-bsf:a", "aac_adtstoasc",
                        "-y", str(main_highlights_path)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    merge_list_path.unlink(missing_ok=True)
                    for ts in ts_paths:
                        Path(ts).unlink(missing_ok=True)

                progress_queue.put({"status": "completed", "progress": 100, "message": "Analysis complete"})
                
                return {
                    "success": True,
                    "main_highlights": str(main_highlights_path) if main_highlights_path else None,
                    "event_clips": event_clips,
                    "analysis_data": analysis_data
                }
                
            except Exception as e:
                import traceback
                error_msg = f"Error in analysis: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                progress_queue.put({"status": "failed", "progress": 0, "message": str(e)})
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Start background task for processing updates
        async def process_progress_updates():
            while True:
                try:
                    # Get progress update from queue (non-blocking)
                    update = progress_queue.get_nowait()
                    
                    # Update tracker and broadcast
                    progress_tracker.update(
                        match_id, 
                        update.get("status", "processing"), 
                        update.get("progress", 0), 
                        update.get("message", "")
                    )
                    await progress_tracker.broadcast(match_id)
                    await matches_collection.update_one(
                        {"match_id": match_id},
                        {"$set": {
                            "progress": update.get("progress", 0), 
                            "progress_message": update.get("message", "")
                        }}
                    )
                except queue.Empty:
                    # No more updates, wait a bit
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Error processing progress update: {e}")
                    break
        
        # Start the progress updater task
        progress_task = asyncio.create_task(process_progress_updates())
        
        # Run analysis in background
        try:
            result = await run_in_threadpool(run_analysis)
        except Exception as e:
            # Cancel progress task on error
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            raise e
        
        # Cancel progress task
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        
        # Validate result
        if result is None:
            raise Exception("Analysis returned None")
        
        if not isinstance(result, dict):
            raise Exception(f"Analysis returned invalid type: {type(result)}")
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error occurred")
            traceback_info = result.get("traceback", "")
            if traceback_info:
                print(f"Analysis failed with traceback:\n{traceback_info}")
            raise Exception(error_msg)
        
        # Update database
        await matches_collection.update_one(
            {"match_id": match_id},
            {
                "$set": {
                    "status": "completed",
                    "main_highlights": result.get("main_highlights"),
                    "event_clips": result.get("event_clips", {}),
                    "analysis_data": result.get("analysis_data", {}),
                    "updated_at": datetime.utcnow()
                }
            }
        )

        return JSONResponse(content=result)

    except Exception as e:
        # Update progress with error
        progress_tracker.update(match_id, "failed", 0, str(e))
        await progress_tracker.broadcast(match_id)
        
        await matches_collection.update_one(
            {"match_id": match_id},
            {"$set": {"status": "failed", "updated_at": datetime.utcnow()}}
        )
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/matches/{match_id}/progress")
async def get_progress(match_id: str):
    """Get current analysis progress"""
    if match_id in progress_tracker.progress:
        return JSONResponse(content=progress_tracker.progress[match_id])
    
    # Fallback to database
    match = await matches_collection.find_one({"match_id": match_id})
    if match:
        return JSONResponse(content={
            "status": match.get("status", "unknown"),
            "progress": match.get("progress", 0),
            "message": match.get("progress_message", "")
        })
    
    raise HTTPException(status_code=404, detail="Match not found")

@app.get("/api/matches")
async def get_all_matches(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None
):
    """
    Get all matches with pagination
    """
    try:
        query = {}
        if status:
            query["status"] = status

        cursor = matches_collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        matches = await cursor.to_list(length=limit)

        total = await matches_collection.count_documents(query)

        return JSONResponse(content={
            "success": True,
            "matches": [match_helper(match) for match in matches],
            "total": total,
            "skip": skip,
            "limit": limit
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/matches/{match_id}")
async def get_match_details(match_id: str):
    """
    Get detailed information about a specific match
    """
    try:
        match = await matches_collection.find_one({"match_id": match_id})
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        return JSONResponse(content={
            "success": True,
            "match": match_helper(match)
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/media/{match_id}/{file_path:path}")
async def serve_media(match_id: str, file_path: str):
    """
    Serve media files (videos, images)
    """
    try:
        # Normalize incoming file_path to avoid duplicated prefixes such as
        # /api/media/{match_id}/media/{match_id}/poster.png which can happen
        # if callers pass a stored filesystem path instead of a relative one.
        # Allow both 'poster.png', 'clips/clip.mp4' and full stored paths like
        # 'media/match_x/poster.png' or absolute paths.

        requested = file_path.replace('\\', '/')

        # If the client accidentally sent a path that already contains the base media dir,
        # strip everything up through the match_id so we end up with a path relative to match_id.
        parts = requested.split('/')
        if len(parts) >= 3 and parts[0] == 'media' and parts[1] == match_id:
            # e.g. media/match_x/poster.png -> poster.png
            relative = '/'.join(parts[2:])
        else:
            # If the path begins with match_id (e.g., match_x/poster.png) strip it
            if len(parts) >= 1 and parts[0] == match_id:
                relative = '/'.join(parts[1:])
            else:
                relative = requested

        # Prevent path traversal and ensure the final path is inside BASE_MEDIA_DIR/match_id
        full_path = (BASE_MEDIA_DIR / match_id / Path(relative)).resolve()
        base_path = (BASE_MEDIA_DIR / match_id).resolve()
        try:
            # Ensure full_path is within base_path
            full_path.relative_to(base_path)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(str(full_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/matches/{match_id}")
async def delete_match(match_id: str):
    """
    Delete a match and all its associated files
    """
    try:
        match = await matches_collection.find_one({"match_id": match_id})
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        # Delete files
        match_dir = BASE_MEDIA_DIR / match_id
        if match_dir.exists():
            shutil.rmtree(match_dir)

        # Delete from database
        await matches_collection.delete_one({"match_id": match_id})

        return JSONResponse(content={
            "success": True,
            "message": "Match deleted successfully"
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)