#!/usr/bin/env python3
"""
Hybrid Football Video Analysis System

Combines:
1. Time-based indexing (for precise temporal queries like "first goal")
2. FAISS semantic search (for content-based queries like "aggressive tackles")

This gives you the best of both worlds:
- Precise temporal analysis for event sequences
- Semantic similarity search for content discovery

Installation required:
    pip install faiss-cpu sentence-transformers numpy

Usage:
    python hybrid_rag.py

Author: GitHub Copilot
Date: October 2025
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re

try:
    import faiss
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  For semantic search: pip install faiss-cpu sentence-transformers")

try:
    import google.generativeai as genai
    from config import GEMINI_API_KEY
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  For AI analysis: setup Gemini API key")

class TimeBasedIndex:
    """Handles precise temporal queries - your existing system"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.index_data = None
        self.combined_data = None
        self.load_data()
    
    def load_data(self):
        """Load preprocessed video data"""
        try:
            index_path = os.path.join(self.workspace_path, "global_video_index.json")
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)
            
            combined_path = os.path.join(self.workspace_path, "combined_video_data.json")
            with open(combined_path, 'r', encoding='utf-8') as f:
                self.combined_data = json.load(f)
            
            print("✅ Time-based index loaded")
        except FileNotFoundError as e:
            print(f"❌ Error loading time index: {e}")
    
    def get_event_sequence(self, event_type: str, sequence_number: int, context_seconds: int = 15) -> Dict[str, Any]:
        """Get specific event occurrence with time context"""
        if not self.index_data:
            return {"error": "No data loaded"}
        
        events = self.index_data["event_index"].get(event_type, [])
        
        if len(events) < sequence_number:
            return {
                "error": f"Only {len(events)} {event_type} events found, requested #{sequence_number}",
                "available_events": len(events)
            }
        
        target_event = events[sequence_number - 1]
        target_timestamp = target_event["global_timestamp"]
        
        # Calculate time window
        target_seconds = self.parse_timestamp(target_timestamp)
        start_seconds = max(0, target_seconds - context_seconds)
        end_seconds = target_seconds + context_seconds
        
        start_time_str = self.format_timestamp(start_seconds)
        end_time_str = self.format_timestamp(end_seconds)
        
        # Get context frames
        context_frames = self.get_frames_in_range(start_time_str, end_time_str)
        
        return {
            "query_type": "temporal",
            "event_type": event_type,
            "sequence_number": sequence_number,
            "target_event": target_event,
            "time_range": {
                "start": start_time_str,
                "end": end_time_str,
                "center": target_timestamp
            },
            "context_frames": context_frames,
            "frame_count": len(context_frames)
        }
    
    def get_frames_in_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """Get frames in time range"""
        if not self.combined_data:
            return []
        
        start_seconds = self.parse_timestamp(start_time)
        end_seconds = self.parse_timestamp(end_time)
        
        frames = []
        for segment in self.combined_data["segments"]:
            for frame in segment["frames"]:
                frame_seconds = self.parse_timestamp(frame["global_timestamp"])
                if start_seconds <= frame_seconds <= end_seconds:
                    frames.append(frame)
        
        frames.sort(key=lambda x: self.parse_timestamp(x["global_timestamp"]))
        return frames
    
    def parse_timestamp(self, ts: str) -> float:
        """Convert timestamp to seconds"""
        parts = ts.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

class FAISSSemanticIndex:
    """Handles semantic similarity queries"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.embeddings_path = os.path.join(workspace_path, "embeddings")
        self.model = None
        self.index = None
        self.frame_metadata = []
        self.setup_model()
    
    def setup_model(self):
        """Setup sentence transformer model"""
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available. Install: pip install faiss-cpu sentence-transformers")
            return
        
        try:
            # Use a lightweight, fast model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded")
            
            # Try to load existing index
            self.load_existing_index()
        except Exception as e:
            print(f"⚠️  Semantic model setup failed: {e}")
    
    def create_embeddings(self, combined_data: Dict) -> None:
        """Create embeddings for all frames"""
        if not self.model:
            return
        
        print("🔄 Creating semantic embeddings...")
        
        texts = []
        metadata = []
        
        for segment in combined_data["segments"]:
            for frame in segment["frames"]:
                # Combine all textual information for embedding
                text_content = []
                
                # Add events as text
                if frame["active_events"]:
                    text_content.append(f"Events: {', '.join(frame['active_events'])}")
                
                # Add description
                if frame.get("rag_description"):
                    text_content.append(frame["rag_description"])
                
                # Create searchable text
                full_text = " | ".join(text_content) if text_content else "football match scene"
                texts.append(full_text)
                
                # Store metadata for retrieval
                metadata.append({
                    "timestamp": frame["global_timestamp"],
                    "file": frame["file"],
                    "events": frame["active_events"],
                    "description": frame.get("rag_description", ""),
                    "segment_id": frame["segment_id"],
                    "frame_number": frame["frame_number"]
                })
        
        # Generate embeddings
        print(f"📊 Generating embeddings for {len(texts)} frames...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.frame_metadata = metadata
        
        # Save index and metadata
        self.save_index()
        print(f"✅ Semantic index created with {len(texts)} frames")
    
    def save_index(self):
        """Save FAISS index and metadata"""
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.embeddings_path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(self.embeddings_path, "frame_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_metadata, f, indent=2, ensure_ascii=False)
        
        print("💾 Semantic index saved")
    
    def load_existing_index(self):
        """Load existing FAISS index if available"""
        index_path = os.path.join(self.embeddings_path, "faiss_index.bin")
        metadata_path = os.path.join(self.embeddings_path, "frame_metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.frame_metadata = json.load(f)
                
                print(f"✅ Existing semantic index loaded ({len(self.frame_metadata)} frames)")
            except Exception as e:
                print(f"⚠️  Could not load existing index: {e}")
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for semantically similar frames"""
        if not self.model or not self.index:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.frame_metadata):
                result = self.frame_metadata[idx].copy()
                result["similarity_score"] = float(score)
                result["rank"] = i + 1
                result["query_type"] = "semantic"
                results.append(result)
        
        return results

class HybridVideoRAGQuerier:
    """Intelligent router between temporal and semantic search"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.time_index = TimeBasedIndex(workspace_path)
        self.semantic_index = FAISSSemanticIndex(workspace_path)
        self.gemini_model = None
        
        # Initialize embeddings if needed
        self.initialize_semantic_index()
        
        # Setup Gemini
        if GEMINI_AVAILABLE:
            self.setup_gemini()
    
    def initialize_semantic_index(self):
        """Create semantic embeddings if they don't exist"""
        if (FAISS_AVAILABLE and 
            self.time_index.combined_data and 
            not self.semantic_index.index):
            
            create = input("Create semantic index for content search? (y/n): ").lower() == 'y'
            if create:
                self.semantic_index.create_embeddings(self.time_index.combined_data)
    
    def setup_gemini(self):
        """Setup Gemini for analysis"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini 2.5 Flash ready")
        except Exception as e:
            print(f"⚠️  Gemini setup failed: {e}")
    
    def detect_query_type(self, query: str) -> str:
        """Intelligently detect whether to use temporal or semantic search"""
        query_lower = query.lower()
        
        # Temporal indicators
        temporal_patterns = [
            'first', 'second', 'third', 'last', 'final', '1st', '2nd', '3rd',
            'when', 'at what time', 'timestamp', 'sequence', 'before', 'after',
            'goal 1', 'goal 2', 'pass 3', 'tackle 1'
        ]
        
        # Semantic indicators  
        semantic_patterns = [
            'similar', 'like', 'aggressive', 'creative', 'skillful', 'style',
            'find all', 'show me', 'type of', 'kind of', 'tactical', 'pattern'
        ]
        
        # Check for temporal patterns
        if any(pattern in query_lower for pattern in temporal_patterns):
            return "temporal"
        
        # Check for semantic patterns
        if any(pattern in query_lower for pattern in semantic_patterns):
            return "semantic"
        
        # Default: if it mentions specific ordinals, use temporal
        if re.search(r'\b(first|second|third|\d+(?:st|nd|rd|th))\b', query_lower):
            return "temporal"
        
        # Otherwise, use semantic
        return "semantic"
    
    def parse_temporal_query(self, query: str) -> Dict[str, Any]:
        """Parse temporal query to extract event and sequence"""
        query_lower = query.lower()
        
        # Event mapping
        event_mapping = {
            'goal': 'goal', 'goals': 'goal',
            'pass': 'pass', 'passes': 'pass',
            'tackle': 'tackle', 'tackles': 'tackle',
            'save': 'save', 'saves': 'save',
            'card': 'card', 'cards': 'card',
            'foul': 'foul', 'fouls': 'foul',
            'celebration': 'celebration',
            'goal_attempt': 'goal_attempt', 'shot': 'goal_attempt'
        }
        
        # Find event type
        event_type = None
        for word in query_lower.split():
            if word in event_mapping:
                event_type = event_mapping[word]
                break
        
        if not event_type:
            return {"error": "Could not identify event type"}
        
        # Find sequence number
        sequence_number = 1
        if 'first' in query_lower or '1st' in query_lower:
            sequence_number = 1
        elif 'second' in query_lower or '2nd' in query_lower:
            sequence_number = 2
        elif 'third' in query_lower or '3rd' in query_lower:
            sequence_number = 3
        elif 'last' in query_lower or 'final' in query_lower:
            if self.time_index.index_data:
                events = self.time_index.index_data["event_index"].get(event_type, [])
                sequence_number = len(events) if events else 1
        
        # Check for numbers
        numbers = re.findall(r'\b(\d+)\b', query_lower)
        if numbers:
            sequence_number = int(numbers[0])
        
        return {
            "event_type": event_type,
            "sequence_number": sequence_number
        }
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Main query interface - intelligently routes to best search method"""
        
        query_type = self.detect_query_type(user_query)
        
        print(f"🔍 Query: '{user_query}'")
        print(f"🎯 Using: {query_type} search")
        
        if query_type == "temporal":
            return self.temporal_query(user_query)
        else:
            return self.semantic_query(user_query)
    
    def temporal_query(self, query: str) -> Dict[str, Any]:
        """Handle temporal queries using time-based index"""
        parsed = self.parse_temporal_query(query)
        
        if "error" in parsed:
            return parsed
        
        result = self.time_index.get_event_sequence(
            parsed["event_type"], 
            parsed["sequence_number"]
        )
        
        if "error" not in result:
            result["search_method"] = "temporal_index"
            result["explanation"] = f"Found {parsed['event_type']} #{parsed['sequence_number']} using precise temporal lookup"
        
        return result
    
    def semantic_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Handle semantic queries using FAISS similarity search"""
        if not self.semantic_index.index:
            return {"error": "Semantic search not available. Create embeddings first."}
        
        results = self.semantic_index.semantic_search(query, top_k)
        
        return {
            "query_type": "semantic", 
            "search_method": "faiss_similarity",
            "explanation": f"Found {len(results)} semantically similar frames",
            "results": results,
            "total_results": len(results)
        }
    
    def create_analysis_prompt(self, query_result: Dict[str, Any], original_query: str) -> str:
        """Create appropriate prompt for Gemini based on query type"""
        
        if query_result.get("query_type") == "temporal":
            # Temporal analysis prompt
            frames = query_result.get("context_frames", [])
            target_event = query_result.get("target_event", {})
            
            prompt = f"""PORTUGAL vs SPAIN 2018 UCL FINAL - TACTICAL ANALYSIS

USER QUERY: "{original_query}"

EVENT: {query_result.get('event_type', '').upper()} #{query_result.get('sequence_number', '')}
TIMESTAMP: {target_event.get('global_timestamp', '')}
TIME WINDOW: {query_result.get('time_range', {}).get('start', '')} to {query_result.get('time_range', {}).get('end', '')}

FRAME SEQUENCE:
"""
            
            for i, frame in enumerate(frames[:20], 1):  # Limit to 20 frames for performance
                is_target = frame.get("global_timestamp") == target_event.get("global_timestamp")
                marker = "🎯 KEY MOMENT" if is_target else f"   {i:2d}"
                
                prompt += f"\n{marker} [{frame.get('global_timestamp')}]\n"
                if frame.get("active_events"):
                    prompt += f"Events: {', '.join(frame['active_events'])}\n"
                prompt += f"Scene: {frame.get('rag_description', '')[:200]}...\n"
            
            prompt += f"""

ANALYSIS REQUEST:
Provide tactical analysis of this {query_result.get('event_type', '')} in the Portugal vs Spain 2018 UCL Final.

Focus on:
1. HOW IT DEVELOPED: What led to this moment?
2. EXECUTION: How was it executed?
3. TACTICAL IMPACT: What does this reveal about team strategies?
4. KEY DETAILS: Specific technical aspects

Be specific and insightful. Use the frame sequence to tell the tactical story."""
            
        else:
            # Semantic analysis prompt
            results = query_result.get("results", [])
            
            prompt = f"""PORTUGAL vs SPAIN 2018 UCL FINAL - CONTENT ANALYSIS

USER QUERY: "{original_query}"

SIMILAR FRAMES FOUND: {len(results)}

MATCHING CONTENT:
"""
            
            for i, result in enumerate(results[:10], 1):  # Top 10 results
                prompt += f"\n{i}. [{result.get('timestamp')}] (Similarity: {result.get('similarity_score', 0):.3f})\n"
                if result.get("events"):
                    prompt += f"Events: {', '.join(result['events'])}\n"
                prompt += f"Scene: {result.get('description', '')[:150]}...\n"
            
            prompt += f"""

ANALYSIS REQUEST:
Analyze the common patterns and themes across these similar moments.

Focus on:
1. COMMON PATTERNS: What makes these frames similar?
2. TACTICAL THEMES: What tactical elements appear repeatedly?
3. KEY INSIGHTS: What does this tell us about the match?
4. CONTEXT: How do these moments fit into the overall game?

Provide insights based on the similar content found."""
        
        return prompt
    
    def analyze_with_gemini(self, query_result: Dict[str, Any], original_query: str) -> str:
        """Get AI analysis using Gemini"""
        if not self.gemini_model:
            return "❌ Gemini AI not available"
        
        prompt = self.create_analysis_prompt(query_result, original_query)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def interactive_mode(self):
        """Interactive query interface"""
        print("\n🎥 HYBRID Football Video Analysis")
        print("=" * 50)
        print("🎯 Temporal queries: 'first goal', 'second pass'")
        print("🔍 Semantic queries: 'aggressive tackles', 'creative passes'")
        print("Commands: 'stats', 'help', 'quit'")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print(f"\n📖 Help:")
                    print("Temporal: 'first goal', 'second tackle', 'last pass'")
                    print("Semantic: 'show aggressive defending', 'creative attacks'")
                    if self.time_index.index_data:
                        available = list(self.time_index.index_data['event_index'].keys())
                        print(f"Available events: {', '.join(available)}")
                
                elif user_input.lower() == 'stats':
                    if self.time_index.index_data:
                        print(f"\n📊 Video Statistics:")
                        print(f"Duration: {self.time_index.index_data['total_duration']}")
                        for event_type, occurrences in self.time_index.index_data['event_index'].items():
                            if len(occurrences) > 0:
                                print(f"  {event_type}: {len(occurrences)} times")
                        
                        if self.semantic_index.index:
                            print(f"\n🔍 Semantic Search: {len(self.semantic_index.frame_metadata)} frames indexed")
                
                else:
                    # Process query
                    result = self.query(user_input)
                    
                    if "error" in result:
                        print(f"❌ {result['error']}")
                        continue
                    
                    # Display results based on type
                    if result.get("query_type") == "temporal":
                        self.display_temporal_result(result, user_input)
                    else:
                        self.display_semantic_result(result, user_input)
                    
                    # Offer AI analysis
                    if self.gemini_model:
                        analyze = input("\n🤖 Get AI analysis? (y/n): ").lower() == 'y'
                        if analyze:
                            print("\n🔄 Analyzing with Gemini 2.5 Flash...")
                            analysis = self.analyze_with_gemini(result, user_input)
                            print(f"\n🤖 GEMINI ANALYSIS:")
                            print("=" * 60)
                            print(analysis)
                            print("=" * 60)
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_temporal_result(self, result: Dict[str, Any], query: str):
        """Display temporal search results"""
        print(f"\n✅ {result.get('event_type', '').upper()} #{result.get('sequence_number', '')}")
        print(f"⏱️  Time: {result.get('target_event', {}).get('global_timestamp', '')}")
        print(f"📁 Frame: {result.get('target_event', {}).get('file', '')}")
        print(f"🎬 Context: {result.get('frame_count', 0)} frames")
        time_range = result.get('time_range', {})
        print(f"🕒 Range: {time_range.get('start', '')} to {time_range.get('end', '')}")
    
    def display_semantic_result(self, result: Dict[str, Any], query: str):
        """Display semantic search results"""
        results = result.get("results", [])
        print(f"\n🔍 Found {len(results)} similar frames:")
        
        for i, res in enumerate(results[:5], 1):  # Show top 5
            score = res.get('similarity_score', 0)
            timestamp = res.get('timestamp', '')
            events = res.get('events', [])
            
            print(f"{i:2d}. [{timestamp}] (Score: {score:.3f})")
            if events:
                print(f"    Events: {', '.join(events)}")
            desc = res.get('description', '')[:100]
            print(f"    Scene: {desc}...")
            print()


def main():
    """Main function"""
    workspace_path = r"c:\Users\SHUBHAM\Downloads\match"
    
    print("🚀 HYBRID Football Video Analysis System")
    print("=" * 50)
    print("Combining temporal precision + semantic similarity")
    
    try:
        analyzer = HybridVideoRAGQuerier(workspace_path)
        
        if analyzer.time_index.index_data:
            print(f"\n🎯 System Ready:")
            print(f"⏱️  Temporal search: ✅ ({sum(analyzer.time_index.index_data['frame_count_by_segment'].values())} frames)")
            
            if analyzer.semantic_index.index:
                print(f"🔍 Semantic search: ✅ ({len(analyzer.semantic_index.frame_metadata)} embeddings)")
            else:
                print(f"🔍 Semantic search: ⚠️  (not created)")
            
            if analyzer.gemini_model:
                print(f"🤖 AI Analysis: ✅ (Gemini 2.5 Flash)")
            else:
                print(f"🤖 AI Analysis: ⚠️  (not available)")
            
            # Quick demo
            print(f"\n🚀 Quick Demo:")
            demo_result = analyzer.query("first goal")
            if "error" not in demo_result:
                target = demo_result.get('target_event', {})
                print(f"✅ Temporal query works: Goal at {target.get('global_timestamp', '')}")
            
            analyzer.interactive_mode()
        else:
            print("❌ No video data found. Run process_video_segments.py first!")
    
    except Exception as e:
        print(f"❌ System error: {e}")


if __name__ == "__main__":
    main()