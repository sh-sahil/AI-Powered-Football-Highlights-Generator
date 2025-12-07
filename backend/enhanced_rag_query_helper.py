#!/usr/bin/env python3
"""
Smart Football Video RAG Query Helper with Rule-Based Deduplication
NO EXTRA API CALLS - Uses existing data intelligently

Works exactly like original RAG helper but removes duplicates automatically
"""

import json
import os
import re
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from config import GEMINI_API_KEY


class SmartVideoRAGQuerier:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.index_data = None
        self.combined_data = None
        self.gemini_model = None
        
        self.load_data()
        self.setup_gemini_api()
        self.smart_deduplicate_events()  # Rule-based, no API calls!
    
    def load_data(self):
        """Load the processed video data"""
        try:
            index_path = os.path.join(self.workspace_path, "global_video_index.json")
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)
            
            combined_path = os.path.join(self.workspace_path, "combined_video_data.json")
            with open(combined_path, 'r', encoding='utf-8') as f:
                self.combined_data = json.load(f)
            
            print("✓ Video data loaded")
            print(f"  Segments: {self.index_data['total_segments']}")
            print(f"  Duration: {self.index_data['total_duration']}\n")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
    
    def setup_gemini_api(self):
        """Setup Gemini - only for final analysis, not deduplication"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✓ Gemini configured\n")
        except Exception as e:
            print(f"Gemini setup failed: {e}")
            raise
    
    def smart_deduplicate_events(self):
        """
        Rule-based deduplication - NO API CALLS
        
        Strategy:
        1. Group events within 30s windows
        2. In each group, pick the FIRST goal frame (actual moment)
        3. Mark others as celebration/replay based on:
           - If celebration tag exists → celebration
           - If it's after first goal → celebration
           - If it's >60s later and similar → replay
        """
        print("🔍 Smart deduplication starting...")
        
        for event_type in self.index_data['event_index'].keys():
            events = self.index_data['event_index'][event_type]
            if not events:
                continue
            
            original_count = len(events)
            
            # Apply smart filtering based on event type
            if event_type == 'goal':
                deduplicated = self._deduplicate_goals_rule_based(events)
            elif event_type == 'celebration':
                # Keep celebrations but don't count as goals
                deduplicated = events  
            else:
                # Other events: simple time-based deduplication
                deduplicated = self._deduplicate_generic(events)
            
            self.index_data['event_index'][event_type] = deduplicated
            
            removed = original_count - len(deduplicated)
            if removed > 0:
                print(f"  ✓ {event_type}: {original_count} → {len(deduplicated)} "
                      f"({removed} removed)")
        
        print("✓ Deduplication complete!\n")
    
    def _deduplicate_goals_rule_based(self, events: List[Dict]) -> List[Dict]:
        """
        Rule-based goal deduplication - NO API CALLS
        
        Logic:
        1. Sort by timestamp
        2. Create 30-second windows
        3. In each window: keep ONLY the first frame tagged as goal
           (First = actual goal, rest = celebrations/replays)
        4. Between windows: if <60s gap, might be real goals (keep both)
        """
        if not events:
            return []
        
        # Sort by time
        sorted_events = sorted(events, 
                              key=lambda x: self.parse_timestamp(x['global_timestamp']))
        
        # Get corresponding frames to check for celebration tags
        frame_lookup = self._build_frame_lookup()
        
        deduplicated = []
        i = 0
        
        while i < len(sorted_events):
            current_event = sorted_events[i]
            current_time = self.parse_timestamp(current_event['global_timestamp'])
            
            # Check if this frame has celebration tag too
            frame_data = frame_lookup.get(current_event['file'])
            has_celebration = False
            if frame_data and 'celebration' in frame_data.get('active_events', []):
                has_celebration = True
            
            # Start a new window
            window = [current_event]
            j = i + 1
            
            # Collect all events within 30 seconds
            while j < len(sorted_events):
                next_event = sorted_events[j]
                next_time = self.parse_timestamp(next_event['global_timestamp'])
                
                if next_time - current_time <= 30:  # 30-second window
                    window.append(next_event)
                    j += 1
                else:
                    break
            
            # From this window, pick the ACTUAL goal
            actual_goal = self._pick_actual_goal_from_window(window, frame_lookup)
            
            if actual_goal:
                deduplicated.append(actual_goal)
            
            i = j
        
        # Final pass: remove distant replays (same goal shown 2+ mins later)
        final_goals = self._remove_distant_replays_rule_based(deduplicated)
        
        return final_goals
    
    def _build_frame_lookup(self) -> Dict[str, Dict]:
        """Build quick lookup: filename -> frame data"""
        lookup = {}
        for segment in self.combined_data['segments']:
            for frame in segment['frames']:
                lookup[frame['file']] = frame
        return lookup
    
    def _pick_actual_goal_from_window(self, window: List[Dict], 
                                      frame_lookup: Dict) -> Optional[Dict]:
        """
        From a 30s window of 'goal' events, pick the actual goal frame
        
        Rules:
        1. If only 1 event → that's the goal
        2. If multiple events:
           - Skip frames that ALSO have 'celebration' tag (those are celebrations)
           - Pick the EARLIEST remaining frame (actual goal happens first)
        """
        if len(window) == 1:
            return window[0]
        
        # Filter out frames that are clearly celebrations
        goal_candidates = []
        
        for event in window:
            frame_data = frame_lookup.get(event['file'])
            if not frame_data:
                goal_candidates.append(event)
                continue
            
            active_events = frame_data.get('active_events', [])
            
            # If frame has ONLY celebration tag (no goal), skip it
            if 'celebration' in active_events and active_events.count('goal') == 0:
                continue  # This is celebration only
            
            # If both goal AND celebration, check description
            if 'celebration' in active_events and 'goal' in active_events:
                desc = frame_data.get('rag_description', '').lower()
                
                # Celebration keywords
                if any(word in desc for word in ['celebrating', 'hugging', 'arms raised', 
                                                  'embrace', 'team celebration']):
                    continue  # Skip celebrations
            
            goal_candidates.append(event)
        
        # Return earliest goal candidate
        if goal_candidates:
            return min(goal_candidates, 
                      key=lambda x: self.parse_timestamp(x['global_timestamp']))
        
        # Fallback: return first event in window
        return window[0]
    
    def _remove_distant_replays_rule_based(self, goals: List[Dict]) -> List[Dict]:
        """
        Remove replays shown minutes after the actual goal
        
        Rule: If two goals are >90 seconds apart but in same batch,
        the later one is likely a replay
        """
        if len(goals) <= 1:
            return goals
        
        final_goals = []
        
        for i, goal in enumerate(goals):
            is_replay = False
            goal_time = self.parse_timestamp(goal['global_timestamp'])
            goal_batch = goal['file'].split('/')[0]  # e.g., 'batch1'
            
            # Check against all previous goals
            for prev_goal in goals[:i]:
                prev_time = self.parse_timestamp(prev_goal['global_timestamp'])
                prev_batch = prev_goal['file'].split('/')[0]
                
                time_diff = goal_time - prev_time
                
                # If same batch and >90s later, likely a replay
                if goal_batch == prev_batch and time_diff > 90:
                    is_replay = True
                    break
                
                # If different batch but <180s (3 mins), likely real goal
                if time_diff < 180:
                    continue
            
            if not is_replay:
                final_goals.append(goal)
        
        return final_goals
    
    def _deduplicate_generic(self, events: List[Dict]) -> List[Dict]:
        """Generic deduplication for non-goal events"""
        if len(events) <= 1:
            return events
        
        sorted_events = sorted(events, 
                              key=lambda x: self.parse_timestamp(x['global_timestamp']))
        
        deduplicated = []
        last_time = -999
        
        for event in sorted_events:
            current_time = self.parse_timestamp(event['global_timestamp'])
            
            # Keep if >15 seconds from last event of same type
            if current_time - last_time > 15:
                deduplicated.append(event)
                last_time = current_time
        
        return deduplicated
    
    def parse_timestamp(self, ts: str) -> float:
        """Convert timestamp to seconds"""
        parts = ts.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        millis = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + millis / 1000
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    # === REST OF YOUR ORIGINAL RAG QUERY CODE ===
    
    def detect_query_type(self, query: str) -> Dict[str, Any]:
        """Detect if query is a question or analysis request"""
        query_lower = query.lower().strip()
        
        question_words = ['is', 'was', 'did', 'does', 'can', 'could', 'would', 
                         'what', 'when', 'where', 'who', 'why', 'how', 'which']
        
        is_question = any(query_lower.startswith(qw) for qw in question_words) or '?' in query
        
        event_info = self.quick_event_query(query)
        
        return {
            "is_question": is_question,
            "user_query": query,
            "event_info": event_info
        }
    
    def quick_event_query(self, query: str) -> Dict[str, Any]:
        """Handle natural language queries"""
        query = query.lower().strip()
        
        event_mapping = {
            'goal': 'goal', 'goals': 'goal',
            'pass': 'pass', 'passes': 'pass',
            'tackle': 'tackle', 'tackles': 'tackle',
            'save': 'save', 'saves': 'save',
            'card': 'card', 'cards': 'card', 'yellow': 'card', 'red': 'card',
            'foul': 'foul', 'fouls': 'foul',
            'celebration': 'celebration',
            'goal_attempt': 'goal_attempt', 'shot': 'goal_attempt',
            'free_kick': 'free_kick', 'free kick': 'free_kick',
            'corner_kick': 'corner_kick', 'corner': 'corner_kick',
            'dribble': 'dribble', 'dribbling': 'dribble',
            'penalty': 'penalty', 'penalties': 'penalty'
        }
        
        event_type = None
        for word in query.split():
            if word in event_mapping:
                event_type = event_mapping[word]
                break
        
        if not event_type:
            if 'free kick' in query:
                event_type = 'free_kick'
            elif 'corner kick' in query:
                event_type = 'corner_kick'
        
        if not event_type:
            return {"error": f"Event type not found. Available: {list(set(event_mapping.values()))}"}
        
        # Extract sequence number
        sequence_number = self._extract_sequence_number(query)
        
        events = self.index_data["event_index"].get(event_type, [])
        if not events:
            return {"error": f"No {event_type} events found"}
        
        if sequence_number == -1:
            sequence_number = len(events)
        
        return self.get_event_sequence(event_type, sequence_number)
    
    def _extract_sequence_number(self, query: str) -> int:
        """Extract which occurrence (first, second, etc.)"""
        number_words = {
            'first': 1, '1st': 1,
            'second': 2, '2nd': 2,
            'third': 3, '3rd': 3,
            'fourth': 4, '4th': 4,
            'fifth': 5, '5th': 5,
            'last': -1, 'final': -1
        }
        
        for word, num in number_words.items():
            if word in query:
                return num
        
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            return int(numbers[0])
        
        return 1
    
    def get_event_sequence(self, event_type: str, sequence_number: int = 1,
                          context_seconds: int = 10) -> Dict[str, Any]:
        """Get specific event with context frames"""
        events = self.index_data["event_index"].get(event_type, [])
        
        if len(events) < sequence_number:
            return {
                "error": f"Only {len(events)} unique {event_type} events (after deduplication), requested #{sequence_number}",
                "available_events": len(events)
            }
        
        target_event = events[sequence_number - 1]
        target_timestamp = target_event["global_timestamp"]
        
        target_seconds = self.parse_timestamp(target_timestamp)
        start_seconds = max(0, target_seconds - context_seconds)
        end_seconds = target_seconds + context_seconds
        
        start_time = self.format_timestamp(start_seconds)
        end_time = self.format_timestamp(end_seconds)
        
        context_frames = self.get_frames_in_range(start_time, end_time)
        
        return {
            "event_type": event_type,
            "sequence_number": sequence_number,
            "total_events_of_type": len(events),
            "target_event": target_event,
            "time_range": {
                "start": start_time,
                "end": end_time,
                "duration_seconds": context_seconds * 2,
                "center_timestamp": target_timestamp
            },
            "context_frames": context_frames,
            "frame_count": len(context_frames)
        }
    
    def get_frames_in_range(self, start_time: str, end_time: str) -> List[Dict]:
        """Get all frames in time range"""
        start_seconds = self.parse_timestamp(start_time)
        end_seconds = self.parse_timestamp(end_time)
        
        frames = []
        for segment in self.combined_data['segments']:
            for frame in segment['frames']:
                frame_seconds = self.parse_timestamp(frame['global_timestamp'])
                if start_seconds <= frame_seconds <= end_seconds:
                    frames.append(frame)
        
        frames.sort(key=lambda x: self.parse_timestamp(x['global_timestamp']))
        return frames
    
    def create_question_prompt(self, user_query: str, event_data: Dict) -> str:
        """Create prompt for answering questions"""
        target = event_data['target_event']
        frames = event_data['context_frames']
        
        event_timeline = []
        for frame in frames:
            if frame["active_events"]:
                event_timeline.append({
                    "time": frame["global_timestamp"],
                    "events": frame["active_events"]
                })
        
        prompt = f"""You are a football analyst answering a question.

USER QUESTION: "{user_query}"

EVENT: {event_data['event_type'].upper()} #{event_data['sequence_number']}
Timestamp: {target['global_timestamp']}
Note: Duplicates and replays have been filtered out

EVENT TIMELINE:
"""
        for evt in event_timeline:
            prompt += f"{evt['time']}: {', '.join(evt['events'])}\n"
        
        prompt += "\nFRAME DESCRIPTIONS:\n"
        
        for frame in frames:
            is_target = frame["global_timestamp"] == target["global_timestamp"]
            marker = ">>> TARGET <<<" if is_target else ""
            
            prompt += f"\n[{frame['global_timestamp']}] {marker}\n"
            prompt += f"{frame['rag_description']}\n"
        
        prompt += """

Answer directly based on the data above.
- Start with YES/NO if applicable
- Provide 1-2 sentences of evidence
- Keep under 100 words
"""
        return prompt
    
    def create_narrative_prompt(self, event_data: Dict) -> str:
        """Create prompt for commentary"""
        target = event_data['target_event']
        frames = event_data['context_frames']
        event_type = event_data['event_type']
        
        prompt = f"""Portugal vs Spain - 2018 UCL Final

Analyze this {event_type.upper()} #{event_data['sequence_number']} at {target['global_timestamp']}

Note: Replays and celebrations have been filtered out automatically.

SEQUENCE ({event_data['time_range']['duration_seconds']}s window):
"""
        
        for frame in frames:
            is_target = frame["global_timestamp"] == target["global_timestamp"]
            events_text = f" → {', '.join(frame['active_events'])}" if frame['active_events'] else ""
            
            if is_target:
                prompt += f"\n🎯 {frame['global_timestamp']}: THE {event_type.upper()}{events_text}"
            else:
                prompt += f"\n   {frame['global_timestamp']}{events_text}"
        
        prompt += f"""

Write 3-4 paragraphs analyzing this {event_type}:

1. THE BUILDUP: What led to this moment?
2. THE KEY MOMENT: What happened at {target['global_timestamp']}?
3. THE AFTERMATH: Immediate impact
4. TACTICAL INSIGHT: What does this reveal about the teams?

Write as a football analyst. Be direct. Start immediately.
"""
        return prompt
    
    def analyze_with_gemini(self, query: str) -> Dict[str, Any]:
        """Analyze event with Gemini"""
        query_info = self.detect_query_type(query)
        
        if "error" in query_info["event_info"]:
            return query_info["event_info"]
        
        event_data = query_info["event_info"]
        
        try:
            print(f"Analyzing: {event_data['event_type']} #{event_data['sequence_number']} "
                  f"of {event_data['total_events_of_type']} unique events...\n")
            
            if query_info["is_question"]:
                prompt = self.create_question_prompt(query, event_data)
                response_type = "answer"
            else:
                prompt = self.create_narrative_prompt(event_data)
                response_type = "commentary"
            
            response = self.gemini_model.generate_content(prompt)
            
            frame_files = [f['file'] for f in event_data['context_frames']]
            
            return {
                "query": query,
                "query_type": response_type,
                "event_type": event_data['event_type'],
                "sequence_number": event_data['sequence_number'],
                "total_events": event_data['total_events_of_type'],
                "timestamp": event_data['target_event']['global_timestamp'],
                "key_frame": event_data['target_event']['file'],
                "time_range": f"{event_data['time_range']['start']} to {event_data['time_range']['end']}",
                "duration_seconds": event_data['time_range']['duration_seconds'],
                "frame_count": event_data['frame_count'],
                "frame_files": frame_files,
                "response": response.text,
                "deduplication_applied": True,
                "success": True
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def show_event_summary(self):
        """Show all unique events"""
        print("\n" + "="*60)
        print("EVENT SUMMARY (After Smart Deduplication)")
        print("="*60)
        
        for event_type, events in sorted(self.index_data['event_index'].items()):
            if events:
                print(f"\n{event_type.upper()}: {len(events)} unique events")
                for i, event in enumerate(events[:5], 1):
                    print(f"  {i}. {event['global_timestamp']} - {event['file']}")
                if len(events) > 5:
                    print(f"  ... and {len(events) - 5} more")
        print("\n" + "="*60)
    
    def interactive_mode(self):
        """Interactive query interface"""
        print("\n" + "="*60)
        print("Football Video Analysis - Smart Deduplication Active")
        print("="*60)
        
        print("\nExamples:")
        print("  'first goal' - Analyze first unique goal")
        print("  'second goal' - Your missing goal should appear here!")
        print("  'was the first goal a penalty?' - Ask questions")
        print("  'last tackle' - Analyze final tackle")
        
        if self.index_data:
            print("\nAvailable unique events (duplicates removed):")
            for event_type, events in sorted(self.index_data['event_index'].items()):
                if events:
                    print(f"  {event_type}: {len(events)} times")
        
        print("\nCommands: 'stats', 'summary', 'help', 'quit'")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() in ['help', '?']:
                    print("\nHelp:")
                    print("  Questions: 'was first goal a penalty?'")
                    print("  Commentary: 'first goal', 'second goal'")
                    print("  Commands: 'stats', 'summary', 'quit'")
                    if self.index_data:
                        available = ', '.join([k for k, v in self.index_data['event_index'].items() if v])
                        print(f"  Events: {available}")
                
                elif user_input.lower() == 'stats':
                    if self.index_data:
                        print("\nMatch Statistics:")
                        print(f"Duration: {self.index_data['total_duration']}")
                        print("\nUnique Events (after deduplication):")
                        for event_type, events in sorted(self.index_data['event_index'].items()):
                            if events:
                                print(f"  {event_type}: {len(events)}")
                
                elif user_input.lower() == 'summary':
                    self.show_event_summary()
                
                else:
                    result = self.analyze_with_gemini(user_input)
                    
                    if "error" not in result:
                        print(f"\n{result['event_type'].upper()} #{result['sequence_number']} "
                              f"of {result['total_events']} unique events")
                        print(f"Timestamp: {result['timestamp']}")
                        print(f"Time Range: {result['time_range']}")
                        print(f"Frames: {result['frame_count']}")
                        print(f"\nKey Frame: {result['key_frame']}")
                        
                        if result['query_type'] == 'answer':
                            print(f"\nANSWER:")
                            print("="*60)
                            print(result['response'])
                            print("="*60)
                        else:
                            print(f"\nAll Frames ({len(result['frame_files'])} files):")
                            for i, frame_file in enumerate(result['frame_files'], 1):
                                print(f"  {i}. {frame_file}")
                            
                            print(f"\nCOMMENTARY:")
                            print("="*60)
                            print(result['response'])
                            print("="*60)
                    else:
                        print(f"Error: {result['error']}")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    workspace_path = r"c:\Users\SHUBHAM\Downloads\match"
    
    print("Smart Football Video Analysis")
    print("Rule-Based Deduplication (No extra API calls)")
    print("="*60)
    
    try:
        querier = SmartVideoRAGQuerier(workspace_path)
        
        # Quick test
        print("\nQuick Test:")
        result = querier.analyze_with_gemini("first goal")
        
        if "error" not in result:
            print(f"✓ First goal: {result['timestamp']}")
            print(f"✓ Frames: {result['frame_count']}")
        
        # Test second goal
        result2 = querier.analyze_with_gemini("second goal")
        if "error" not in result2:
            print(f"✓ Second goal: {result2['timestamp']}")
            print(f"✓ This should be your missing goal!\n")
        else:
            print(f"Second goal: {result2['error']}\n")
        
        querier.interactive_mode()
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()