import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import timedelta  # Added for time calculations

import google.generativeai as genai
from config import GEMINI_API_KEY


class VideoRAGQuerier:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.index_data = None
        self.combined_data = None
        self.commentary_data = None  # New: Football commentary data
        self.gemini_model = None
        self.dedup_event_index = None  # New: Deduplicated event index
        self.load_data()
        self.setup_gemini_api()
    
    def load_data(self):
        """Load the processed video data and perform deduplication"""
        try:
            index_path = os.path.join(self.workspace_path, "global_video_index.json")
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)
            
            combined_path = os.path.join(self.workspace_path, "combined_video_data.json")
            with open(combined_path, 'r', encoding='utf-8') as f:
                self.combined_data = json.load(f)
            
            # New: Load football commentary
            commentary_path = os.path.join(self.workspace_path, "football_commentary.json")
            if os.path.exists(commentary_path):
                with open(commentary_path, 'r', encoding='utf-8') as f:
                    self.commentary_data = json.load(f)
                print("Football commentary loaded successfully")
            else:
                print("Warning: football_commentary.json not found")
                self.commentary_data = None
            
            print("Video data loaded successfully")
            print(f"Total segments: {self.index_data['total_segments']}")
            print(f"Total duration: {self.index_data['total_duration']}")
            print(f"Total frames: {sum(self.index_data.get('frame_count_by_segment', {}).values())}")
            
            # New: Deduplicate events after loading
            self.dedup_event_index = self.deduplicate_events(self.index_data["event_index"])
            print("Event deduplication complete")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run process_video_segments.py first!")
            raise
    
    def deduplicate_events(self, event_index: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Deduplicate events by clustering close timestamps (handles overlaps and replays)"""
        dedup_index = {}
        cluster_threshold = timedelta(seconds=5)  # Adjust: Group if within 5s (actual events are short, replays farther)
        
        for event_type, events in event_index.items():
            if not events:
                continue
            
            # Sort by timestamp
            sorted_events = sorted(events, key=lambda e: self.parse_timestamp(e["global_timestamp"]))
            
            clusters = []
            current_cluster = [sorted_events[0]]
            
            for event in sorted_events[1:]:
                prev_ts = self.parse_timestamp(current_cluster[-1]["global_timestamp"])
                curr_ts = self.parse_timestamp(event["global_timestamp"])
                if timedelta(seconds=curr_ts - prev_ts) <= cluster_threshold:
                    current_cluster.append(event)
                else:
                    # End cluster, pick representative (e.g., middle one)
                    mid_idx = len(current_cluster) // 2
                    clusters.append(current_cluster[mid_idx])
                    current_cluster = [event]
            
            # Add last cluster
            if current_cluster:
                mid_idx = len(current_cluster) // 2
                clusters.append(current_cluster[mid_idx])
            
            dedup_index[event_type] = clusters
            print(f"Deduplicated {event_type}: {len(events)} -> {len(clusters)}")
        
        return dedup_index
    
    def setup_gemini_api(self):
        """Setup Gemini AI API"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("Gemini 2.0 Flash configured successfully\n")
        except Exception as e:
            print(f"Gemini setup failed: {e}")
            raise
    
    def detect_query_type(self, query: str) -> Dict[str, Any]:
        """Detect if query is a question, single event, or 'all' request"""
        query_lower = query.lower().strip()
        
        # Question patterns
        question_words = ['is', 'was', 'did', 'does', 'can', 'could', 'would', 'should', 
                          'what', 'when', 'where', 'who', 'why', 'how', 'which']
        
        is_question = any(query_lower.startswith(qw) for qw in question_words) or '?' in query
        
        # New: Check for 'all' queries
        is_all = 'all' in query_lower or 'every' in query_lower or 'list' in query_lower
        
        # Extract event from query
        event_info = self.quick_event_query(query, is_all=is_all)
        
        return {
            "is_question": is_question,
            "is_all": is_all,
            "user_query": query,
            "event_info": event_info
        }
    
    def quick_event_query(self, query: str, is_all: bool = False) -> Dict[str, Any]:
        """Handle natural language queries, including 'all'"""
        query = query.lower().strip()
        
        event_mapping = {
            'goal': 'goal', 'goals': 'goal',
            'pass': 'pass', 'passes': 'pass', 'passing': 'pass',
            'tackle': 'tackle', 'tackles': 'tackle', 'tackling': 'tackle',
            'save': 'save', 'saves': 'save', 'saving': 'save',
            'card': 'card', 'cards': 'card', 'yellow': 'card', 'red': 'card',
            'foul': 'foul', 'fouls': 'foul', 'fouling': 'foul',
            'celebration': 'celebration', 'celebrating': 'celebration',
            'goal_attempt': 'goal_attempt', 'shot': 'goal_attempt', 'shots': 'goal_attempt',
            'free_kick': 'free_kick', 'free kick': 'free_kick',
            'corner_kick': 'corner_kick', 'corner': 'corner_kick',
            'dribble': 'dribble', 'dribbling': 'dribble'
        }
        
        event_type = None
        for word in query.split():
            if word in event_mapping:
                event_type = event_mapping[word]
                break
        
        # Check for multi-word events
        if not event_type:
            if 'free kick' in query or 'free_kick' in query:
                event_type = 'free_kick'
            elif 'corner kick' in query or 'corner_kick' in query:
                event_type = 'corner_kick'
            elif 'goal attempt' in query or 'goal_attempt' in query:
                event_type = 'goal_attempt'
        
        if not event_type:
            return {
                "error": f"Could not identify event type. Available: {list(set(event_mapping.values()))}"
            }
        
        # New: Handle 'with celebrations' or related
        with_celebrations = 'with celebrations' in query or 'celebrations' in query
        
        if is_all:
            # Return all deduplicated events
            return self.get_all_events(event_type, with_celebrations=with_celebrations)
        
        sequence_number = 1
        
        if any(word in query for word in ['first', '1st', 'initial']):
            sequence_number = 1
        elif any(word in query for word in ['second', '2nd']):
            sequence_number = 2
        elif any(word in query for word in ['third', '3rd']):
            sequence_number = 3
        elif any(word in query for word in ['fourth', '4th']):
            sequence_number = 4
        elif any(word in query for word in ['last', 'final', 'latest']):
            events = self.dedup_event_index.get(event_type, [])  # Use dedup
            sequence_number = len(events) if events else 1
        
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            sequence_number = int(numbers[0])
        
        return self.get_event_sequence(event_type, sequence_number)
    
    def get_all_events(self, event_type: str, with_celebrations: bool = False) -> Dict[str, Any]:
        """Get all deduplicated events of a type, optionally filtered with celebrations"""
        if not self.dedup_event_index:
            return {"error": "No data loaded"}
        
        events = self.dedup_event_index.get(event_type, [])
        
        all_events = []
        for i, event in enumerate(events, 1):
            event_data = self.get_event_sequence(event_type, i, from_all=True)
            if with_celebrations:
                # Check if celebration in context frames
                has_cele = any("celebration" in frame.get("active_events", []) for frame in event_data.get("context_frames", []))
                if not has_cele:
                    continue
            all_events.append(event_data)
        
        return {
            "event_type": event_type,
            "all_events": all_events,
            "total": len(all_events),
            "with_celebrations": with_celebrations
        }
    
    def get_event_sequence(self, event_type: str, sequence_number: int = 1, 
                           context_seconds: int = 10, from_all: bool = False) -> Dict[str, Any]:
        """Get specific event occurrence with time range and context (uses dedup index)"""
        if not self.dedup_event_index:
            return {"error": "No data loaded"}
        
        events = self.dedup_event_index.get(event_type, [])
        
        if len(events) < sequence_number:
            return {
                "error": f"Only {len(events)} {event_type} events found, requested #{sequence_number}",
                "available_events": len(events)
            }
        
        target_event = events[sequence_number - 1]
        target_timestamp = target_event["global_timestamp"]
        
        target_seconds = self.parse_timestamp(target_timestamp)
        start_seconds = max(0, target_seconds - context_seconds)
        end_seconds = target_seconds + context_seconds
        
        start_time_str = self.format_timestamp(start_seconds)
        end_time_str = self.format_timestamp(end_seconds)
        
        context_frames = self.get_frames_in_range(start_time_str, end_time_str)
        context_commentary = self.get_commentary_in_range(start_time_str, end_time_str)  # New: Get commentary
        
        # New: Optional Gemini check for replay (if descriptions mention it)
        if not from_all:  # Skip for 'all' to avoid overhead
            is_replay = self.check_if_replay(context_frames)
            if is_replay:
                print(f"Warning: This might be a replay for {event_type} #{sequence_number}")
        
        return {
            "event_type": event_type,
            "sequence_number": sequence_number,
            "total_events_of_type": len(events),
            "target_event": target_event,
            "time_range": {
                "start": start_time_str,
                "end": end_time_str,
                "duration_seconds": context_seconds * 2,
                "center_timestamp": target_timestamp
            },
            "context_frames": context_frames,
            "context_commentary": context_commentary,  # New: Add commentary
            "frame_count": len(context_frames),
            "commentary_count": len(context_commentary)  # New: Add commentary count
        }
    
    def check_if_replay(self, frames: List[Dict]) -> bool:
        """Use Gemini to check if context suggests a replay (based on descriptions)"""
        if not self.gemini_model:
            return False
        
        prompt = "Review these frame descriptions. Is this likely a replay? YES/NO, explain briefly.\n"
        for frame in frames:
            prompt += f"{frame['global_timestamp']}: {frame['rag_description']}\n"
        
        response = self.gemini_model.generate_content(prompt)
        return "YES" in response.text.upper()
    
    def get_frames_in_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """Get all frames with full details in the given time range"""
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
    
    def get_commentary_in_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """Get all commentary utterances in the given time range"""
        if not self.commentary_data:
            return []
        
        start_seconds = self.parse_timestamp(start_time)
        end_seconds = self.parse_timestamp(end_time)
        
        commentaries = []
        for segment in self.commentary_data["segments"]:
            for utterance in segment["commentary_segments"]:
                # Parse the timestamp range (e.g., "00:00:00 - 00:00:04")
                timestamp_range = utterance["global_timestamp"]
                times = timestamp_range.split(" - ")
                if len(times) == 2:
                    utterance_start = self.parse_timestamp(times[0].strip())
                    utterance_end = self.parse_timestamp(times[1].strip())
                    
                    # Check if this utterance overlaps with our time range
                    if (utterance_start <= end_seconds and utterance_end >= start_seconds):
                        commentaries.append(utterance)
        
        commentaries.sort(key=lambda x: self.parse_timestamp(x["global_timestamp"].split(" - ")[0]))
        return commentaries
    
    def parse_timestamp(self, ts: str) -> float:
        """Convert timestamp string to seconds"""
        parts = ts.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def create_question_prompt(self, user_query: str, event_data: Dict) -> str:
        """Create prompt for answering specific questions about an event"""
        # Updated: Add note about scorecard/replays (assume descriptions include them)
        target = event_data['target_event']
        frames = event_data['context_frames']
        commentaries = event_data.get('context_commentary', [])  # New: Get commentary
        
        event_timeline = []
        for frame in frames:
            if frame["active_events"]:
                event_timeline.append({
                    "time": frame["global_timestamp"],
                    "events": frame["active_events"]
                })
        
        prompt = f"""You are a football analyst answering a specific question about a match event.
Consider scorecard changes and replay indicators in descriptions to confirm real events.

USER QUESTION: "{user_query}"

EVENT BEING ASKED ABOUT:
- Type: {event_data['event_type'].upper()}
- Occurrence: #{event_data['sequence_number']} 
- Timestamp: {target['global_timestamp']}
- Time Range: {event_data['time_range']['start']} to {event_data['time_range']['end']}

EVENT TIMELINE (All tagged events in this time range):
"""
        
        for evt in event_timeline:
            prompt += f"{evt['time']}: {', '.join(evt['events'])}\n"
        
        # New: Add commentary context
        if commentaries:
            prompt += f"""

LIVE COMMENTARY (Audio commentary from broadcast):
"""
            for comm in commentaries:
                prompt += f"[{comm['global_timestamp']}] {comm['full_description']}\n"
        
        prompt += f"""

FRAME DESCRIPTIONS:
"""
        
        for frame in frames:
            is_target = frame["global_timestamp"] == target["global_timestamp"]
            marker = ">>> TARGET EVENT <<<" if is_target else ""
            
            prompt += f"\n[{frame['global_timestamp']}] {marker}\n"
            prompt += f"{frame['rag_description']}\n"
            if frame["active_events"]:
                prompt += f"Events: {', '.join(frame['active_events'])}\n"
        
        prompt += f"""

INSTRUCTIONS:
Answer the user's question directly and specifically based ONLY on the data provided above.
Use both the visual frame descriptions AND the live commentary to provide a complete answer.

Key points:
1. Look at the EVENT TIMELINE to see what events actually occurred
2. Check the LIVE COMMENTARY for context and play-by-play details
3. Check if the events the user is asking about are present or not
4. Answer with YES/NO if it's a yes/no question
5. Provide brief explanation using the event tags, descriptions, and commentary
6. If the question asks about something not in the data, clearly state "Based on the available data, [answer]"
7. Be concise and direct - answer the question, don't write unnecessary commentary
8. If descriptions suggest a replay (e.g., no scorecard change), note it.

Answer format:
- Start with a direct answer to the question
- Then provide 1-2 sentences of supporting evidence from the event timeline and commentary
- Keep total response under 100 words unless more detail is specifically requested"""
        
        return prompt
    
    def create_narrative_prompt(self, event_data: Dict) -> str:
        """Create direct analysis prompt for Portugal vs Spain 2018 UCL Final"""
        # Updated: Add scorecard/replay awareness
        target = event_data['target_event']
        frames = event_data['context_frames']
        commentaries = event_data.get('context_commentary', [])  # New: Get commentary
        event_type = event_data['event_type']
        
        prompt = f"""PORTUGAL vs SPAIN - 2018 UEFA Champions League Final
Consider scorecard and if this is live or replay based on descriptions.

Analyze this {event_type.upper()} #{event_data['sequence_number']} at {target['global_timestamp']}

Here's what happened in the {event_data['time_range']['duration_seconds']} seconds around this moment:

SEQUENCE OF EVENTS:
"""
        
        for frame in frames:
            is_target = frame["global_timestamp"] == target["global_timestamp"]
            
            if frame["active_events"]:
                events_text = f" → {', '.join(frame['active_events'])}"
            else:
                events_text = ""
            
            if is_target:
                prompt += f"\n🎯 {frame['global_timestamp']}: THE {event_type.upper()}{events_text}"
            else:
                prompt += f"\n   {frame['global_timestamp']}{events_text}"
        
        # New: Add commentary context
        if commentaries:
            prompt += f"""

LIVE COMMENTARY (What the commentators said):
"""
            for comm in commentaries:
                prompt += f"\n[{comm['global_timestamp']}] {comm['full_description']}"
        
        prompt += f"""

Now write a clear, direct analysis of how this {event_type} happened in 3-4 paragraphs:

1. **THE BUILDUP**: What events led to this scoring opportunity? Look at the timeline and commentary above.

2. **THE {event_type.upper()}**: Describe how it was executed at {target['global_timestamp']}. Use both visual descriptions and commentary.

3. **THE AFTERMATH**: What happened immediately after - celebrations, reactions, momentum shift. Reference the commentary.

4. **TACTICAL INSIGHT**: What does this tell us about Portugal vs Spain's playing styles in this UCL Final?

Write as if you're a football analyst explaining this moment to viewers. Be direct and engaging. 
Use BOTH the event timeline AND the live commentary to create a rich narrative.
The commentary provides play-by-play context that enhances the visual analysis.
Don't complain about missing information, just work with what you have. If it seems like a replay, note it.

Start immediately with the analysis - no meta-commentary about the data."""
        
        return prompt
    
    def create_all_narrative_prompt(self, all_events: List[Dict]) -> str:
        """New: Prompt for narrating all events in natural language"""
        event_type = all_events[0]['event_type'] if all_events else ""
        prompt = f"""Summarize all {event_type.upper()}s in the match in natural language.

Events:
"""
        for evt in all_events:
            target = evt['target_event']
            commentary_count = evt.get('commentary_count', 0)
            prompt += f"- #{evt['sequence_number']} at {target['global_timestamp']}: Time range {evt['time_range']['start']} to {evt['time_range']['end']} ({commentary_count} commentary utterances)\n"
        
        prompt += """
For each, provide a short narrative (1-2 paragraphs) covering buildup, execution, and aftermath.
Use BOTH the visual frame descriptions AND the live commentary to create rich narratives.
The commentary provides play-by-play context that enhances the visual analysis.
If celebrations are present, highlight them.
Output as a numbered list, engaging like a match report."""
        return prompt
    
    def analyze_with_gemini(self, query: str) -> Dict[str, Any]:
        """Analyze event with Gemini AI, handling 'all' queries"""
        if not self.gemini_model:
            return {"error": "Gemini not configured"}
        
        # Detect query type
        query_info = self.detect_query_type(query)
        
        if "error" in query_info["event_info"]:
            return query_info["event_info"]
        
        event_data = query_info["event_info"]
        
        try:
            if query_info["is_all"]:
                # New: Handle all events with detailed output for each
                all_events = event_data["all_events"]
                print(f"Analyzing all {event_data['event_type']} ({len(all_events)} found)...\n")
                prompt = self.create_all_narrative_prompt(all_events)
                response = self.gemini_model.generate_content(prompt)
                
                # Prepare detailed data for each event
                detailed_events = []
                for evt in all_events:
                    frame_files = [frame['file'] for frame in evt['context_frames']]
                    detailed_events.append({
                        "sequence_number": evt['sequence_number'],
                        "timestamp": evt['target_event']['global_timestamp'],
                        "key_frame": evt['target_event']['file'],
                        "time_range": f"{evt['time_range']['start']} to {evt['time_range']['end']}",
                        "duration_seconds": evt['time_range']['duration_seconds'],
                        "frame_count": evt['frame_count'],
                        "commentary_count": evt.get('commentary_count', 0),
                        "frame_files": frame_files,
                        "context_frames": evt['context_frames'],
                        "context_commentary": evt.get('context_commentary', [])
                    })
                
                return {
                    "query": query,
                    "query_type": "all_commentary",
                    "event_type": event_data['event_type'],
                    "total": event_data['total'],
                    "all_events": detailed_events,
                    "response": response.text,
                    "success": True
                }
            else:
                print(f"Analyzing: {event_data['event_type']} #{event_data['sequence_number']}...\n")
                
                # Choose appropriate prompt based on query type
                if query_info["is_question"]:
                    prompt = self.create_question_prompt(query, event_data)
                    response_type = "answer"
                else:
                    prompt = self.create_narrative_prompt(event_data)
                    response_type = "commentary"
                
                response = self.gemini_model.generate_content(prompt)
                
                frame_files = [frame['file'] for frame in event_data['context_frames']]
                
                return {
                    "query": query,
                    "query_type": response_type,
                    "event_type": event_data['event_type'],
                    "sequence_number": event_data['sequence_number'],
                    "timestamp": event_data['target_event']['global_timestamp'],
                    "key_frame": event_data['target_event']['file'],
                    "time_range": f"{event_data['time_range']['start']} to {event_data['time_range']['end']}",
                    "duration_seconds": event_data['time_range']['duration_seconds'],
                    "frame_count": event_data['frame_count'],
                    "commentary_count": event_data.get('commentary_count', 0),  # New: Add commentary count
                    "frame_files": frame_files,
                    "context_frames": event_data['context_frames'],
                    "context_commentary": event_data.get('context_commentary', []),
                    "response": response.text,
                    "success": True
                }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def interactive_mode(self):
        """Interactive query interface"""
        print("\nFootball Video Analysis - Interactive Mode")
        print("=" * 60)
        
        print("\nQuery Examples:")
        print("  'first goal' - Get commentary for first goal")
        print("  'all goals' - List all goals with narratives")
        print("  'all goals with celebrations' - List goals that include celebrations")
        print("  'was the first goal a free kick?' - Ask specific questions")
        print("  'how did the second pass happen?' - Get detailed analysis")
        print("  'last tackle' - Analyze the final tackle")
        
        if self.index_data:
            print("\nAvailable events (deduplicated):")
            for event_type, occurrences in self.dedup_event_index.items():
                if len(occurrences) > 0:
                    print(f"  {event_type}: {len(occurrences)} times")
        
        print("\nCommands: 'stats', 'help', 'quit'")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() in ['help', '?']:
                    print("\nHelp:")
                    print("Ask questions: 'was the first goal a free kick?'")
                    print("Get commentary: 'first goal', 'second pass'")
                    print("Get all: 'all goals', 'all goals with celebrations'")
                    print("Commands: 'stats', 'quit'")
                    if self.dedup_event_index:
                        available = ', '.join([k for k, v in self.dedup_event_index.items() if v])
                        print(f"Available events: {available}")
                
                elif user_input.lower() == 'stats':
                    if self.dedup_event_index:
                        print("\nVideo Statistics (deduplicated):")
                        print(f"Duration: {self.index_data['total_duration']}")
                        for event_type, occurrences in self.dedup_event_index.items():
                            if len(occurrences) > 0:
                                print(f"  {event_type}: {len(occurrences)} times")
                
                else:
                    result = self.analyze_with_gemini(user_input)
                    
                    if "error" not in result:
                        if result['query_type'] == 'all_commentary':
                            print(f"\n{'=' * 60}")
                            print(f"All {result['event_type'].upper()}s ({result['total']} found)")
                            print(f"{'=' * 60}\n")
                            
                            # Print Gemini's narrative first
                            print(result['response'])
                            print(f"\n{'=' * 60}")
                            print("DETAILED BREAKDOWN:")
                            print(f"{'=' * 60}\n")
                            
                            # Then print detailed info for each event
                            for evt in result['all_events']:
                                print(f"\n{'-' * 60}")
                                print(f"{result['event_type'].upper()} #{evt['sequence_number']}")
                                print(f"Timestamp: {evt['timestamp']}")
                                print(f"Duration: {evt['duration_seconds']} seconds")
                                print(f"Time Range: {evt['time_range']}")
                                print(f"Frames: {evt['frame_count']} frames")
                                print(f"Commentary: {evt['commentary_count']} utterances")
                                print(f"\nKey Frame: {evt['key_frame']}")
                                
                                print(f"\nAll Frame Files ({len(evt['frame_files'])} files):")
                                for i, frame_file in enumerate(evt['frame_files'], 1):
                                    print(f"  {i}. {frame_file}")
                                
                                if evt.get('context_commentary'):
                                    print(f"\nCOMMENTARY UTTERANCES:")
                                    for comm in evt['context_commentary']:
                                        print(f"  [{comm['global_timestamp']}] {comm['full_description']}")
                            
                            print(f"\n{'=' * 60}")
                        else:
                            print(f"\n{'=' * 60}")
                            print(f"{result['event_type'].upper()} #{result['sequence_number']}")
                            print(f"Timestamp: {result['timestamp']}")
                            print(f"Duration: {result['duration_seconds']} seconds")
                            print(f"Time Range: {result['time_range']}")
                            print(f"Frames: {result['frame_count']} frames")
                            print(f"Commentary: {result.get('commentary_count', 0)} utterances")
                            print(f"\nKey Frame: {result['key_frame']}")
                            
                            print(f"\nAll Frame Files ({len(result['frame_files'])} files):")
                            for i, frame_file in enumerate(result['frame_files'], 1):
                                print(f"  {i}. {frame_file}")
                            
                            if result.get('context_commentary'):
                                print(f"\nCOMMENTARY:")
                                for comm in result['context_commentary']:
                                    print(f"  [{comm['global_timestamp']}] {comm['full_description']}")
                            
                            print(f"\n{'=' * 60}")
                            if result['query_type'] == 'answer':
                                print("ANSWER:")
                                print("=" * 60)
                            else:
                                print("ANALYSIS:")
                                print("=" * 60)
                            print(result['response'])
                            print(f"{'=' * 60}")
                    else:
                        print(f"Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function"""
    workspace_path = r"C:\Users\SHUBHAM\Downloads\match\backend"
    
    print("Football Video Analysis with Gemini AI")
    print("=" * 60)
    
    try:
        querier = VideoRAGQuerier(workspace_path)
        
        if querier.index_data:
            print("\nQuick Demo:")
            result = querier.analyze_with_gemini("first goal")
            
            if "error" not in result:
                print(f"First goal found at: {result['timestamp']}")
                print(f"Duration: {result['duration_seconds']} seconds")
                print(f"Frames analyzed: {result['frame_count']}\n")
            
            querier.interactive_mode()
        else:
            print("No video data found. Run process_video_segments.py first!")
    
    except Exception as e:
        print(f"Failed to initialize: {e}")


if __name__ == "__main__":
    main()