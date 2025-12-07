#!/usr/bin/env python3
"""
Video Segment Processor with Global Timestamps and RAG Optimization

This script processes video segment JSON files to:
1. Add global timestamps based on segment timing
2. Update file paths to local structure
3. Optimize structure for RAG systems
4. Create searchable event summaries

Author: GitHub Copilot
Date: October 2025
"""

import json
import os
from datetime import timedelta
from typing import Dict, List, Any, Tuple
import re

class VideoSegmentProcessor:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        
        # Segment mapping based on your provided table
        self.segment_mapping = {
            1: {"start": "00:00:00", "end": "00:05:01", "duration": "05:01"},
            2: {"start": "00:05:01", "end": "00:10:00", "duration": "04:59"},
            3: {"start": "00:10:00", "end": "00:14:59", "duration": "04:59"},
            4: {"start": "00:14:59", "end": "00:19:58", "duration": "04:59"},
            5: {"start": "00:19:58", "end": "00:24:57", "duration": "04:59"},
            6: {"start": "00:24:57", "end": "00:29:57", "duration": "05:00"},
            7: {"start": "00:29:57", "end": "00:34:56", "duration": "04:59"},
            8: {"start": "00:34:56", "end": "00:39:55", "duration": "04:59"},
            9: {"start": "00:39:55", "end": "00:46:25", "duration": "06:30"}
        }
    
    def parse_timestamp(self, timestamp_str: str) -> timedelta:
        """Convert timestamp string to timedelta object"""
        # Handle formats like "00:05:01" or "00:01.000"
        if '.' in timestamp_str:
            # Format: MM:SS.mmm
            parts = timestamp_str.split(':')
            minutes = int(parts[0])
            seconds_parts = parts[1].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            return timedelta(minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        else:
            # Format: HH:MM:SS
            time_parts = timestamp_str.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    def format_timestamp(self, td: timedelta) -> str:
        """Convert timedelta to HH:MM:SS.mmm format"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def calculate_global_timestamp(self, segment_id: int, local_timestamp: str) -> str:
        """Calculate global timestamp for a frame"""
        segment_start = self.parse_timestamp(self.segment_mapping[segment_id]["start"])
        local_time = self.parse_timestamp(local_timestamp)
        global_time = segment_start + local_time
        return self.format_timestamp(global_time)
    
    def update_file_path(self, original_path: str, segment_id: int) -> str:
        """Update file path to local structure"""
        # Extract frame filename from original path
        filename = os.path.basename(original_path)
        # Update to local batch structure
        return f"batch{segment_id}/{filename}"
    
    def extract_events_summary(self, events: Dict[str, bool]) -> List[str]:
        """Extract active events for RAG optimization"""
        active_events = [event for event, is_active in events.items() if is_active]
        return active_events
    
    def create_rag_optimized_description(self, frame_data: Dict[str, Any]) -> str:
        """Create a concise description optimized for RAG retrieval"""
        events = self.extract_events_summary(frame_data["events"])
        
        # Extract key information from full description
        full_desc = frame_data["full_description"]
        
        # Try to extract scene description (between "SCENE DESCRIPTION:" and "2. QUESTIONS:")
        scene_match = re.search(r'SCENE DESCRIPTION:\s*(.*?)(?:\n\n2\.|2\. QUESTIONS:|$)', full_desc, re.DOTALL | re.IGNORECASE)
        scene_desc = scene_match.group(1).strip() if scene_match else "Football match in progress"
        
        # Create optimized description
        event_text = f"Events: {', '.join(events)}" if events else "No significant events"
        
        return f"{scene_desc} | {event_text}"
    
    def process_batch_file(self, batch_file: str, segment_id: int) -> Dict[str, Any]:
        """Process a single batch JSON file"""
        print(f"Processing {batch_file} as segment {segment_id}...")
        
        file_path = os.path.join(self.workspace_path, batch_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = {
            "segment_info": {
                "segment_id": segment_id,
                "global_start_time": self.segment_mapping[segment_id]["start"],
                "global_end_time": self.segment_mapping[segment_id]["end"],
                "duration": self.segment_mapping[segment_id]["duration"],
                "total_frames": len(data)
            },
            "frames": []
        }
        
        for frame in data:
            # Calculate global timestamp
            global_timestamp = self.calculate_global_timestamp(segment_id, frame["timestamp"])
            
            # Update file path
            updated_file_path = self.update_file_path(frame["file"], segment_id)
            
            # Create RAG-optimized description
            rag_description = self.create_rag_optimized_description(frame)
            
            # Extract active events for easy searching
            active_events = self.extract_events_summary(frame["events"])
            
            processed_frame = {
                "frame_number": frame["frame_number"],
                "segment_id": segment_id,
                "local_timestamp": frame["timestamp"],
                "global_timestamp": global_timestamp,
                "file": updated_file_path,
                "rag_description": rag_description,
                "active_events": active_events,
                "events": frame["events"],
                "full_description": frame.get("full_description", "")  # Keep original for detailed analysis
            }
            
            processed_data["frames"].append(processed_frame)
        
        return processed_data
    
    def create_global_index(self, all_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a global index for RAG system optimization"""
        global_index = {
            "total_segments": len(all_segments),
            "total_duration": self.segment_mapping[len(all_segments)]["end"],
            "event_index": {},
            "timestamp_index": [],
            "frame_count_by_segment": {}
        }
        
        # Build event index and timestamp index
        for segment_data in all_segments:
            segment_id = segment_data["segment_info"]["segment_id"]
            global_index["frame_count_by_segment"][segment_id] = segment_data["segment_info"]["total_frames"]
            
            for frame in segment_data["frames"]:
                # Add to timestamp index
                global_index["timestamp_index"].append({
                    "global_timestamp": frame["global_timestamp"],
                    "segment_id": segment_id,
                    "frame_number": frame["frame_number"],
                    "file": frame["file"],
                    "active_events": frame["active_events"]
                })
                
                # Build event index
                for event in frame["active_events"]:
                    if event not in global_index["event_index"]:
                        global_index["event_index"][event] = []
                    
                    global_index["event_index"][event].append({
                        "global_timestamp": frame["global_timestamp"],
                        "segment_id": segment_id,
                        "frame_number": frame["frame_number"],
                        "file": frame["file"]
                    })
        
        return global_index
    
    def process_all_batches(self, batch_files: List[Tuple[str, int]]) -> None:
        """Process all batch files and create output"""
        all_segments = []
        
        for batch_file, segment_id in batch_files:
            processed_data = self.process_batch_file(batch_file, segment_id)
            all_segments.append(processed_data)
            
            # Save individual processed file
            output_file = f"processed_{batch_file}"
            output_path = os.path.join(self.workspace_path, output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved processed data to {output_file}")
        
        # Create global index
        global_index = self.create_global_index(all_segments)
        
        # Save global index
        index_path = os.path.join(self.workspace_path, "global_video_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(global_index, f, indent=2, ensure_ascii=False)
        
        print(f"Saved global index to global_video_index.json")
        
        # Save combined data
        combined_data = {
            "metadata": {
                "total_segments": len(all_segments),
                "processing_date": "2025-10-13",
                "total_duration": self.segment_mapping[len(all_segments)]["end"]
            },
            "segments": all_segments,
            "global_index": global_index
        }
        
        combined_path = os.path.join(self.workspace_path, "combined_video_data.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved combined data to combined_video_data.json")
    
    def search_by_event(self, event_type: str, index_file: str = "global_video_index.json") -> List[Dict[str, Any]]:
        """Search for frames by event type (for RAG queries)"""
        index_path = os.path.join(self.workspace_path, index_file)
        
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        return index_data["event_index"].get(event_type, [])
    
    def search_by_timerange(self, start_time: str, end_time: str, index_file: str = "global_video_index.json") -> List[Dict[str, Any]]:
        """Search for frames within a time range (for RAG queries)"""
        start_td = self.parse_timestamp(start_time)
        end_td = self.parse_timestamp(end_time)
        
        index_path = os.path.join(self.workspace_path, index_file)
        
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        results = []
        for frame_info in index_data["timestamp_index"]:
            frame_td = self.parse_timestamp(frame_info["global_timestamp"])
            if start_td <= frame_td <= end_td:
                results.append(frame_info)
        
        return results


def main():
    """Main function to process video segments"""
    # Configuration
    workspace_path = r"c:\Users\SHUBHAM\Downloads\match"
    
    # Define batch files and their corresponding segment IDs
    batch_files = [
        ("batch1.json", 1),
        ("batch2.json", 2),
        ("batch3.json", 3),
        ("batch4.json", 4),
        ("batch5.json", 5),
        ("batch6.json", 6),
        ("batch7.json", 7),
        ("batch8.json", 8),
        ("batch9.json", 9)
    ]
    
    # Initialize processor
    processor = VideoSegmentProcessor(workspace_path)
    
    # Process all batches
    processor.process_all_batches(batch_files)
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print("\nFiles created:")
    print("1. processed_batch1.json - Batch 1 with global timestamps")
    print("2. processed_batch2.json - Batch 2 with global timestamps")
    print("3. global_video_index.json - Index for RAG queries")
    print("4. combined_video_data.json - All data combined")
    
    print("\nRAG Optimization Features:")
    print("- Global timestamps for exact time references")
    print("- Event indexing for quick event searches")
    print("- Concise descriptions for better retrieval")
    print("- Local file paths (batch1/frame_xxxx.jpg)")
    print("- Active events list for each frame")
    
    # Example searches
    print("\n" + "="*30)
    print("Example RAG Queries:")
    print("="*30)
    
    try:
        # Search for goal events
        goal_events = processor.search_by_event("goal")
        print(f"Found {len(goal_events)} goal events")
        
        # Search for pass events
        pass_events = processor.search_by_event("pass")
        print(f"Found {len(pass_events)} pass events")
        
        # Search in first minute
        first_minute = processor.search_by_timerange("00:00:00.000", "00:01:00.000")
        print(f"Found {len(first_minute)} frames in first minute")
        
    except FileNotFoundError:
        print("Run the script to generate index files first!")


if __name__ == "__main__":
    main()