import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Set, List, Dict

from dotenv import load_dotenv
load_dotenv()

from announcement_system import AnnouncementProcessor


# Configuration
WATCH_FOLDER = "event_photos_folder"
SUPPORTED_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.webp'}
DELAY_BETWEEN_PAGES = 5
CHECK_INTERVAL = 10
MAX_RETRIES = 3
RETRY_WAIT = 40


def load_processed_files(tracker_file: Path) -> Set[str]:
    """Load list of already processed files."""
    if tracker_file.exists():
        with open(tracker_file, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_files', []))
    return set()


def save_processed_files(tracker_file: Path, processed_files: Set[str]):
    """Save list of processed files."""
    data = {
        'processed_files': sorted(list(processed_files)),
        'last_updated': datetime.now().isoformat()
    }
    with open(tracker_file, 'w') as f:
        json.dump(data, f, indent=2)


def get_new_files(watch_folder: Path, processed_files: Set[str]) -> List[Path]:
    """Get list of new unprocessed files in the watch folder."""
    if not watch_folder.exists():
        return []
    
    new_files = []
    for file_path in watch_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            if str(file_path) not in processed_files:
                new_files.append(file_path)
    
    return sorted(new_files)


def process_single_image(processor: AnnouncementProcessor, image_path: Path, 
                        output_dir: Path, file_counter: int) -> Dict:
    """Process a single image file."""
    print(f"\n{'='*70}")
    print(f"Processing Image: {image_path.name}")
    print(f"{'='*70}")
    
    event_data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            from PIL import Image
            image = Image.open(image_path)
            event_data = processor.process_with_gemini_vision(image)
            
            if isinstance(event_data, dict) and "error" in event_data:
                if any(x in str(event_data["error"]) for x in ["429", "RESOURCE_EXHAUSTED"]):
                    raise RuntimeError(event_data["error"])
            break
        except Exception as e:
            if any(x in str(e) for x in ["429", "RESOURCE_EXHAUSTED"]) and attempt < MAX_RETRIES:
                print(f"[WARN] Rate limit (attempt {attempt}). Waiting {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                event_data = None
            else:
                print(f"[ERROR] Error processing {image_path.name}: {e}")
                event_data = {"error": str(e)}
                break
    
    if event_data and "error" not in event_data:
        event_data.update({
            "source_file": str(image_path),
            "file_type": "image",
            "processed_at": datetime.now().isoformat()
        })
        
        with open(output_dir / f"event_{file_counter}.json", 'w', encoding='utf-8') as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        notification = processor.generate_notification_text(event_data)
        with open(output_dir / f"notification_{file_counter}.txt", 'w', encoding='utf-8') as f:
            f.write(notification)
        
        print(f"[OK] Processed: {event_data.get('event_name', '?')}")
        return event_data
    else:
        print(f"[ERROR] Failed to process {image_path.name}")
        return {"error": "Processing failed"}


def process_pdf_file(processor: AnnouncementProcessor, pdf_path: Path, 
                     output_dir: Path, file_counter_start: int) -> List[Dict]:
    """Process a PDF file (can have multiple pages)."""
    print(f"\n{'='*70}")
    print(f"Processing PDF: {pdf_path.name}")
    print(f"{'='*70}")
    
    images = processor.pdf_to_images(pdf_path)
    if not images:
        print(f"[ERROR] Failed to convert PDF to images: {pdf_path.name}")
        return []
    
    print(f"[OK] Found {len(images)} pages in PDF")
    
    events = []
    file_counter = file_counter_start
    
    for i, image in enumerate(images, 1):
        print(f"\n{'-'*70}")
        print(f"Processing Page {i}/{len(images)} from {pdf_path.name}")
        print(f"{'-'*70}")
        
        event_data = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                event_data = processor.process_with_gemini_vision(image)
                if isinstance(event_data, dict) and "error" in event_data:
                    if any(x in str(event_data["error"]) for x in ["429", "RESOURCE_EXHAUSTED"]):
                        raise RuntimeError(event_data["error"])
                break
            except Exception as e:
                if any(x in str(e) for x in ["429", "RESOURCE_EXHAUSTED"]) and attempt < MAX_RETRIES:
                    print(f"[WARN] Rate limit (attempt {attempt}). Waiting {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                    event_data = None
                else:
                    print(f"[ERROR] Error page {i}: {e}")
                    event_data = {"error": str(e)}
                    break
        
        if event_data and "error" not in event_data:
            event_data.update({
                "source_file": str(pdf_path),
                "source_page": i,
                "file_type": "pdf",
                "processed_at": datetime.now().isoformat()
            }) # type: ignore
            
            with open(output_dir / f"event_{file_counter}.json", 'w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, ensure_ascii=False)
            
            notification = processor.generate_notification_text(event_data)
            with open(output_dir / f"notification_{file_counter}.txt", 'w', encoding='utf-8') as f:
                f.write(notification)
            
            print(f"[OK] Page {i}: {event_data.get('event_name', '?')}")
            events.append(event_data)
            file_counter += 1
        else:
            print(f"[ERROR] Failed page {i}")
        
        if i < len(images):
            print(f"[WAIT] Waiting {DELAY_BETWEEN_PAGES}s...")
            time.sleep(DELAY_BETWEEN_PAGES)
    
    return events


def update_master_database(output_dir: Path, new_events: List[Dict]):
    """Update the master events database with new events."""
    database_file = output_dir / "events_database.json"
    
    if database_file.exists():
        with open(database_file, 'r', encoding='utf-8') as f:
            database = json.load(f)
    else:
        database = {
            "metadata": {
                "total_events": 0,
                "last_updated": datetime.now().isoformat()
            },
            "events": []
        }
    
    database["events"].extend(new_events)
    database["metadata"]["total_events"] = len(database["events"])
    database["metadata"]["last_updated"] = datetime.now().isoformat()
    
    with open(database_file, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Master database updated -> {database_file}")
    print(f"[INFO] Total events in database: {database['metadata']['total_events']}")


def generate_announcement(event: dict) -> str:
    """Generate a formatted announcement text for WhatsApp/notifications."""
    if "error" in event:
        return f"Error processing event: {event['error']}"
    
    lines = []
    
    # Header with event name
    event_name = event.get("event_name", "Event Announcement")
    lines.append("*" + "="*60 + "*")
    lines.append(f"*{event_name.upper()}*")
    lines.append("*" + "="*60 + "*")
    lines.append("")
    
    # Event type and department
    if event.get("event_type"):
        lines.append(f"*Type:* {event['event_type']}")
    if event.get("department"):
        lines.append(f"*Organized by:* {event['department']}")
    lines.append("")
    
    # Dates
    dates = event.get("dates", {})
    if dates:
        if dates.get("event_date"):
            lines.append(f"*Date:* {dates['event_date']}")
        if dates.get("end_date"):
            lines.append(f"*End Date:* {dates['end_date']}")
        if dates.get("registration_deadline"):
            lines.append(f"*Registration Deadline:* {dates['registration_deadline']}")
        if dates.get("abstract_submission"):
            lines.append(f"*Abstract Submission:* {dates['abstract_submission']}")
        lines.append("")
    
    # Time and Venue
    if event.get("time"):
        lines.append(f"*Time:* {event['time']}")
    if event.get("venue"):
        lines.append(f"*Venue:* {event['venue']}")
    if event.get("time") or event.get("venue"):
        lines.append("")
    
    # Description
    if event.get("description"):
        lines.append("*About:*")
        lines.append(event['description'])
        lines.append("")
    
    # Topics
    topics = event.get("topics")
    if topics and isinstance(topics, list) and any(topics):
        lines.append("*Topics Covered:*")
        for topic in topics:
            if topic:
                lines.append(f"  - {topic}")
        lines.append("")
    
    # Target Audience
    if event.get("target_audience"):
        lines.append(f"*Target Audience:* {event['target_audience']}")
        lines.append("")
    
    # Registration Fee
    if event.get("registration_fee"):
        lines.append(f"*Registration Fee:* {event['registration_fee']}")
    else:
        lines.append("*Registration Fee:* Free / Not Specified")
    lines.append("")
    
    # Prizes
    if event.get("prizes"):
        lines.append(f"*Prizes:* {event['prizes']}")
        lines.append("")
    
    # Contact Information
    contacts = event.get("contact_persons")
    if contacts and isinstance(contacts, list):
        lines.append("*Contact Information:*")
        for contact in contacts:
            if isinstance(contact, dict):
                name = contact.get("name", "")
                role = contact.get("role", "")
                phone = contact.get("phone", "")
                email = contact.get("email", "")
                
                if name:
                    contact_line = f"  {name}"
                    if role:
                        contact_line += f" ({role})"
                    lines.append(contact_line)
                    if phone:
                        lines.append(f"    Phone: {phone}")
                    if email:
                        lines.append(f"    Email: {email}")
            elif isinstance(contact, str):
                lines.append(f"  {contact}")
        lines.append("")
    
    # Additional Info
    if event.get("additional_info"):
        lines.append("*Additional Information:*")
        lines.append(event['additional_info'])
        lines.append("")
    
    # Footer
    lines.append("*" + "="*60 + "*")
    lines.append("Don't miss this opportunity! Register now!")
    lines.append("*" + "="*60 + "*")
    
    return "\n".join(lines)


def auto_process_events(watch_mode: bool = True):
    """Automatically process new event photos/PDFs added to the watch folder."""
    watch_folder = Path(WATCH_FOLDER)
    watch_folder.mkdir(exist_ok=True)
    
    output_dir = Path("processed_events")
    output_dir.mkdir(exist_ok=True)
    
    tracker_file = output_dir / ".processed_files_tracker.json"
    
    print("[*] Initializing Gemini AI Processor...")
    try:
        processor = AnnouncementProcessor()
        print("[OK] Processor initialized!")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    print(f"\n[*] Watching folder: {watch_folder.absolute()}")
    print(f"[*] Output folder: {output_dir.absolute()}")
    print(f"[*] Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    print(f"\n{'='*70}")
    print("DROP NEW EVENT PHOTOS/PDFs INTO THE FOLDER TO AUTO-PROCESS!")
    print(f"{'='*70}\n")
    
    if not watch_mode:
        print("[MODE] Running in single-process mode (process once and exit)...\n")
    
    processed_files = load_processed_files(tracker_file)
    file_counter = len(processed_files) + 1
    
    try:
        while True:
            new_files = get_new_files(watch_folder, processed_files)
            
            if new_files:
                print(f"\n[NEW] Found {len(new_files)} new file(s) to process!")
                all_new_events = []
                
                for file_path in new_files:
                    print(f"\n{'#'*70}")
                    print(f"NEW FILE DETECTED: {file_path.name}")
                    print(f"{'#'*70}")
                    
                    new_events = []
                    
                    if file_path.suffix.lower() == '.pdf':
                        new_events = process_pdf_file(processor, file_path, output_dir, file_counter)
                        file_counter += len(new_events)
                    else:
                        event_data = process_single_image(processor, file_path, output_dir, file_counter)
                        if "error" not in event_data:
                            new_events = [event_data]
                            file_counter += 1
                    
                    if new_events:
                        all_new_events.extend(new_events)
                    
                    processed_files.add(str(file_path))
                    save_processed_files(tracker_file, processed_files)
                    
                    print(f"\n[OK] Completed: {file_path.name}")
                
                if all_new_events:
                    update_master_database(output_dir, all_new_events)
                    
                    # Generate announcement for each new event
                    print("\n" + "="*70)
                    print("GENERATING ANNOUNCEMENTS...")
                    print("="*70)
                    
                    announcement_dir = output_dir / "announcements"
                    announcement_dir.mkdir(exist_ok=True)
                    
                    for idx, event in enumerate(all_new_events, 1):
                        announcement = generate_announcement(event)
                        
                        # Save announcement to file
                        event_name = event.get('event_name', f'Event_{idx}')
                        safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in event_name)
                        announcement_file = announcement_dir / f"{safe_name}_announcement.txt"
                        
                        with open(announcement_file, 'w', encoding='utf-8') as f:
                            f.write(announcement)
                        
                        # Display announcement
                        print(f"\n{'='*70}")
                        print(f"ANNOUNCEMENT #{idx} - {event.get('event_name', 'Event')}")
                        print(f"{'='*70}")
                        print(announcement)
                        print(f"{'='*70}")
                        print(f"[SAVED] {announcement_file}\n")
                    
                    print("\n" + "="*70)
                    print(f"[OK] BATCH COMPLETE - Processed {len(new_files)} file(s), extracted {len(all_new_events)} event(s)")
                    print(f"[OK] Announcements saved to: {announcement_dir.absolute()}")
                    print("="*70)
            
            if not watch_mode:
                break
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Stopping auto-processor...")
        print(f"[OK] Total files processed: {len(processed_files)}")
        print("\n[INFO] Restart anytime to continue watching!")


if __name__ == "__main__":
    import sys
    
    print()
    print("=" * 74)
    print("         AUTOMATIC EVENT PHOTO PROCESSOR")
    print("         Watches folder and auto-processes new files")
    print("=" * 74)
    print()
    
    watch_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        watch_mode = False
    
    if watch_mode:
        print("[MODE] CONTINUOUS WATCH MODE")
        print("   Will keep running and process new files as they appear")
        print("   Press Ctrl+C to stop\n")
    else:
        print("[MODE] SINGLE-RUN MODE")
        print("   Will process current unprocessed files and exit\n")
    
    auto_process_events(watch_mode=watch_mode)