

from pathlib import Path

def setup_auto_processor():
    """Create necessary folders and display setup instructions."""
    
    print("=" * 74)
    print("         AUTO EVENT PROCESSOR - QUICK SETUP")
    print("=" * 74)
    print()
    
    # Create folders
    watch_folder = Path("event_photos_folder")
    output_folder = Path("processed_events")
    
    watch_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)
    
    print("✅ Created folders:")
    print(f"   📁 {watch_folder.absolute()}")
    print(f"   📂 {output_folder.absolute()}")
    print()
    
    # Create a sample README in the watch folder
    readme_path = watch_folder / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""
========================================================================
              DROP EVENT PHOTOS/PDFs HERE
========================================================================

Supported Formats:
  * PDF files (.pdf) - all pages will be processed
  * Images (.png, .jpg, .jpeg, .webp)

How to Use:
  1. Start the auto-processor:
     python auto_event_processor.py

  2. Drop your event photos/PDFs into this folder

  3. The system will automatically:
     - Detect new files
     - Process them with Gemini AI
     - Extract event details
     - Update the database

  4. Query your events:
     python batch_processor.py query all events
     python batch_processor.py query <event name>

That's it! The system handles everything automatically.
""")
    
    print()
    print("   1)  START THE AUTO-PROCESSOR:")
    print("      python auto_event_processor.py")
    print()
    print("   2)  ADD EVENT PHOTOS/PDFs:")
    print(f"      Copy files into: {watch_folder.absolute()}")
    print()
    print("   3)  WATCH IT PROCESS AUTOMATICALLY!")
    print()
    print("   4)  QUERY YOUR EVENTS:")
    print("      python batch_processor.py query all events")
    print("      python batch_processor.py query <event name>")
    print()
    print("=" * 74)
    print("SETUP COMPLETE!")
    print("=" * 74)
    print()
    print("Next step: python auto_event_processor.py")
    print()

if __name__ == "__main__":
    setup_auto_processor()