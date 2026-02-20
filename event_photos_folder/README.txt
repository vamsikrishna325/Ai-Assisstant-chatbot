
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
