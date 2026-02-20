from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Import the student chatbot
from student_chatbot import StudentChatbot, Config, EnhancedRAGChatbot 

app = Flask(__name__)
# Enable CORS for React frontend (running on port 3000)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:5173"]}})

# Initialize the chatbot
chatbot = None

# Path to events database created by auto_event_processor.py
EVENTS_DATABASE_PATH = Path("processed_events/events_database.json")

def initialize_chatbot():
    """Initialize the chatbot with API key"""
    global chatbot
    api_key = Config.GEMINI_API_KEY
    
    if api_key == "YOUR_API_KEY_HERE":
        print("\nâš ï¸  WARNING: Please set your Gemini API key!")
        print("Set environment variable: export GEMINI_API_KEY='your-key'")
        return False
    
    try:
        chatbot = StudentChatbot(api_key)
        print("âœ… Chatbot initialized successfully!")
        return True
    except Exception as e:
        print(f"âŒ Error initializing chatbot: {str(e)}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Flask server is running"
    }), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    Expects JSON: {"message": "user query"}
    Returns JSON: {"response": "bot response", "intent": "intent_type"}
    """
    try:
        if chatbot is None:
            return jsonify({
                "error": "Chatbot not initialized. Please check API key.",
                "response": "Sorry, the chatbot service is currently unavailable."
            }), 500
        
        # Get user message from request
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "No message provided",
                "response": "Please provide a message."
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                "error": "Empty message",
                "response": "Please enter a message."
            }), 400
        
        # Process the query through the chatbot (3-way routing)
        response = chatbot.query(user_message)
        
        # Get route type for intent field
        intent = chatbot.get_route_type(user_message).lower()
        
        # Update memory
        chatbot.memory.add_interaction(user_message, response, intent)
        
        # Return clean response
        return jsonify({
            "response": response,
            "intent": intent,
            "success": True
        }), 200
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "success": False
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        if chatbot is None:
            return jsonify({"error": "Chatbot not initialized"}), 500
        
        history = chatbot.memory.conversation_history
        return jsonify({
            "history": history,
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        if chatbot is None:
            return jsonify({"error": "Chatbot not initialized"}), 500
        
        chatbot.memory.conversation_history = []
        return jsonify({
            "message": "History cleared successfully",
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
import tempfile
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Receive file from React frontend, save it temporarily,
    pass to chatbot for processing, then generate a response.
    
    Expects: multipart/form-data with 'file' and optional 'message'
    Returns: JSON with chatbot response about the file content
    """
    try:
        if chatbot is None:
            return jsonify({
                "error": "Chatbot not initialized",
                "response": "Sorry, the chatbot service is currently unavailable."
            }), 500

        # Check file is present
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "response": "Please upload a file."
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "error": "Empty filename",
                "response": "Please select a file."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "File type not supported",
                "response": "Please upload a PDF or image file (PNG, JPG, etc.)."
            }), 400

        # Get optional message from user
        user_message = request.form.get('message', '').strip()

        # Save file temporarily
        if not file.filename:
            return jsonify({
                "error": "Invalid filename",
                "response": "The uploaded file has no name."
            }), 400
        filename = secure_filename(file.filename)
        upload_dir = Path(Config.UPLOAD_FOLDER)
        upload_dir.mkdir(exist_ok=True)
        filepath = str(upload_dir / filename)
        file.save(filepath)

        print(f"✓ File saved: {filepath}")

        # Process the file into the RAG vector DB
        success = chatbot.upload_document(filepath)

        if not success:
            return jsonify({
                "error": "Failed to process file",
                "response": "Sorry, I could not read the file. Make sure it's a valid PDF or image."
            }), 500

        # Build query — use user message if given, else ask to summarize
        query = user_message if user_message else f"Summarize the content of the uploaded file: {filename}"

        # Generate response about the file content
        response = chatbot.query(query)
        intent = chatbot.get_route_type(query).lower()
        chatbot.memory.add_interaction(query, response, intent)

        return jsonify({
            "response": response,
            "intent": intent,
            "filename": filename,
            "success": True
        }), 200

    except Exception as e:
        print(f"Error in upload endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "response": "I encountered an error processing your file. Please try again.",
            "success": False
        }), 500


@app.route('/api/announcements', methods=['GET'])
def get_announcements():
    """
    Get all event announcements from the auto event processor
    Returns: List of announcements with event details
    """
    try:
        if not EVENTS_DATABASE_PATH.exists():
            return jsonify({
                "announcements": [],
                "total": 0,
                "message": "No events database found. Run auto_event_processor.py first.",
                "success": True
            }), 200
        
        # Read events database
        with open(EVENTS_DATABASE_PATH, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        events = database.get('events', [])
        
        # Convert events to announcement format for frontend
        announcements = []
        for idx, event in enumerate(events):
            # Skip events with errors
            if "error" in event:
                continue
            
            # Format announcement
            announcement = format_event_as_announcement(event, idx + 1)
            announcements.append(announcement)
        
        return jsonify({
            "announcements": announcements,
            "total": len(announcements),
            "last_updated": database.get('metadata', {}).get('last_updated'),
            "success": True
        }), 200
        
    except Exception as e:
        print(f"Error in announcements endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "announcements": [],
            "success": False
        }), 500

@app.route('/api/announcements/latest', methods=['GET'])
def get_latest_announcements():
    """
    Get the latest N announcements
    Query param: limit (default: 5)
    """
    try:
        limit = int(request.args.get('limit', 5))
        
        if not EVENTS_DATABASE_PATH.exists():
            return jsonify({
                "announcements": [],
                "success": True
            }), 200
        
        with open(EVENTS_DATABASE_PATH, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        events = database.get('events', [])
        
        # Get latest events (sorted by processed_at)
        sorted_events = sorted(
            [e for e in events if "error" not in e],
            key=lambda x: x.get('processed_at', ''),
            reverse=True
        )
        
        # Format announcements
        announcements = []
        for idx, event in enumerate(sorted_events[:limit]):
            announcement = format_event_as_announcement(event, idx + 1)
            announcements.append(announcement)
        
        return jsonify({
            "announcements": announcements,
            "success": True
        }), 200
        
    except Exception as e:
        print(f"Error in latest announcements endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "announcements": [],
            "success": False
        }), 500

@app.route('/api/announcements/<int:event_id>', methods=['GET'])
def get_announcement_by_id(event_id):
    """
    Get a specific announcement by ID
    """
    try:
        if not EVENTS_DATABASE_PATH.exists():
            return jsonify({
                "error": "Events database not found",
                "success": False
            }), 404
        
        with open(EVENTS_DATABASE_PATH, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        events = database.get('events', [])
        
        # Find event by index (event_id is 1-based)
        if 0 < event_id <= len(events):
            event = events[event_id - 1]
            if "error" not in event:
                announcement = format_event_as_announcement(event, event_id)
                return jsonify({
                    "announcement": announcement,
                    "success": True
                }), 200
        
        return jsonify({
            "error": "Event not found",
            "success": False
        }), 404
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

def format_event_as_announcement(event: dict, event_id: int) -> dict:
    """
    Convert event data to announcement format for frontend
    """
    event_name = event.get('event_name', 'Event Announcement')
    event_type = event.get('event_type', 'General')
    dates = event.get('dates', {})
    
    # Create title
    title = f"📢 {event_name}"
    
    # Create body summary
    body_parts = []
    
    # Event type
    if event_type:
        body_parts.append(f"📌 {event_type}")
    
    # Event date
    if isinstance(dates, dict):
        event_date = dates.get('event_date', '')
        if event_date:
            body_parts.append(f"📅 {event_date}")
    
    # Venue
    venue = event.get('venue', '')
    if venue:
        body_parts.append(f"📍 {venue}")
    
    # Registration deadline
    if isinstance(dates, dict):
        reg_deadline = dates.get('registration_deadline', '')
        if reg_deadline:
            body_parts.append(f"⏰ Register by: {reg_deadline}")
    
    # Description (truncated)
    description = event.get('description', '')
    if description:
        truncated = description[:150] + '...' if len(description) > 150 else description
        body_parts.append(truncated)
    
    body = '\n'.join(body_parts)
    
    # Determine announcement type based on urgency
    announcement_type = 'info'
    if isinstance(dates, dict):
        reg_deadline = dates.get('registration_deadline', '')
        if reg_deadline:
            try:
                # Check if deadline is soon (within 7 days)
                from datetime import datetime, timedelta
                # Simple check - you can make this more sophisticated
                if 'today' in reg_deadline.lower() or 'tomorrow' in reg_deadline.lower():
                    announcement_type = 'warning'
            except:
                pass
    
    return {
        "id": str(event_id),
        "title": title,
        "body": body,
        "type": announcement_type,
        "read": False,
        "event_data": event,  # Include full event data for detailed view
        "processed_at": event.get('processed_at', '')
    }

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    
    # Initialize chatbot
    if initialize_chatbot():
        print("\nðŸš€ Starting Flask server on http://localhost:5000")
        print("ðŸ“¡ CORS enabled for React frontend")
        print("="*60)
        print("\nðŸ“¢ ANNOUNCEMENTS ENDPOINTS:")
        print("   GET  /api/announcements          - All announcements")
        print("   GET  /api/announcements/latest   - Latest N announcements")
        print("   GET  /api/announcements/<id>     - Specific announcement")
        print("="*60 + "\n")
        
        # Run Flask server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True
        )
    else:
        print("\nâŒ Failed to initialize chatbot. Server not started.")
        print("Please set your GEMINI_API_KEY environment variable.")
        sys.exit(1)