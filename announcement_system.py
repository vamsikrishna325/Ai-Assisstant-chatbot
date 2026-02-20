
import os
import json
from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# NEW PACKAGE - google.genai instead of google.generativeai
from google import genai
from google.genai import types

from PIL import Image
import pdfplumber # type: ignore
from pdf2image import convert_from_path
import pytesseract


class AnnouncementProcessor:
    """
    Processes event posters and extracts structured event information
    """
    
    def __init__(self, api_key: str = None): # type: ignore
       
        # =====================================================================
        # OPTION A: Put your API key in the .env file as:  GOOGLE_API_KEY=your_key
        # OPTION B: Replace None below with your key in quotes:
        #           DIRECT_API_KEY = "AIzaSy..."
        # =====================================================================
        DIRECT_API_KEY = "AIzaSyC34LcjIrZ4CRU4POJ_WLlhxK0jwc9IyQo"   # e.g.  "AIzaSy..."  — or leave None to use .env
        # =====================================================================

        self.api_key = DIRECT_API_KEY or api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini with new API
        self.client = genai.Client(api_key=self.api_key)
        
        # Create output directories
        self.output_dir = Path("announcement_data")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "processed_images").mkdir(exist_ok=True)
        (self.output_dir / "extracted_data").mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
        
        return text.strip()
    
    def extract_text_with_ocr(self, image_path: Path) -> str:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"Error with OCR: {e}")
            return ""
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Image objects
        """
        try:
            images = convert_from_path(str(pdf_path), dpi=dpi)
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []
    
    def process_with_gemini_vision(self, image: Union[Path, Image.Image]) -> Dict:
        """
        Use Gemini Vision to extract event details from poster image
        
        Args:
            image: PIL Image object or path to image
            
        Returns:
            Structured event data dictionary
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Prepare prompt for event extraction
        prompt = """
        Analyze this event poster and extract ALL event details in a structured JSON format.
        
        Extract the following information:
        1. Event Name/Title
        2. Event Type (workshop, seminar, conference, competition, industrial visit, etc.)
        3. Department/Organization
        4. Date(s) - extract all dates mentioned (start date, end date, registration deadline, abstract submission, etc.)
        5. Time (if mentioned)
        6. Venue/Location
        7. Event Description/Purpose
        8. Registration Fee (if any)
        9. Contact Information (names, phone numbers, emails)
        10. Important Deadlines
        11. Prize/Rewards (if any)
        12. Organizers/Resource Persons
        13. Topics/Themes covered
        14. Target Audience
        15. Any other relevant details
        
        Return ONLY a valid JSON object with this structure:
        {
            "event_name": "...",
            "event_type": "...",
            "department": "...",
            "dates": {
                "event_date": "...",
                "end_date": "...",
                "registration_deadline": "...",
                "abstract_submission": "..."
            },
            "time": "...",
            "venue": "...",
            "description": "...",
            "registration_fee": "...",
            "contact_persons": [
                {
                    "name": "...",
                    "role": "...",
                    "phone": "...",
                    "email": "..."
                }
            ],
            "prizes": "...",
            "topics": ["...", "..."],
            "target_audience": "...",
            "additional_info": "...",
            "raw_text": "..."
        }
        
        If any field is not found, use null or empty string/array.
        Extract dates in DD-MM-YYYY format if possible.
        """
        
        try:
            # Convert PIL Image to bytes for the new API
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Use new API format
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Content(
                        role='user',
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=img_byte_arr, mime_type='image/png')
                        ]
                    )
                ]
            )
            
            # Extract JSON from response
            response_text = response.text.strip() # type: ignore
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            event_data = json.loads(response_text)
            
            return event_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            return {"error": "Failed to parse response", "raw_response": response_text if 'response_text' in locals() else str(e)}
        except Exception as e:
            print(f"Error processing with Gemini: {e}")
            return {"error": str(e)}
    
    def process_poster(self, file_path: Union[str, Path], 
                      use_vision: bool = True,
                      use_ocr_fallback: bool = True) -> Dict:
        """
        Process a poster file (PDF or image) and extract event details
        
        Args:
            file_path: Path to poster file
            use_vision: Use Gemini Vision API (recommended)
            use_ocr_fallback: Use OCR if vision fails
            
        Returns:
            Structured event data dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        file_ext = file_path.suffix.lower()
        
        # Process based on file type
        if file_ext == '.pdf':
            # Try vision API first with PDF converted to images
            if use_vision:
                images = self.pdf_to_images(file_path)
                if images:
                    print(f"Processing PDF with Gemini Vision ({len(images)} pages)...")
                    # Process first page (main poster)
                    result = self.process_with_gemini_vision(images[0])
                    if "error" not in result:
                        return result
                    print(f"Vision API failed: {result['error']}")
            
            # Fallback to text extraction
            if use_ocr_fallback:
                print("Falling back to OCR text extraction...")
                text = self.extract_text_from_pdf(file_path)
                if text:
                    return self.process_text_with_gemini(text)
        
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            # Process image directly with vision
            if use_vision:
                print("Processing image with Gemini Vision...")
                result = self.process_with_gemini_vision(file_path)
                if "error" not in result:
                    return result
                print(f"Vision API failed: {result['error']}")
            
            # Fallback to OCR
            if use_ocr_fallback:
                print("Falling back to OCR...")
                text = self.extract_text_with_ocr(file_path)
                if text:
                    return self.process_text_with_gemini(text)
        
        else:
            return {"error": f"Unsupported file type: {file_ext}"}
        
        return {"error": "Failed to extract any information"}
    
    def process_text_with_gemini(self, text: str) -> Dict:
        """
        Process extracted text with Gemini to structure event data
        
        Args:
            text: Extracted text from poster
            
        Returns:
            Structured event data dictionary
        """
        prompt = f"""
        Analyze this text extracted from an event poster and extract ALL event details in a structured JSON format.
        
        Text:
        {text}
        
        Extract the following information:
        1. Event Name/Title
        2. Event Type (workshop, seminar, conference, competition, industrial visit, etc.)
        3. Department/Organization
        4. Date(s) - extract all dates mentioned
        5. Time (if mentioned)
        6. Venue/Location
        7. Event Description/Purpose
        8. Registration Fee (if any)
        9. Contact Information (names, phone numbers, emails)
        10. Important Deadlines
        11. Prize/Rewards (if any)
        12. Organizers/Resource Persons
        13. Topics/Themes covered
        14. Target Audience
        15. Any other relevant details
        
        Return ONLY a valid JSON object with the same structure as before.
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            response_text = response.text.strip() # type: ignore
            
            # Clean markdown
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            event_data = json.loads(response_text)
            event_data["raw_text"] = text
            
            return event_data
            
        except Exception as e:
            print(f"Error processing text with Gemini: {e}")
            return {"error": str(e), "raw_text": text}
    
    def save_event_data(self, event_data: Dict, output_name: str = None) -> Path: # pyright: ignore[reportArgumentType]
        """
        Save extracted event data to JSON file
        
        Args:
            event_data: Event data dictionary
            output_name: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"event_{timestamp}"
        
        output_path = self.output_dir / "extracted_data" / f"{output_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        print(f"Event data saved to: {output_path}")
        return output_path
    
    def generate_notification_text(self, event_data: Dict) -> str:
        """
        Generate a notification text from event data
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            Formatted notification text
        """
        if "error" in event_data:
            return f"Error processing event: {event_data['error']}"
        
        notification = []
        
        # Event name
        if event_data.get("event_name"):
            notification.append(f"📢 {event_data['event_name']}")
            notification.append("=" * 50)
        
        # Event type and department
        if event_data.get("event_type"):
            notification.append(f"Type: {event_data['event_type']}")
        if event_data.get("department"):
            notification.append(f"Department: {event_data['department']}")
        
        # Dates
        if event_data.get("dates"):
            dates = event_data["dates"]
            if dates.get("event_date"):
                notification.append(f"📅 Event Date: {dates['event_date']}")
            if dates.get("end_date"):
                notification.append(f"📅 End Date: {dates['end_date']}")
            if dates.get("registration_deadline"):
                notification.append(f"⏰ Registration Deadline: {dates['registration_deadline']}")
        
        # Time and venue
        if event_data.get("time"):
            notification.append(f"🕒 Time: {event_data['time']}")
        if event_data.get("venue"):
            notification.append(f"📍 Venue: {event_data['venue']}")
        
        # Description
        if event_data.get("description"):
            notification.append(f"\nDescription:\n{event_data['description']}")
        
        # Topics
        if event_data.get("topics") and isinstance(event_data["topics"], list):
            notification.append(f"\nTopics: {', '.join(event_data['topics'])}")
        
        # Registration fee
        if event_data.get("registration_fee"):
            notification.append(f"\n💰 Registration Fee: {event_data['registration_fee']}")
        
        # Prizes
        if event_data.get("prizes"):
            notification.append(f"🏆 Prizes: {event_data['prizes']}")
        
        # Contact persons
        if event_data.get("contact_persons"):
            notification.append("\n📞 Contact Information:")
            for contact in event_data["contact_persons"]:
                if isinstance(contact, dict):
                    name = contact.get("name", "")
                    role = contact.get("role", "")
                    phone = contact.get("phone", "")
                    email = contact.get("email", "")
                    
                    contact_line = f"  • {name}"
                    if role:
                        contact_line += f" ({role})"
                    if phone:
                        contact_line += f" - {phone}"
                    if email:
                        contact_line += f" - {email}"
                    
                    notification.append(contact_line)
        
        # Target audience
        if event_data.get("target_audience"):
            notification.append(f"\n👥 Target Audience: {event_data['target_audience']}")
        
        # Additional info
        if event_data.get("additional_info"):
            notification.append(f"\nℹ️ Additional Info:\n{event_data['additional_info']}")
        
        return "\n".join(notification)
    
    def query_event_data(self, query: str, event_data: Dict) -> str:
        """
        Answer questions about event data using Gemini
        
        Args:
            query: User question
            event_data: Event data dictionary
            
        Returns:
            Answer to the query
        """
        prompt = f"""
        You are an event information assistant. Answer the user's question based on the following event data.
        
        Event Data:
        {json.dumps(event_data, indent=2)}
        
        User Question: {query}
        
        Provide a clear, concise answer. If the information is not available in the event data, say so politely.
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip() # type: ignore
        except Exception as e:
            return f"Error answering query: {e}"


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = AnnouncementProcessor()
    
    # Process a poster
    poster_path = "events_photos.pdf"  # or path to image
    
    print("Processing poster...")
    event_data = processor.process_poster(poster_path)
    
    # Save extracted data
    processor.save_event_data(event_data)
    
    # Generate notification
    notification = processor.generate_notification_text(event_data)
    print("\n" + notification)
    
    # Example query
    query = "What is the registration fee?"
    answer = processor.query_event_data(query, event_data)
    print(f"\nQ: {query}")
    print(f"A: {answer}")