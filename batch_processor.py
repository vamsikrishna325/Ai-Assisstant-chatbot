
import os
import json
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from announcement_system import AnnouncementProcessor


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def batch_process_event_photos(pdf_path: str = "events_photos.pdf"):
    DELAY_BETWEEN_PAGES = 5
    MAX_RETRIES         = 3
    RETRY_WAIT          = 40

    print("=" * 70)
    print("EVENT ANNOUNCEMENT BATCH PROCESSOR")
    print("=" * 70)

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: File not found - {pdf_path}")
        return

    print("\n🔧 Initializing Gemini AI Processor...")
    try:
        processor = AnnouncementProcessor()
        print("✅ Processor initialized!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print(f"\n📄 Processing PDF: {pdf_path}")
    images = processor.pdf_to_images(pdf_file)
    if not images:
        print("❌ Failed to convert PDF to images")
        return

    total_pages = len(images)
    est = total_pages * DELAY_BETWEEN_PAGES
    print(f"✅ Found {total_pages} pages — ~{est//60}m {est%60}s estimated\n")

    output_dir = Path("processed_events")
    output_dir.mkdir(exist_ok=True)

    all_events, successful, failed = [], 0, 0

    for i, image in enumerate(images, 1):
        print(f"\n{'='*70}\nProcessing Page {i}/{total_pages}\n{'='*70}")

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
                    print(f"⚠️  Rate limit (attempt {attempt}). Waiting {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                    event_data = None
                else:
                    print(f"❌ Error page {i}: {e}")
                    event_data = {"error": str(e)}
                    break

        if event_data is None or "error" in event_data:
            print(f"❌ Skipping page {i}")
            failed += 1
        else:
            event_data.update({"source_page": i, "source_file": pdf_path,
                               "processed_at": datetime.now().isoformat()}) # type: ignore
            with open(output_dir / f"event_page_{i}.json", 'w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Page {i}: {event_data.get('event_name','?')}")
            notification = processor.generate_notification_text(event_data)
            with open(output_dir / f"notification_page_{i}.txt", 'w', encoding='utf-8') as f:
                f.write(notification)
            all_events.append(event_data)
            successful += 1

        if i < total_pages:
            print(f"⏳ Waiting {DELAY_BETWEEN_PAGES}s...")
            time.sleep(DELAY_BETWEEN_PAGES)

    database = {
        "metadata": {"source_file": pdf_path, "total_pages": total_pages,
                     "successful_extractions": successful, "failed_extractions": failed,
                     "processed_at": datetime.now().isoformat()},
        "events": all_events
    }
    database_file = output_dir / "events_database.json"
    with open(database_file, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Database saved → {database_file}")
    print(f"📊 {successful}/{total_pages} pages processed successfully\n")
    print("💡 Query your data (no API needed):")
    print("   python batch_processor.py query all events")
    print("   python batch_processor.py query IoT workshop")


# ═══════════════════════════════════════════════════════════════════════════
#  LOCAL QUERY ENGINE  (zero API calls)
# ═══════════════════════════════════════════════════════════════════════════

def _score_event(event: dict, keywords: list) -> int:
    """Score how well an event matches the keywords."""
    score = 0
    name  = (event.get("event_name")  or "").lower()
    etype = (event.get("event_type")  or "").lower()
    dept  = (event.get("department")  or "").lower()
    desc  = (event.get("description") or "").lower()
    full  = json.dumps(event).lower()
    for kw in keywords:
        k = kw.lower()
        if k in name:   score += 15
        if k in etype:  score += 8
        if k in dept:   score += 5
        if k in desc:   score += 3
        elif k in full: score += 1
    return score


def _print_full_event(event: dict, number: int):
    """Print ALL available fields of an event in a clean formatted layout."""
    SEP  = "─" * 70
    SEP2 = "═" * 70
    name = (event.get("event_name") or "Unnamed Event").upper()

    print(f"\n{SEP2}")
    print(f"  EVENT #{number}  —  {name}")
    print(SEP2)

    def row(label, value):
        if value:
            print(f"  {label:<30} {value}")

    # Basic info
    row("🎯 Type:",              event.get("event_type"))
    row("🏢 Department:",        event.get("department"))
    row("📍 Venue:",             event.get("venue"))
    row("🕒 Time:",              event.get("time"))
    row("💰 Registration Fee:",  event.get("registration_fee") or "Free / Not specified")
    row("🏆 Prizes:",            event.get("prizes"))
    row("👥 Target Audience:",   event.get("target_audience"))

    # Dates
    dates = event.get("dates") or {}
    if any(v for v in dates.values() if v):
        print(f"\n  📅 DATES")
        print(f"  {SEP[:60]}")
        row("   Event Date:",            dates.get("event_date"))
        row("   End Date:",              dates.get("end_date"))
        row("   Registration Deadline:", dates.get("registration_deadline"))
        row("   Abstract Submission:",   dates.get("abstract_submission"))

    # Description  (word-wrapped)
    desc = event.get("description")
    if desc:
        print(f"\n  📋 DESCRIPTION")
        print(f"  {SEP[:60]}")
        words, line = desc.split(), "  "
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(line)
                line = "  " + w
            else:
                line += (" " if line.strip() else "") + w
        if line.strip():
            print(line)

    # Topics
    topics = event.get("topics")
    if topics and isinstance(topics, list):
        print(f"\n  📚 TOPICS / THEMES")
        print(f"  {SEP[:60]}")
        for t in topics:
            if t:
                print(f"    • {t}")

    # Contacts  (ALL contacts, not just the first one)
    contacts = event.get("contact_persons")
    if contacts and isinstance(contacts, list):
        print(f"\n  📞 CONTACT PERSONS")
        print(f"  {SEP[:60]}")
        for c in contacts:
            if isinstance(c, dict):
                cname = c.get("name",  "")
                role  = c.get("role",  "")
                phone = c.get("phone", "")
                email = c.get("email", "")
                line  = f"    👤 {cname}"
                if role:  line += f"  ({role})"
                print(line)
                if phone: print(f"       ☎  {phone}")
                if email: print(f"       ✉  {email}")
            elif isinstance(c, str) and c:
                print(f"    👤 {c}")

    # Additional info
    extra = event.get("additional_info")
    if extra:
        print(f"\n  ℹ️  ADDITIONAL INFO")
        print(f"  {SEP[:60]}")
        print(f"  {extra}")

    print(f"\n{SEP2}\n")


def query_events_database_local(query: str,
                                 database_path: str = "processed_events/events_database.json"):
    """
    Query the events database locally — no API calls needed.
    Supports:
      • 'all events'                → list all events briefly
      • '<event name>'              → full details of matching event(s)
      • 'details of <event name>'   → full details
      • 'fee/date/venue/contact for <event>' → specific field only
    """
    database_file = Path(database_path)
    if not database_file.exists():
        print(f"❌ Database not found: {database_path}")
        print("Run batch processing first:  python batch_processor.py")
        return

    with open(database_file, 'r', encoding='utf-8') as f:
        database = json.load(f)

    events = database.get("events", [])
    if not events:
        print("❌ Database is empty. Re-run batch processing.")
        return

    query_lower = query.lower().strip()
    print(f"\n🔍 Searching {len(events)} events...")
    print(f"Query: \"{query}\"\n")

    # ── List all ──────────────────────────────────────────────────────
    if query_lower in ("all", "all events", "list", "list all", "show all"):
        print(f"📋 All {len(events)} events:\n")
        for i, ev in enumerate(events, 1):
            name  = ev.get("event_name") or "Unnamed Event"
            etype = ev.get("event_type") or "N/A"
            dates = ev.get("dates") or {}
            date  = dates.get("event_date") or "N/A"
            fee   = ev.get("registration_fee") or "N/A"
            print(f"  {i:>2}. {name}")
            print(f"      Type: {etype}  |  Date: {date}  |  Fee: {fee}\n")
        return

    # ── Detect what the user is asking for ────────────────────────────
    asking_fee     = any(w in query_lower for w in ['fee', 'cost', 'price', 'charge', 'payment'])
    asking_date    = any(w in query_lower for w in ['when', 'date', 'schedule', 'deadline', 'time'])
    asking_venue   = any(w in query_lower for w in ['where', 'venue', 'location', 'place'])
    asking_contact = any(w in query_lower for w in ['contact', 'phone', 'email', 'coordinator', 'who'])
    asking_topics  = any(w in query_lower for w in ['topic', 'topics', 'theme', 'themes', 'cover'])
    asking_prizes  = any(w in query_lower for w in ['prize', 'prizes', 'reward', 'award'])
    asking_details = any(w in query_lower for w in ['detail', 'details', 'full', 'complete',
                                                     'everything', 'about', 'info', 'information',
                                                     'describe', 'tell', 'show'])
    single_field   = asking_fee or asking_date or asking_venue or asking_contact or asking_topics or asking_prizes

    # ── Extract search keywords ───────────────────────────────────────
    noise = {
        'what', 'when', 'where', 'how', 'the', 'for', 'and', 'with', 'this',
        'that', 'about', 'give', 'show', 'tell', 'details', 'information',
        'event', 'of', 'me', 'is', 'are', 'was', 'were', 'a', 'an', 'all',
        'its', 'in', 'on', 'at', 'to', 'from', 'list', 'get', 'find', 'full',
        'complete', 'everything', 'describe', 'info'
    }
    keywords = [w for w in query_lower.split() if len(w) > 2 and w not in noise]

    # ── Score events ──────────────────────────────────────────────────
    scored = []
    for i, event in enumerate(events, 1):
        score = _score_event(event, keywords) if keywords else 1
        if score > 0:
            scored.append((score, i, event))
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        print("❌ No matching events found.")
        print("\n💡 Try:")
        print("   python batch_processor.py query all events")
        print("   python batch_processor.py query IoT workshop")
        return

    # Keep results within 50% of the top score (to avoid irrelevant noise)
    top_score = scored[0][0]
    if top_score >= 10:
        matches = [(sc, n, ev) for sc, n, ev in scored if sc >= top_score * 0.5]
    else:
        matches = scored

    # ── Single-field answer (fee / date / venue / contact etc.) ──────
    if single_field and not asking_details:
        print(f"✅ Found {len(matches)} matching event(s):\n")
        for sc, num, event in matches:
            ename = event.get("event_name") or "Unnamed Event"
            print(f"{num}. {ename}")

            if asking_fee:
                print(f"   💰 Fee:     {event.get('registration_fee') or 'Not specified'}")

            if asking_date:
                dates = event.get("dates") or {}
                if dates.get("event_date"):
                    print(f"   📅 Date:    {dates['event_date']}")
                if dates.get("end_date"):
                    print(f"   📅 Ends:    {dates['end_date']}")
                if dates.get("registration_deadline"):
                    print(f"   ⏰ Reg. by: {dates['registration_deadline']}")
                if not any([dates.get("event_date"), dates.get("end_date"), dates.get("registration_deadline")]):
                    print(f"   📅 Date: Not specified")

            if asking_venue:
                print(f"   📍 Venue:   {event.get('venue') or 'Not specified'}")

            if asking_contact:
                contacts = event.get("contact_persons") or []
                if contacts:
                    print(f"   📞 Contacts:")
                    for c in contacts:
                        if isinstance(c, dict):
                            cname = c.get("name","")
                            role  = c.get("role","")
                            phone = c.get("phone","")
                            email = c.get("email","")
                            line  = f"      👤 {cname}"
                            if role:  line += f" ({role})"
                            if phone: line += f"  ☎ {phone}"
                            if email: line += f"  ✉ {email}"
                            print(line)
                        elif isinstance(c, str):
                            print(f"      👤 {c}")
                else:
                    print(f"   📞 Contact: Not specified")

            if asking_topics:
                topics = event.get("topics") or []
                if topics:
                    print(f"   📚 Topics:  {', '.join(t for t in topics if t)}")
                else:
                    print(f"   📚 Topics:  Not specified")

            if asking_prizes:
                print(f"   🏆 Prizes:  {event.get('prizes') or 'Not specified'}")

            print()

        if len(matches) == 1:
            best_name = matches[0][2].get("event_name") or "event"
            print(f"💡 Full details: python batch_processor.py query details of {best_name}")
        return

    # ── Full detail display ───────────────────────────────────────────
    if len(matches) > 4 and not asking_details:
        # Too many — list them briefly
        print(f"⚠️  {len(matches)} events matched. Showing brief list:\n")
        for sc, num, ev in matches[:10]:
            print(f"  {num}. {ev.get('event_name') or 'Unnamed'}")
        if len(matches) > 10:
            print(f"  ... and {len(matches)-10} more")
        print(f"\n💡 Be more specific:")
        print(f"   python batch_processor.py query details of <event name>")
        print(f"   python batch_processor.py query all events")
    else:
        print(f"✅ Found {len(matches)} matching event(s):\n")
        for sc, num, ev in matches:
            _print_full_event(ev, num)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         EVENT ANNOUNCEMENT PROCESSING SYSTEM                        ║")
    print("║         Powered by Gemini 2.0 Flash  (v2 — Full Details)            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] == "query":
            if len(sys.argv) > 2:
                query_events_database_local(" ".join(sys.argv[2:]))
            else:
                print("Usage: python batch_processor.py query <question>")
                print()
                print("Examples:")
                print("  python batch_processor.py query all events")
                print("  python batch_processor.py query IoT workshop")
                print("  python batch_processor.py query details of TECHNIMBLE")
                print("  python batch_processor.py query What is the fee for IoT workshop?")
                print("  python batch_processor.py query When is the Cyber Security workshop?")
                print("  python batch_processor.py query contacts for industrial visit")
                print("  python batch_processor.py query topics covered in IoT workshop")
        else:
            print("Unknown command. Use 'query' to search, or no argument to process the PDF.")
    else:
        batch_process_event_photos("events_photos.pdf")