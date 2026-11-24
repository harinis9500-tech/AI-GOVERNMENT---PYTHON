# AI-Based Government Complaint Tracking and Unresolved Grievance Analysis System

A Flask-based web application for citizens to submit and track public-service complaints, and for officials to review, assign, investigate, and resolve them. It includes AI-assisted category/urgency suggestions and same-area duplicate-complaint matching. AI suggestions are advisory only; a human administrator makes every duplicate decision.

## Features
- Citizen registration, login, complaint submission, tracking and feedback
- Complaint photo upload and optional GPS coordinates
- Unique tracking IDs and status history
- TF-IDF/cosine-similarity suggestions restricted to complaints in the same area
- Admin review to confirm or reject duplicate suggestions (never automatic rejection)
- Official assignment, investigation notes, status updates and resolution-photo submission
- Admin dashboard, overdue/unresolved summaries, area/category summaries and CSV/PDF exports
- Password hashing, role checks, upload validation and audit events

## Tech stack
Python 3.12, Flask, SQLite, Bootstrap 5, scikit-learn, Pillow, ReportLab.

## Windows setup (VS Code terminal)
```powershell
cd path\to\AI-GOVERNMENT---PYTHON
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

## First admin account
Register normally, then promote the account in SQLite:
```sql
UPDATE users SET role='admin' WHERE email='your-email@example.com';
```
Open the database with DB Browser for SQLite. Restart the app and sign in again.

## Database safety
The application uses `government_complaints.sqlite3` in this project folder. It does not overwrite an existing `complaints.db` file. Back up your database before making schema changes.

## Configuration
Set a strong `SECRET_KEY` environment variable before deployment. For local development a fallback key is used. Do not deploy the development server publicly. Use HTTPS, a production WSGI server, secure cookie settings, and an appropriately protected database in production.

## AI safety
Similarity scores are decision-support signals, not proof that complaints are duplicates. Only same-area records are compared. Administrators must review and confirm/reject each suggestion; complaints are never automatically rejected or closed.

## Project structure
- `app.py` – routes, database setup, authentication, complaint workflow and exports
- `templates/` – Bootstrap pages
- `static/style.css` – application styles
- `uploads/` – uploaded images (created automatically; excluded from Git)
