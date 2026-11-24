import csv
import io
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, abort, flash, g, make_response, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "government_complaints.sqlite3"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-this-secret"),
    MAX_CONTENT_LENGTH=MAX_IMAGE_BYTES + 256 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

STATUSES = ["Submitted", "Under Review", "Assigned", "In Progress", "Resolution Submitted", "Resolved", "Rejected"]
CATEGORIES = ["Roads", "Water Supply", "Sanitation", "Electricity", "Streetlights", "Drainage", "Public Safety", "Other"]


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'citizen',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id TEXT NOT NULL UNIQUE,
            user_id INTEGER NOT NULL REFERENCES users(id),
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            area TEXT NOT NULL,
            address TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            image_name TEXT,
            status TEXT NOT NULL DEFAULT 'Submitted',
            urgency TEXT NOT NULL DEFAULT 'Normal',
            severity TEXT NOT NULL DEFAULT 'Medium',
            assigned_to INTEGER REFERENCES users(id),
            due_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolution_note TEXT,
            resolution_image TEXT,
            feedback TEXT
        );
        CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER NOT NULL REFERENCES complaints(id),
            actor_id INTEGER REFERENCES users(id),
            status TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS duplicate_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER NOT NULL REFERENCES complaints(id),
            matched_complaint_id INTEGER NOT NULL REFERENCES complaints(id),
            similarity REAL NOT NULL,
            decision TEXT NOT NULL DEFAULT 'Pending',
            reviewed_by INTEGER REFERENCES users(id),
            created_at TEXT NOT NULL,
            UNIQUE(complaint_id, matched_complaint_id)
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_id INTEGER REFERENCES users(id),
            action TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL
        );
        """)


def now():
    return datetime.now().isoformat(timespec="seconds")


def audit(action, details=""):
    db().execute("INSERT INTO audit_logs(actor_id, action, details, created_at) VALUES(?,?,?,?)",
                 (session.get("user_id"), action, details[:500], now()))
    db().commit()


@app.before_request
def load_user():
    g.user = None
    if session.get("user_id"):
        g.user = db().execute("SELECT id,name,email,role FROM users WHERE id=?", (session["user_id"],)).fetchone()


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not g.user:
            flash("Please sign in first.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped


def role_required(*roles):
    def decorator(fn):
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            if g.user["role"] not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return decorator


def save_image(file):
    if not file or not file.filename:
        return None
    original = secure_filename(file.filename)
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Use a PNG, JPG, JPEG or WEBP image.")
    data = file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be smaller than 5 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a valid image.")
    name = f"{uuid.uuid4().hex}.{ext}"
    (UPLOAD_DIR / name).write_bytes(data)
    return name


def suggest_category(text):
    rules = {
        "Roads": ["road", "pothole", "street", "footpath"],
        "Water Supply": ["water", "pipe", "leak", "tap"],
        "Sanitation": ["garbage", "waste", "rubbish", "clean"],
        "Electricity": ["power", "electric", "transformer", "wire"],
        "Streetlights": ["streetlight", "street light", "lamp", "dark"],
        "Drainage": ["drain", "sewage", "flood", "blocked"],
        "Public Safety": ["unsafe", "accident", "danger", "exposed"],
    }
    lower = text.lower()
    return next((category for category, words in rules.items() if any(word in lower for word in words)), "Other")


def find_similar(complaint_id):
    current = db().execute("SELECT * FROM complaints WHERE id=?", (complaint_id,)).fetchone()
    peers = db().execute(
        "SELECT * FROM complaints WHERE area=? AND id<>? AND status NOT IN ('Rejected')",
        (current["area"], complaint_id)
    ).fetchall()
    if not peers:
        return []
    documents = [current["title"] + " " + current["description"]] + [
        p["title"] + " " + p["description"] for p in peers
    ]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(documents)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        return []
    matches = []
    for peer, score in zip(peers, scores):
        if score >= 0.25:
            percent = round(float(score) * 100, 1)
            db().execute(
                "INSERT OR IGNORE INTO duplicate_reviews(complaint_id,matched_complaint_id,similarity,created_at) VALUES(?,?,?,?)",
                (complaint_id, peer["id"], percent, now())
            )
            matches.append({"tracking_id": peer["tracking_id"], "title": peer["title"], "score": percent})
    db().commit()
    return matches


@app.route("/")
def index():
    stats = db().execute("SELECT COUNT(*) total, SUM(status!='Resolved' AND status!='Rejected') open_count FROM complaints").fetchone()
    return render_template("index.html", stats=stats)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) or len(password) < 8:
            flash("Enter a name, valid email and password of at least 8 characters.", "danger")
        else:
            try:
                db().execute("INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                             (name, email, generate_password_hash(password), now()))
                db().commit()
                flash("Account created. Please sign in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That email is already registered.", "danger")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = db().execute("SELECT * FROM users WHERE email=?", (request.form.get("email", "").strip().lower(),)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session["user_id"] = user["id"]
            flash("Welcome back.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have signed out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    if g.user["role"] == "admin":
        rows = db().execute("SELECT c.*,u.name citizen FROM complaints c JOIN users u ON u.id=c.user_id ORDER BY c.created_at DESC").fetchall()
    elif g.user["role"] == "official":
        rows = db().execute("SELECT c.*,u.name citizen FROM complaints c JOIN users u ON u.id=c.user_id WHERE c.assigned_to=? ORDER BY c.created_at DESC", (g.user["id"],)).fetchall()
    else:
        rows = db().execute("SELECT * FROM complaints WHERE user_id=? ORDER BY created_at DESC", (g.user["id"],)).fetchall()
    return render_template("dashboard.html", complaints=rows)


@app.route("/complaint/new", methods=["GET", "POST"])
@role_required("citizen", "admin")
def new_complaint():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        area = request.form.get("area", "").strip()
        address = request.form.get("address", "").strip()
        if not all([title, description, area, address]):
            flash("Please complete all required fields.", "danger")
            return render_template("new_complaint.html", categories=CATEGORIES)
        try:
            image_name = save_image(request.files.get("image"))
            category = request.form.get("category") or suggest_category(title + " " + description)
            tracking = "GOV-" + uuid.uuid4().hex[:8].upper()
            cur = db().execute("""INSERT INTO complaints
                (tracking_id,user_id,title,description,category,area,address,latitude,longitude,image_name,urgency,severity,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (tracking,g.user["id"],title,description,category,area,address,
                 request.form.get("latitude") or None,request.form.get("longitude") or None,image_name,
                 "High" if any(x in (title+" "+description).lower() for x in ["urgent","danger","emergency"]) else "Normal",
                 "High" if any(x in (title+" "+description).lower() for x in ["danger","accident","exposed wire"]) else "Medium",now(),now()))
            cid = cur.lastrowid
            db().execute("INSERT INTO status_history(complaint_id,actor_id,status,note,created_at) VALUES(?,?,?,?,?)",
                         (cid,g.user["id"],"Submitted","Complaint registered",now()))
            db().commit()
            matches = find_similar(cid)
            flash(f"Complaint submitted. Tracking ID: {tracking}", "success")
            if matches:
                flash(f"AI found {len(matches)} possible same-area match(es). An administrator will review them.", "info")
            audit("complaint_submitted", tracking)
            return redirect(url_for("complaint_detail", complaint_id=cid))
        except ValueError as exc:
            flash(str(exc), "danger")
    return render_template("new_complaint.html", categories=CATEGORIES)


@app.route("/complaint/<int:complaint_id>")
@login_required
def complaint_detail(complaint_id):
    complaint = db().execute("SELECT c.*,u.name citizen FROM complaints c JOIN users u ON u.id=c.user_id WHERE c.id=?", (complaint_id,)).fetchone()
    if not complaint:
        abort(404)
    if g.user["role"] == "citizen" and complaint["user_id"] != g.user["id"]:
        abort(403)
    if g.user["role"] == "official" and complaint["assigned_to"] != g.user["id"]:
        abort(403)
    history = db().execute("SELECT h.*,u.name actor FROM status_history h LEFT JOIN users u ON u.id=h.actor_id WHERE complaint_id=? ORDER BY h.created_at DESC", (complaint_id,)).fetchall()
    matches = db().execute("""SELECT d.*,c.tracking_id,c.title FROM duplicate_reviews d
        JOIN complaints c ON c.id=d.matched_complaint_id WHERE d.complaint_id=? ORDER BY d.similarity DESC""", (complaint_id,)).fetchall() if g.user["role"] == "admin" else []
    return render_template("complaint_detail.html", complaint=complaint, history=history, matches=matches, statuses=STATUSES)


@app.route("/admin")
@role_required("admin")
def admin():
    counts = db().execute("""SELECT COUNT(*) total,
        SUM(status NOT IN ('Resolved','Rejected')) open_count,
        SUM(status='Resolved') resolved_count,
        SUM(status NOT IN ('Resolved','Rejected') AND created_at < ?) overdue
        FROM complaints""", ((datetime.now()-timedelta(days=14)).isoformat(timespec="seconds"),)).fetchone()
    areas = db().execute("SELECT area,COUNT(*) count FROM complaints GROUP BY area ORDER BY count DESC LIMIT 8").fetchall()
    categories = db().execute("SELECT category,COUNT(*) count FROM complaints GROUP BY category ORDER BY count DESC").fetchall()
    officials = db().execute("SELECT id,name FROM users WHERE role='official' ORDER BY name").fetchall()
    pending = db().execute("SELECT d.*,a.tracking_id source_id,b.tracking_id match_id FROM duplicate_reviews d JOIN complaints a ON a.id=d.complaint_id JOIN complaints b ON b.id=d.matched_complaint_id WHERE d.decision='Pending' ORDER BY d.similarity DESC").fetchall()
    return render_template("admin.html", counts=counts, areas=areas, categories=categories, officials=officials, pending=pending)


@app.route("/admin/role", methods=["POST"])
@role_required("admin")
def set_role():
    email = request.form.get("email", "").strip().lower()
    role = request.form.get("role")
    if role not in {"citizen", "official", "admin"}:
        abort(400)
    db().execute("UPDATE users SET role=? WHERE email=?", (role,email))
    db().commit()
    audit("user_role_changed", f"{email}: {role}")
    flash("User role updated if the email exists.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/complaint/<int:complaint_id>/update", methods=["POST"])
@role_required("admin", "official")
def update_complaint(complaint_id):
    complaint = db().execute("SELECT * FROM complaints WHERE id=?", (complaint_id,)).fetchone()
    if not complaint:
        abort(404)
    if g.user["role"] == "official" and complaint["assigned_to"] != g.user["id"]:
        abort(403)
    status = request.form.get("status")
    if status not in STATUSES:
        abort(400)
    note = request.form.get("note", "").strip()
    db().execute("UPDATE complaints SET status=?,updated_at=? WHERE id=?", (status,now(),complaint_id))
    db().execute("INSERT INTO status_history(complaint_id,actor_id,status,note,created_at) VALUES(?,?,?,?,?)",
                 (complaint_id,g.user["id"],status,note,now()))
    db().commit()
    audit("status_updated", f"{complaint['tracking_id']}: {status}")
    flash("Complaint status updated.", "success")
    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/admin/complaint/<int:complaint_id>/assign", methods=["POST"])
@role_required("admin")
def assign_complaint(complaint_id):
    official_id = request.form.get("official_id", type=int)
    official = db().execute("SELECT id FROM users WHERE id=? AND role='official'", (official_id,)).fetchone()
    if not official:
        flash("Choose a valid official account.", "danger")
    else:
        db().execute("UPDATE complaints SET assigned_to=?,status='Assigned',updated_at=? WHERE id=?", (official_id,now(),complaint_id))
        db().execute("INSERT INTO status_history(complaint_id,actor_id,status,note,created_at) VALUES(?,?,?,?,?)",
                     (complaint_id,g.user["id"],"Assigned","Assigned to official",now()))
        db().commit()
        audit("complaint_assigned", str(complaint_id))
        flash("Complaint assigned.", "success")
    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/admin/duplicate/<int:review_id>", methods=["POST"])
@role_required("admin")
def duplicate_decision(review_id):
    decision = request.form.get("decision")
    if decision not in {"Confirmed", "Rejected"}:
        abort(400)
    db().execute("UPDATE duplicate_reviews SET decision=?,reviewed_by=? WHERE id=? AND decision='Pending'",
                 (decision,g.user["id"],review_id))
    db().commit()
    audit("duplicate_review", f"{review_id}: {decision}")
    flash(f"Possible duplicate marked {decision.lower()}. No complaint was automatically rejected.", "success")
    return redirect(url_for("admin"))


@app.route("/complaint/<int:complaint_id>/feedback", methods=["POST"])
@role_required("citizen")
def feedback(complaint_id):
    value = request.form.get("feedback", "").strip()[:1000]
    db().execute("UPDATE complaints SET feedback=? WHERE id=? AND user_id=? AND status='Resolved'",
                 (value,complaint_id,g.user["id"]))
    db().commit()
    flash("Thank you for your feedback.", "success")
    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    # Only serve images to authenticated users; access to complaint records is separately role-checked.
    if not re.fullmatch(r"[a-f0-9]{32}\.(png|jpg|jpeg|webp)", filename):
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/admin/export.csv")
@role_required("admin")
def export_csv():
    rows = db().execute("SELECT tracking_id,title,category,area,status,urgency,severity,created_at,updated_at FROM complaints ORDER BY created_at DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Tracking ID","Title","Category","Area","Status","Urgency","Severity","Created","Updated"])
    writer.writerows([tuple(row) for row in rows])
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=complaints-report.csv"
    return response


@app.route("/admin/export.pdf")
@role_required("admin")
def export_pdf():
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
    rows = db().execute("SELECT tracking_id,title,category,area,status,created_at FROM complaints ORDER BY created_at DESC").fetchall()
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=landscape(A4))
    width, height = landscape(A4)
    y = height - 45
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(35, y, "Government Complaint Report")
    y -= 30
    pdf.setFont("Helvetica-Bold", 8)
    headers = ["Tracking ID","Title","Category","Area","Status","Created"]
    xs = [35,140,350,440,535,635]
    for x, label in zip(xs, headers):
        pdf.drawString(x, y, label)
    y -= 16
    pdf.setFont("Helvetica", 8)
    for row in rows:
        if y < 35:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 8)
        values = [str(v or "")[:28] for v in row]
        for x, value in zip(xs, values):
            pdf.drawString(x, y, value)
        y -= 14
    pdf.save()
    buffer.seek(0)
    response = make_response(buffer.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=complaints-report.pdf"
    return response


@app.errorhandler(403)
def forbidden(_error):
    return render_template("error.html", message="You do not have permission to view this page."), 403


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", message="The requested page or complaint was not found."), 404


with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=True)
