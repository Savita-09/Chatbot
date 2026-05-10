"""
Nexus AI — Flask + MySQL Backend
─────────────────────────────────
Endpoints:
  POST /auth/register        — Register new user
  POST /auth/login           — Login & get JWT token
  POST /auth/logout          — Invalidate token
  GET  /user/profile         — Get user profile
  PUT  /user/profile         — Update user profile
  GET  /user/usage           — Get daily usage stats
  PUT  /user/usage           — Bump usage counter
  GET  /user/plan            — Get current plan
  POST /billing/subscribe    — Subscribe to a plan
  POST /billing/cancel       — Cancel subscription
  GET  /sessions             — List all sessions
  POST /sessions             — Create new session
  GET  /sessions/<id>        — Get session with messages
  DELETE /sessions/<id>      — Delete session
  POST /sessions/<id>/messages — Add message to session
  GET  /health               — Health check
"""

import os, jwt, bcrypt
from datetime import datetime, date, timedelta
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
app.config["MYSQL_HOST"]     = os.getenv("MYSQL_HOST", "localhost")
app.config["MYSQL_USER"]     = os.getenv("MYSQL_USER", "root")
app.config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD", "helloworld@123")
app.config["MYSQL_DB"]       = os.getenv("MYSQL_DB", "chatbot")
app.config["SECRET_KEY"]     = os.getenv("SECRET_KEY", "nx7-a2$qL#9wZ@!kP2rF#gT8sVxY*eJ3")  # random secret
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)

# ─────────────────────────────────────────────────────────────
# PLAN LIMITS
# ─────────────────────────────────────────────────────────────
PLAN_LIMITS = {
    "free":       {"chat": 20,    "search": 5,     "image": 3,     "price": 0},
    "pro":        {"chat": 9999,  "search": 9999,  "image": 30,    "price": 9},
    "enterprise": {"chat": 99999, "search": 99999, "image": 99999, "price": 49},
}

# ─────────────────────────────────────────────────────────────
# JWT HELPERS
# ─────────────────────────────────────────────────────────────
def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            request.user_id = data["user_id"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────
# DB HELPER
# ─────────────────────────────────────────────────────────────
def get_cursor():
    return mysql.connection.cursor()

def commit():
    mysql.connection.commit()

# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "nexus-ai-backend"})

# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "username, email, and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    cur = get_cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cur.fetchone():
        return jsonify({"error": "Email already registered"}), 409

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    cur.execute(
        "INSERT INTO users (username, email, password_hash, plan, avatar_color, created_at) "
        "VALUES (%s, %s, %s, 'free', '#4f8cff', NOW())",
        (username, email, pw_hash),
    )
    commit()
    user_id = cur.lastrowid

    # Seed daily usage row for today
    cur.execute(
        "INSERT INTO daily_usage (user_id, usage_date, chat_count, search_count, image_count) "
        "VALUES (%s, %s, 0, 0, 0)",
        (user_id, date.today()),
    )
    commit()

    token = create_token(user_id)
    return jsonify({"token": token, "user_id": user_id, "username": username, "plan": "free"}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data  = request.get_json(force=True) or {}  
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    cur = get_cursor()
    cur.execute("SELECT id, username, password_hash, plan, avatar_color FROM users WHERE email = %s", (email,))
    user = cur.fetchone()

    if not user or not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user["id"])
    return jsonify({
        "token":    token,
        "user_id":  user["id"],
        "username": user["username"],
        "plan":     user["plan"],
        "avatar_color": user["avatar_color"],
    })


@app.route("/auth/logout", methods=["POST"])
@token_required
def logout():
    # Stateless JWT — client just discards token
    # For server-side invalidation, add token to a blocklist table
    return jsonify({"message": "Logged out successfully"})

# ─────────────────────────────────────────────────────────────
# USER PROFILE
# ─────────────────────────────────────────────────────────────
@app.route("/user/profile", methods=["GET"])
@token_required
def get_profile():
    cur = get_cursor()
    cur.execute(
        "SELECT id, username, email, plan, avatar_color, billing_email, billing_card_last4, "
        "sub_start_date, created_at FROM users WHERE id = %s",
        (request.user_id,),
    )
    user = cur.fetchone()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)


@app.route("/user/profile", methods=["PUT"])
@token_required
def update_profile():
    data  = request.get_json()
    fields, values = [], []

    for col in ("username", "email", "avatar_color"):
        if col in data and data[col]:
            fields.append(f"{col} = %s")
            values.append(data[col].strip())

    if not fields:
        return jsonify({"error": "Nothing to update"}), 400

    values.append(request.user_id)
    cur = get_cursor()
    cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = %s", values)
    commit()
    return jsonify({"message": "Profile updated"})

# ─────────────────────────────────────────────────────────────
# DAILY USAGE
# ─────────────────────────────────────────────────────────────
def _ensure_usage_row(cur, user_id, today):
    cur.execute(
        "INSERT IGNORE INTO daily_usage (user_id, usage_date, chat_count, search_count, image_count) "
        "VALUES (%s, %s, 0, 0, 0)",
        (user_id, today),
    )
    commit()


@app.route("/user/usage", methods=["GET"])
@token_required
def get_usage():
    today = date.today()
    cur   = get_cursor()
    _ensure_usage_row(cur, request.user_id, today)

    cur.execute(
        "SELECT chat_count, search_count, image_count FROM daily_usage "
        "WHERE user_id = %s AND usage_date = %s",
        (request.user_id, today),
    )
    usage = cur.fetchone() or {"chat_count": 0, "search_count": 0, "image_count": 0}

    cur.execute("SELECT plan FROM users WHERE id = %s", (request.user_id,))
    row  = cur.fetchone()
    plan = row["plan"] if row else "free"
    limits = PLAN_LIMITS[plan]

    return jsonify({
        "date":         str(today),
        "plan":         plan,
        "usage":        usage,
        "limits":       limits,
        "remaining": {
            "chat":   max(0, limits["chat"]   - usage["chat_count"]),
            "search": max(0, limits["search"] - usage["search_count"]),
            "image":  max(0, limits["image"]  - usage["image_count"]),
        }
    })


@app.route("/user/usage", methods=["PUT"])
@token_required
def bump_usage():
    """
    Body: { "mode": "chat" | "search" | "image" }
    Increments the counter, returns updated usage & whether limit is hit.
    """
    data = request.get_json()
    mode = data.get("mode", "")
    if mode not in ("chat", "search", "image"):
        return jsonify({"error": "mode must be chat, search, or image"}), 400

    today = date.today()
    col   = f"{mode}_count"
    cur   = get_cursor()
    _ensure_usage_row(cur, request.user_id, today)

    # Check limit first
    cur.execute("SELECT plan FROM users WHERE id = %s", (request.user_id,))
    plan = cur.fetchone()["plan"]
    limit = PLAN_LIMITS[plan][mode]

    cur.execute(
        f"SELECT {col} FROM daily_usage WHERE user_id = %s AND usage_date = %s",
        (request.user_id, today),
    )
    current = cur.fetchone()[col]

    if current >= limit:
        return jsonify({"error": "limit_reached", "used": current, "limit": limit}), 429

    cur.execute(
        f"UPDATE daily_usage SET {col} = {col} + 1 WHERE user_id = %s AND usage_date = %s",
        (request.user_id, today),
    )
    commit()
    return jsonify({"used": current + 1, "limit": limit, "remaining": limit - current - 1})

# ─────────────────────────────────────────────────────────────
# PLAN
# ─────────────────────────────────────────────────────────────
@app.route("/user/plan", methods=["GET"])
@token_required
def get_plan():
    cur = get_cursor()
    cur.execute("SELECT plan, sub_start_date FROM users WHERE id = %s", (request.user_id,))
    row = cur.fetchone()
    plan = row["plan"] if row else "free"
    return jsonify({
        "plan":      plan,
        "limits":    PLAN_LIMITS[plan],
        "sub_start": str(row["sub_start_date"]) if row and row["sub_start_date"] else None,
    })

# ─────────────────────────────────────────────────────────────
# BILLING / SUBSCRIPTION
# ─────────────────────────────────────────────────────────────
@app.route("/billing/subscribe", methods=["POST"])
@token_required
def subscribe():
    """
    Body: { "plan": "pro"|"enterprise", "email": "...", "card": "1234...", "expiry": "MM/YY", "cvc": "...", "name": "..." }
    In production, replace card handling with Stripe PaymentIntent.
    """
    data = request.get_json()
    plan = data.get("plan", "")
    if plan not in ("pro", "enterprise"):
        return jsonify({"error": "plan must be pro or enterprise"}), 400

    required = ("email", "card", "expiry", "cvc", "name")
    missing  = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    card_last4 = data["card"].replace(" ", "")[-4:]
    cur = get_cursor()
    cur.execute(
        "UPDATE users SET plan = %s, billing_email = %s, billing_card_last4 = %s, "
        "sub_start_date = %s WHERE id = %s",
        (plan, data["email"], card_last4, date.today(), request.user_id),
    )
    commit()

    # Log billing event
    cur.execute(
        "INSERT INTO billing_events (user_id, event_type, plan, amount, created_at) "
        "VALUES (%s, 'subscribe', %s, %s, NOW())",
        (request.user_id, plan, PLAN_LIMITS[plan]["price"]),
    )
    commit()

    return jsonify({
        "message":  f"Subscribed to {plan}",
        "plan":     plan,
        "card_last4": card_last4,
        "sub_start": str(date.today()),
    })


@app.route("/billing/cancel", methods=["POST"])
@token_required
def cancel_subscription():
    cur = get_cursor()
    cur.execute("SELECT plan FROM users WHERE id = %s", (request.user_id,))
    row = cur.fetchone()
    if not row or row["plan"] == "free":
        return jsonify({"error": "No active subscription"}), 400

    old_plan = row["plan"]
    cur.execute(
        "UPDATE users SET plan = 'free', sub_start_date = NULL WHERE id = %s",
        (request.user_id,),
    )
    commit()

    cur.execute(
        "INSERT INTO billing_events (user_id, event_type, plan, amount, created_at) "
        "VALUES (%s, 'cancel', %s, 0, NOW())",
        (request.user_id, old_plan),
    )
    commit()
    return jsonify({"message": "Subscription cancelled. Now on Free plan."})

# ─────────────────────────────────────────────────────────────
# SESSIONS (CONVERSATIONS)
# ─────────────────────────────────────────────────────────────
@app.route("/sessions", methods=["GET"])
@token_required
def list_sessions():
    cur = get_cursor()
    cur.execute(
        "SELECT id, title, created_at, updated_at "
        "FROM sessions WHERE user_id = %s ORDER BY updated_at DESC",
        (request.user_id,),
    )
    rows = cur.fetchall()
    # Stringify datetimes
    for r in rows:
        r["created_at"] = str(r["created_at"])
        r["updated_at"] = str(r["updated_at"])
    return jsonify(rows)


@app.route("/sessions", methods=["POST"])
@token_required
def create_session():
    data  = request.get_json()
    title = data.get("title", "New Chat")
    cur   = get_cursor()
    cur.execute(
        "INSERT INTO sessions (user_id, title, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())",
        (request.user_id, title),
    )
    commit()
    sid = cur.lastrowid
    return jsonify({"id": sid, "title": title}), 201


@app.route("/sessions/<int:session_id>", methods=["GET"])
@token_required
def get_session(session_id):
    cur = get_cursor()
    cur.execute(
        "SELECT id, title, created_at FROM sessions WHERE id = %s AND user_id = %s",
        (session_id, request.user_id),
    )
    sess = cur.fetchone()
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    cur.execute(
        "SELECT id, role, content, mode, sources, created_at FROM messages "
        "WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,),
    )
    messages = cur.fetchall()
    import json
    for m in messages:
        m["created_at"] = str(m["created_at"])
        try:
            m["sources"] = json.loads(m["sources"]) if m["sources"] else []
        except Exception:
            m["sources"] = []

    sess["created_at"] = str(sess["created_at"])
    sess["messages"]   = messages
    return jsonify(sess)


@app.route("/sessions/<int:session_id>", methods=["DELETE"])
@token_required
def delete_session(session_id):
    cur = get_cursor()
    cur.execute(
        "SELECT id FROM sessions WHERE id = %s AND user_id = %s",
        (session_id, request.user_id),
    )
    if not cur.fetchone():
        return jsonify({"error": "Session not found"}), 404

    cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
    cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    commit()
    return jsonify({"message": "Session deleted"})


@app.route("/sessions/<int:session_id>/rename", methods=["PUT"])
@token_required
def rename_session(session_id):
    data  = request.get_json()
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    cur = get_cursor()
    cur.execute(
        "UPDATE sessions SET title = %s WHERE id = %s AND user_id = %s",
        (title, session_id, request.user_id),
    )
    commit()
    return jsonify({"message": "Renamed", "title": title})


@app.route("/sessions/<int:session_id>/messages", methods=["POST"])
@token_required
def add_message(session_id):
    """
    Body: { "role": "user"|"assistant", "content": "...", "mode": "chat|search|image", "sources": [...] }
    Also updates the session's updated_at timestamp.
    """
    import json as _json
    data = request.get_json()
    role    = data.get("role", "user")
    content = data.get("content", "")
    mode    = data.get("mode", "chat")
    sources = _json.dumps(data.get("sources", []))

    if role not in ("user", "assistant"):
        return jsonify({"error": "role must be user or assistant"}), 400

    cur = get_cursor()
    cur.execute(
        "SELECT id FROM sessions WHERE id = %s AND user_id = %s",
        (session_id, request.user_id),
    )
    if not cur.fetchone():
        return jsonify({"error": "Session not found"}), 404

    cur.execute(
        "INSERT INTO messages (session_id, role, content, mode, sources, created_at) "
        "VALUES (%s, %s, %s, %s, %s, NOW())",
        (session_id, role, content, mode, sources),
    )
    cur.execute(
        "UPDATE sessions SET updated_at = NOW() WHERE id = %s", (session_id,)
    )
    commit()
    msg_id = cur.lastrowid
    return jsonify({"id": msg_id, "role": role, "mode": mode}), 201


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)