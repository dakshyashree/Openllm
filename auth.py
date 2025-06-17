import yaml
from pathlib import Path
import bcrypt
from datetime import datetime

# Define the path to your users file
USERS_FILE = Path("users.yaml")

def _load_users():
    """Loads user data from the YAML file."""
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r') as f:
            return yaml.safe_load(f) or {"users": {}} # Ensure it returns a dict even if file is empty
    return {"users": {}}

def _save_users(users_data):
    """Saves user data to the YAML file."""
    with open(USERS_FILE, 'w') as f:
        yaml.safe_dump(users_data, f)

def authenticate_user(username, password):
    """
    Authenticates a user by checking their username, password, and active status.
    Updates last login timestamp on success.
    """
    users_data = _load_users()
    users = users_data.get("users", {})
    user_info = users.get(username)

    if not user_info:
        return False, "Invalid username or password.", None

    # Check if the account is active
    if not user_info.get("active", True): # Default to True for older entries without 'active' field
        return False, "Account is currently inactive. Please contact an administrator.", None

    hashed_password = user_info["password_hash"].encode('utf-8')
    salt = user_info["salt"].encode('utf-8')

    if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        # Update last_login timestamp on successful login
        user_info["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_users(users_data)
        return True, "Login successful.", user_info["role"]
    else:
        return False, "Invalid username or password.", None

def register_user(username, password):
    """
    Registers a new user. The first registered user becomes an 'admin',
    subsequent users are 'qa_user'. Sets new users as active by default.
    """
    users_data = _load_users()
    users = users_data.get("users", {})

    if username in users:
        return False, "Username already exists."

    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    # Assign role: first user is admin, others are qa_user
    role = "admin" if not users else "qa_user"

    users[username] = {
        "password_hash": hashed_password,
        "salt": salt.decode('utf-8'),
        "role": role,
        "active": True, # New users are active by default
        "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    users_data["users"] = users
    _save_users(users_data)
    return True, f"User '{username}' registered successfully as {role}."

def get_all_users_status():
    """Retrieves a dictionary of all users and their details."""
    users_data = _load_users()
    return users_data.get("users", {})

def update_user_status(username, new_status: bool): # Removed current_admin_username from signature as check is now in app.py
    """
    Updates the active status of a specific user.
    """
    users_data = _load_users()
    users = users_data.get("users", {})

    if username not in users:
        return False, f"User '{username}' not found."

    # Safeguard for sole active admin is primarily in app.py now, this is a redundant backend check.
    # if users[username]['role'] == 'admin' and not new_status: # If target is admin and being deactivated
    #     active_admins = [u for u, info in users.items() if info.get('role') == 'admin' and info.get('active', True)]
    #     if len(active_admins) == 1 and active_admins[0] == username:
    #         return False, "Cannot deactivate the sole active administrator account."

    users[username]["active"] = new_status
    _save_users(users_data)
    status_text = "activated" if new_status else "deactivated"
    return True, f"User '{username}' has been {status_text}."

def change_password(username, old_password, new_password):
    """
    Allows a user to change their password after verifying the old password.
    """
    users_data = _load_users()
    users = users_data.get("users", {})
    user_info = users.get(username)

    if not user_info:
        return False, "User not found."

    hashed_password = user_info["password_hash"].encode('utf-8')
    salt = user_info["salt"].encode('utf-8')

    # Verify old password
    if not bcrypt.checkpw(old_password.encode('utf-8'), hashed_password):
        return False, "Incorrect old password."

    # Hash the new password
    new_salt = bcrypt.gensalt()
    new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), new_salt).decode('utf-8')

    # Update user info with new password and salt
    user_info["password_hash"] = new_hashed_password
    user_info["salt"] = new_salt.decode('utf-8')
    users_data["users"] = users # Ensure updated user_info is stored back
    _save_users(users_data)
    return True, "Password changed successfully."

# --- NEW FUNCTION TO DELETE USER ---
def delete_user(username_to_delete: str, acting_admin_username: str):
    """
    Deletes a user from the users.yaml file.
    Includes safeguards: Admin cannot delete themselves or the sole admin.
    """
    users_data = _load_users()
    users = users_data.get("users", {})

    if username_to_delete not in users:
        return False, f"User '{username_to_delete}' not found."

    # Safeguard 1: Admin cannot delete themselves
    if username_to_delete == acting_admin_username:
        return False, "You cannot delete your own account."

    # Safeguard 2: Admin cannot delete the sole active admin
    if users[username_to_delete]['role'] == 'admin': # If the target is an admin
        active_admins = [u for u, info in users.items() if info.get('role') == 'admin' and info.get('active', True)]
        if len(active_admins) == 1 and active_admins[0] == username_to_delete:
            return False, "Cannot delete the sole active administrator account."

    del users[username_to_delete]
    users_data["users"] = users # Ensure updated users dict is saved back
    _save_users(users_data)
    return True, f"User '{username_to_delete}' deleted successfully."