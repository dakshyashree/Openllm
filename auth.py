# auth.py
import yaml
import hashlib
import os

USERS_FILE = 'users.yaml'


def load_users():
    """
    Loads user data from the users.yaml file.
    If the file does not exist or is empty, it returns an empty dictionary.
    """
    if not os.path.exists(USERS_FILE) or os.stat(USERS_FILE).st_size == 0:
        return {}
    with open(USERS_FILE, 'r') as f:
        users = yaml.safe_load(f)
        return users if users is not None else {}


def save_users(users):
    """
    Saves the provided user data to the users.yaml file.
    """
    with open(USERS_FILE, 'w') as f:
        yaml.safe_dump(users, f, default_flow_style=False)


def hash_password(password):
    """
    Hashes a password using SHA-256 for storage.
    Note: For production, consider stronger hashing algorithms like bcrypt or Argon2
    and a proper salt. This is simplified for demonstration.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    """
    Registers a new user with a specific role.
    The very first user registered will be assigned the 'admin' role.
    All subsequent users will be assigned the 'qa' role.

    Args:
        username (str): The username for the new user.
        password (str): The plain-text password for the new user.
    Returns:
        tuple: (bool, str) - True and a success message if successful,
               False and an error message otherwise.
    """
    if not username or not password:
        return False, "Username and password cannot be empty."

    users = load_users()

    if username in users:
        return False, "Registration failed: Username already exists."

    # Determine role: first user is admin, subsequent are qa
    if not users:  # If no users exist, this is the first registration
        role = 'admin'
    else:
        role = 'qa'

    hashed_password = hash_password(password)
    users[username] = {'password': hashed_password, 'role': role}
    save_users(users)
    return True, f"User '{username}' registered successfully as {role}."


def authenticate_user(username, password):
    """
    Authenticates a user and returns their role.
    Args:
        username (str): The username to authenticate.
        password (str): The plain-text password to authenticate.
    Returns:
        tuple: (bool, str, str | None) - True, success message, and role if successful;
               False, error message, and None otherwise.
    """
    if not username or not password:
        return False, "Username and password cannot be empty.", None

    users = load_users()
    if username not in users:
        return False, "Login failed: Username not found.", None

    hashed_password_input = hash_password(password)
    if users[username]['password'] == hashed_password_input:
        user_role = users[username].get('role', 'qa')  # Default to 'qa' if role not present
        return True, f"User '{username}' logged in successfully as {user_role}.", user_role
    else:
        return False, "Login failed: Incorrect password.", None

