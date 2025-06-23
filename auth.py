# auth.py
import yaml
import hashlib
import os
import datetime

USERS_FILE = "users.yaml"
MAX_ADMIN_COUNT = 5  # Define the maximum number of admin users allowed


def load_users():
    """
    Loads user data from the users.yaml file.
    If the file does not exist or is empty, it returns an empty dictionary.
    """
    if not os.path.exists(USERS_FILE) or os.stat(USERS_FILE).st_size == 0:
        return {}
    with open(USERS_FILE, "r") as f:
        users = yaml.safe_load(f)
        return users if users is not None else {}


def save_users(users):
    """
    Saves the provided user data to the users.yaml file.
    """
    with open(USERS_FILE, "w") as f:
        yaml.safe_dump(users, f, default_flow_style=False)


def hash_password(password):
    """
    Hashes a password using SHA-256 for storage.
    Note: For production, consider stronger hashing algorithms like bcrypt or Argon2
    and a proper salt. This is simplified for demonstration.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def get_admin_count():
    """
    Counts the number of existing admin users.
    Returns:
        int: The current number of admin users.
    """
    users = load_users()
    admin_count = sum(
        1 for user_data in users.values() if user_data.get("role") == "admin"
    )
    return admin_count


def register_user(username, password, desired_role):
    """
    Registers a new user with a specified role, enforcing the admin count limit.

    Args:
        username (str): The username for the new user.
        password (str): The plain-text password for the new user.
        desired_role (str): The role the user wishes to register as ('admin' or 'qa').
    Returns:
        tuple: (bool, str) - True and a success message if successful,
               False and an error message otherwise.
    """
    if not username or not password:
        return False, "Username and password cannot be empty."

    users = load_users()

    if username in users:
        return False, "Registration failed: Username already exists."

    if desired_role == "admin":
        current_admins = get_admin_count()
        if current_admins >= MAX_ADMIN_COUNT:
            # This message should ideally not be seen if UI prevents selection
            return (
                False,
                f"Registration failed: Maximum number of admin users ({MAX_ADMIN_COUNT}) reached. Cannot register as admin.",
            )

    hashed_password = hash_password(password)
    users[username] = {
        "password": hashed_password,
        "role": desired_role,
        "active": True,  # New users are active by default
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "Never",
    }
    save_users(users)
    return True, f"User '{username}' registered successfully as {desired_role}."


def authenticate_user(username, password):
    """
    Authenticates a user and returns their role. Updates last_login timestamp.
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

    user_data = users[username]
    if not user_data.get("active", True):  # Check if user is active
        return (
            False,
            "Login failed: Account is inactive. Please contact an administrator.",
            None,
        )

    hashed_password_input = hash_password(password)
    if user_data["password"] == hashed_password_input:
        user_role = user_data.get("role", "qa")  # Default to 'qa' if role not present

        # Update last_login timestamp
        users[username]["last_login"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        save_users(users)

        return (
            True,
            f"User '{username}' logged in successfully as {user_role}.",
            user_role,
        )
    else:
        return False, "Login failed: Incorrect password.", None


def change_password(username, current_password, new_password):
    """
    Changes a user's password.
    Returns: (bool, message)
    """
    users = load_users()
    if username not in users:
        return False, "User not found."

    if users[username]["password"] != hash_password(current_password):
        return False, "Incorrect current password."

    users[username]["password"] = hash_password(new_password)
    save_users(users)
    return True, "Password changed successfully."


def get_all_users_status():
    """
    Retrieves status of all users, including role, active status, and last login.
    Returns: dict of {username: {role: str, active: bool, last_login: str, ...}}
    """
    users = load_users()
    # Ensure 'active' status is present, default to True for display
    # Also ensure last_login and created_at are present
    for user_data in users.values():
        user_data.setdefault("active", True)
        user_data.setdefault("last_login", "N/A")
        user_data.setdefault("created_at", "N/A")
    return users


def update_user_status(username_to_update, new_status):
    """
    Updates the active status of a user.
    Returns: (bool, message)
    """
    users = load_users()
    if username_to_update not in users:
        return False, "User not found."

    users[username_to_update]["active"] = new_status
    save_users(users)
    return (
        True,
        f"User '{username_to_update}' status updated to {'active' if new_status else 'inactive'}.",
    )


def delete_user(username_to_delete, acting_admin_username):
    """
    Deletes a user account.
    Returns: (bool, message)
    """
    users = load_users()

    if username_to_delete not in users:
        return False, "User not found."

    if username_to_delete == acting_admin_username:
        return False, "You cannot delete your own account."

    # If the user to be deleted is an admin, ensure at least one active admin remains
    if users[username_to_delete].get("role") == "admin":
        # Count active admins *excluding* the one being deleted
        active_admins_after_delete = sum(
            1
            for u, data in users.items()
            if data.get("role") == "admin"
            and data.get("active", True)
            and u != username_to_delete
        )
        if active_admins_after_delete == 0:
            return (
                False,
                "Cannot delete this admin. Deleting this account would leave no active administrators.",
            )

    del users[username_to_delete]
    save_users(users)
    return True, f"User '{username_to_delete}' deleted successfully."
