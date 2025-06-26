#!/usr/bin/env python3
"""
ai_controller.py

A Python AI controller for Unity that:
  1. Listens for JSON "GET" requests from Unity every tick.
  2. Receives state updates as JSON (function handle_state).
  3. Maintains the latest state in a shared object protected by a lock.
  4. Requests a screen dump from Unity and receives it as a PIL Image.
  5. Analyzes the image and decides on actions.
  6. Sends commands back to Unity.
  7. Handles game states for game restart and end conditions.
  8. Handles clean shutdown on Ctrl+C or game end.

Updates need by the user:
- Python code:
  User should implement the ai_image_analyzer function to analyze the image and together with the 
  last received state (see function handle_state) it should decide on the next command.
  The decision can be made either when the images is received and has been analyzed or periodically in the send_loop function.

- Unity configuration:
  Update the game_config.json file for the AI player:
    - The port number should match the one used in the Python script.
    - the type should be set to "AI"
    - optionally: set the name and color of the AI
  Update game_config.json file for the Human player:
    - set the type to "Human"
    - set the inputConfig keypad parameters to the desired keys for movement and actions.
    - optionally: set the name and color of the player

  Unity game parameters:
    - timeScale: 1.0  (The game speed, 1.0 is normal speed)
    - aiTimeBetweenStates: 1.0 (The time between state updates to the AI, in seconds)
    - trainIterations: 1 (The number of training iterations for the AI, set to 1 for testing)
    - screenWidth: 1280 (The width of the game screen)
    - screenHeight: 720 (The height of the game screen)
    - screenResolutionScale: 0.5 (The scale of the game screen resolution, defines how small the images will be when sent to the python script)
    - useParallaxScrolling: false/true (Whether to use parallax scrolling in the game, the background will then move slower than the foreground)
    - autoLoadLevel: <level_name> (The name of the level to load automatically, e.g. "Level1" or leave empty for no auto-load)
Usage:
    python ai_controller.py <port>
"""
import socket
import sys
import json
import random
import signal
import threading
import time
from typing import Dict, Any
import base64
from typing import Optional
from PIL import Image
import io


# Global shutdown flag and server socket
shutdown = False
server_sock: socket.socket
next_command = None

def request_image(conn: socket.socket) -> Optional[Image.Image]:
    """
    Ask Unity for a screen dump and read it back as a PIL Image.
    Unity must reply with a 4-byte big-endian size, then the raw JPEG/PNG bytes.
    """
    try:
        conn.sendall(b"GET_IMAGE\n")

        # Read size prefix
        length_bytes = conn.recv(4)
        if len(length_bytes) < 4:
            print("Image request: incomplete length prefix")
            return None
        img_size = int.from_bytes(length_bytes, byteorder="big")

        # Read the image itself
        data = b""
        while len(data) < img_size:
            packet = conn.recv(img_size - len(data))
            if not packet:
                print("Image request: connection closed mid-image")
                return None
            data += packet

        return Image.open(io.BytesIO(data))
    except Exception as e:
        print("Failed to fetch image:", e)
        return None


def signal_handler(sig, frame):
    """Signal handler for a clean shutdown on Ctrl+C"""
    global shutdown, server_sock
    print("\nShutting down AI server...")
    shutdown = True
    try:
        server_sock.close()
    except Exception:
        pass


signal.signal(signal.SIGINT, signal_handler)


# ---------------------------------------------------------------------------
# Command builder library
# ---------------------------------------------------------------------------
class AICommandBuilder:
    """
    Builder for AI commands to send to Unity.

    Supported commands:
      - LEFT(amount: float 0-1)
      - RIGHT(amount: float 0-1)
      - JUMP(amount: float 0-1)
      - PICKUP
      - DROP
      - SHOOT
    """

    def __init__(self):
        self._commands = []  # type: List[str]

    def left(self, amount: float) -> "AICommandBuilder":
        assert 0.0 <= amount <= 1.0, "LEFT amount must be between 0.0 and 1.0"
        self._commands.append(f"LEFT:{amount:.2f}")
        return self

    def right(self, amount: float) -> "AICommandBuilder":
        assert 0.0 <= amount <= 1.0, "RIGHT amount must be between 0.0 and 1.0"
        self._commands.append(f"RIGHT:{amount:.2f}")
        return self

    def jump(self, amount: float) -> "AICommandBuilder":
        assert 0.0 <= amount <= 1.0, "JUMP amount must be between 0.0 and 1.0"
        self._commands.append(f"JUMP:{amount:.2f}")
        return self

    def pickup(self) -> "AICommandBuilder":
        self._commands.append("PICKUP")
        return self

    def drop(self) -> "AICommandBuilder":
        self._commands.append("DROP")
        return self

    def shoot(self) -> "AICommandBuilder":
        self._commands.append("SHOOT")
        return self

    def clear(self) -> "AICommandBuilder":
        """Clear any previously added commands."""
        self._commands.clear()
        return self

    def build(self) -> str:
        """Serialize all added commands into a single semicolon-delimited string."""
        return ";".join(self._commands)


def random_choose_cmd() -> str:
    """Randomly choose a command to send to Unity."""
    builder = AICommandBuilder().clear()
    cmd_type = random.choice([0, 1, 2, 3, 4, 5])
    if cmd_type == 0:
        builder.left(1.0)
    elif cmd_type == 1:
        builder.right(1.0)
    elif cmd_type == 2:
        builder.jump(1.0)
    elif cmd_type == 3:
        builder.pickup()
    elif cmd_type == 4:
        builder.drop()
    else:
        builder.shoot()
    return builder.build() + "\n"


def test_random_choose_cmd() -> str:
    """Randomly choose a command to send to Unity."""
    builder = AICommandBuilder().clear()
    cmd_type = random.choice([0, 1])
    # if cmd_type == 0:
    #     #builder.right(1.0)

    # if cmd_type == 1:
    builder.jump(1.0)
    return builder.build() + "\n"


def handle_state(state_str, state_lock, shared_state) -> bool:
    has_game_ended = False

    state_json: Dict[str, Any] = json.loads(state_str)
    with state_lock:
        shared_state["isDead"] = state_json.get("isDead", False)
        shared_state["numActivePlayers"] = state_json.get("numActivePlayers", 0)
        shared_state["hasWeapon"] = state_json.get("hasWeapon", False)
        shared_state["numWeapons"] = state_json.get("numWeapons", 0)
        shared_state["gameEnded"] = state_json.get("gameEnded", False)

    print(
        f"isDead: {shared_state['isDead']}, "
        f"numActivePlayers: {shared_state['numActivePlayers']}, "
        f"hasWeapon: {shared_state['hasWeapon']}, "
        f"numWeapons: {shared_state['numWeapons']}, "
        f"gameEnded: {shared_state['gameEnded']}"
    )

    # Check for end conditions
    if state_json.get("gameEnded", False) or shared_state["isDead"]:
        has_game_ended = True

    return has_game_ended

def ai_image_analyzer(image: Image.Image) -> str:
    """
    Analyze the image and return a command based on the analysis.
    This is a placeholder function and should be replaced with actual analysis logic.
    """
    # example how to use the API to build commands
    # builder = AICommandBuilder().clear()
    #     builder.shoot()
    #     builder.left(1.0)
    #     builder.right(0.5)
    #     builder.jump(0.5)
    #     builder.pickup()
    #     builder.drop()
    # cmd_str = builder.build() + "\n"
    # try:
    #     conn.sendall(cmd_str.encode("ascii"))
    
    # Example: Just return a random command for now
    return random_choose_cmd()

def handle_client(conn: socket.socket, addr, port: int):
    """Handle all communication with a single Unity client"""
    print(f"[Port {port}] Connected by {addr}")
    conn.settimeout(1.0)

    # Shared state and lock
    shared_state: Dict[str, Any] = {"isDead": False, "numActivePlayers": 0}
    state_lock = threading.Lock()

    # Events for coordination
    stop_event = threading.Event()

    def should_fetch_image(state: Dict[str, Any]) -> bool:
        # always fetch, or only when numActivePlayers changes, etc.
        return True

    def recv_loop():
        global next_command
        buffer = ""
        try:
            while not stop_event.is_set() and not shutdown:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue

                if not chunk:
                    continue

                buffer += chunk.decode("ascii", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        has_game_ended = handle_state(line, state_lock, shared_state)
                    except json.JSONDecodeError as e:
                        print("JSON parse error:", e, line)
                        continue

                    if has_game_ended:
                        print(f"[Port {port}] Game ended or player is dead.")
                        stop_event.set()
                        break
                    else:
                        if should_fetch_image(shared_state):
                            img = request_image(conn)
                            if img:
                                next_command = ai_image_analyzer(img)
                                img.save("latest_frame.png")
                                print("Saved latest_frame.png")
        finally:
            stop_event.set()

    def send_loop():
        """Send a command each second, based on the latest state."""
        global next_command
        while not stop_event.is_set() and not shutdown:
            # Snapshot the latest state
            with state_lock:
                latest = dict(shared_state)

            # next command is calculated when receiving the image
            # The user can also choose to send the command when the image is received in function ai_image_analyzer
            if next_command:
                try:
                    conn.sendall(next_command.encode("ascii"))     
                    next_command = None          
                except Exception as e:
                    print(f"[Port {port}] Send failed: {e}")
                    stop_event.set()
                    break

            # User can edit here:
            # Update this to send commands faster, by default periodically every 0.5 sec.
            # Sleep ~0.5s in short increments to detect stop_event promptly
            for _ in range(10):
                if stop_event.is_set() or shutdown:
                    break
                time.sleep(0.5)

    # Start receiver and sender threads
    receiver = threading.Thread(target=recv_loop, daemon=True)
    sender = threading.Thread(target=send_loop, daemon=True)
    receiver.start()
    sender.start()

    # Wait for game end or shutdown
    receiver.join()
    stop_event.set()
    sender.join()

    conn.close()
    print(f"[Port {port}] Connection closed.")


def run_ai_server(port: int):
    global server_sock
    HOST = "127.0.0.1"
    PORT = port

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(5)
    server_sock.settimeout(1.0)
    print(f"AI server listening on {HOST}:{PORT} (Ctrl+C to stop)...")

    while not shutdown:
        try:
            conn, addr = server_sock.accept()
            handle_client(conn, addr, PORT)
        except socket.timeout:
            continue
        except OSError:
            break

    print("AI server has shut down.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_controller.py <port>")
        # should be same port as in the unity configuration file for the player. (see game_config.json)
        sys.exit(1)
    try:
        port_num = int(sys.argv[1])
    except ValueError:
        print("Port must be an integer.")
        sys.exit(1)
    run_ai_server(port_num)
