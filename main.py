import asyncio
import json
import sys
import numpy as np
import websockets
import tty
import termios

Num = np.float32

LINEAR_SPEED = 1
ANGULAR_SPEED = np.pi / 4  # radians per second


def R_b_v(e: np.ndarray) -> np.ndarray: # Rotation matrix from body frame to inertial frame
    theta, phi, eta = e[1], e[0], e[2] 
    """
    Rotation matrix from body frame to inertial frame given roll (phi), pitch (theta), and yaw (eta).
    """
    R = np.array([
        [ np.cos(theta) * np.cos(eta), np.cos(theta) * np.sin(eta), -np.sin(theta) ],
        [ -np.cos(phi) * np.sin(eta) + np.sin(phi) * np.sin(theta) * np.cos(eta), np.cos(phi) * np.cos(eta) + np.sin(phi) * np.sin(theta) * np.sin(eta), np.sin(phi) * np.cos(theta)],
        [ np.sin(phi) * np.sin(eta) + np.cos(phi) * np.sin(theta) * np.cos(eta), -np.sin(phi) * np.cos(eta) + np.cos(phi) * np.sin(theta) * np.sin(eta), np.cos(phi) * np.cos(theta)],
    ], dtype=Num)
    return R

# TODO: make this function mutate np arrays instead of returning new ones
# This will improve performance by avoiding unnecessary memory allocation and copying.
def update_state(
    P_i_t0: np.ndarray,  # Position in inertial frame (pn, pe, -h)
    e_t0: np.ndarray,    # Euler angles (phi, theta, eta)
    v_b: np.ndarray,  # Velocity in body frame (u, v, w)
    omega_b: np.ndarray,  # Angular velocity in body frame (p, q, r)
    dt: float = 0.0,  # Time in seconds
):
    R = R_b_v(e_t0)  # Rotation matrix from body to inertial frame
    P_i = P_i_t0 + R @ v_b * dt  # Position in inertial frame at time t
    e = e_t0 + omega_b * dt  # Euler angles at time t

    return P_i, e


t = 0.0
dt = 0.02
P_i = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial position in inertial frame
e = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial Euler angles (phi, theta, eta)
v_b = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial velocity in body frame (u, v, w)
omega_b = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial angular velocity in body frame (p, q, r

async def broadcaster(ws):
    
    while True:
        P_i, e = update_state(
            P_i_t0=P_i,
            e_t0=e,
            v_b=v_b,
            omega_b=omega_b,
            dt=dt
        )

        state = {
            "time": t,
            "p_i": P_i.tolist(),  # Convert numpy array to list for JSON serialization
            "e": e.tolist(),  # Convert numpy array to list for JSON serialization
            "v_b": v_b.tolist(),  # Convert numpy array to list for JSON serialization
            "omega_b": omega_b.tolist()  # Convert numpy array to list for JSON serialization
        }

        await ws.send(json.dumps(state))
        await asyncio.sleep(dt)
        t += dt

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

async def key_reader():
    loop = asyncio.get_running_loop()
    print("CONTROL KEYS: W/S = fwd/back, A/D = left/right, E/Q = up/down")
    print("              J/L = yaw left/right, I/K = pitch up/down, U/O = roll left/right")
    print("Press 'x' to exit.")

    while True:
        ch = await loop.run_in_executor(None, getch)
        if ch == 'w':
            print("Moving forward")
        elif ch == 's':
            print("Moving backward")
        elif ch == 'a':
            print("Turning left")
        elif ch == 'd':
            print("Turning right")
        elif ch == 'e':
            print("Moving up")
        elif ch == 'q':
            print("Moving down")
        elif ch == 'j':
            print("Yawing left")
        elif ch == 'l':
            print("Yawing right")
        elif ch == 'i':
            print("Pitching up")
        elif ch == 'k':
            print("Pitching down")
        elif ch == 'u':
            print("Rolling left")
        elif ch == 'o':
            print("Rolling right")
        elif ch == 'x':
            print("Exiting...")
            break

async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(broadcaster, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())