import asyncio
import json
import sys
import numpy as np
import websockets
import tty
import termios

Num = np.float32

LINEAR_SPEED = 1
ANGULAR_SPEED = np.pi / 4


def R_b_v(e: np.ndarray) -> np.ndarray:
    """
    Rotation matrix from body frame to inertial frame using Euler angles.
    """
    phi, theta, psi = e[0], e[1], e[2]  
    R = np.array(
        [
            [
                np.cos(theta) * np.cos(psi),
                np.sin(theta) * np.sin(phi) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
            ],
            [
                np.cos(theta) * np.sin(psi),
                np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
            ],
            [
                -np.sin(theta),
                np.sin(phi) * np.cos(theta),
                np.cos(phi) * np.cos(theta),
            ]
        ],
        dtype=Num,
    )
    return R


def J_b_v(e: np.ndarray) -> np.ndarray:
    """
    Jacobian of the Euler angles with respect to the angular velocity in body frame.
    This is a 3x3 matrix that relates the angular velocity in body frame to the time derivative of the Euler angles.
    """
    phi, theta, psi = e[0], e[1], e[2]  # noqa: F841
    return np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ],
        dtype=Num,
    )


class Drone:
    def __init__(self, P_i_0: np.ndarray, e_0: np.ndarray):
        self.P_i = P_i_0  # Position in inertial frame (pn, pe, -h)
        self.e = e_0  # Euler angles (phi, theta, eta)

    def update(self, v_b: np.ndarray, omega_b: np.ndarray, dt: Num):
        J = J_b_v(self.e)
        de = J.T @ omega_b
        self.e = self.e + de * dt

        R = R_b_v(self.e)
        dp = R @ v_b
        self.P_i = self.P_i + dp * dt

        print("de", de)
        print("dp", dp)
        return self.P_i, self.e


v_b = np.array([0.0, 0.0, 0.0], dtype=Num)
omega_b = np.array([0.0, 0.0, 0.0], dtype=Num)
dt = np.float32(0.02)


async def broadcaster(ws):
    # TODO: move this outside of the broadcaster function
    t = np.float32(0.0)  
    P_i = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial position in inertial frame
    e = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial Euler angles (phi, theta, psi)

    drone = Drone(P_i_0=P_i, e_0=e)

    while True:
        P_i, e = drone.update(v_b, omega_b, dt)

        state = {
            "time": t.astype(float),  
            "p_i": P_i.tolist(),  
            "e": e.tolist(),  
            "v_b": v_b.tolist(),  
            "omega_b": omega_b.tolist(),
        }

        await ws.send(json.dumps(state))
        await asyncio.sleep(dt.astype(float))
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
    print(
        "              J/L = yaw left/right, I/K = pitch up/down, U/O = roll left/right"
    )
    print("Press 'x' to exit.")

    while True:
        ch = await loop.run_in_executor(None, getch)
        if ch == "w":
            if v_b[0] == 0: 
                v_b[0] = LINEAR_SPEED  # Move forward
            else:
                v_b[0] = 0
        elif ch == "s":
            if v_b[0] == 0:
                v_b[0] = -LINEAR_SPEED  # Move backward
            else:
                v_b[0] = 0
        elif ch == "a":
            if v_b[1] == 0:
                v_b[1] = -LINEAR_SPEED  # Move left
            else:
                v_b[1] = 0
        elif ch == "d":
            if v_b[1] == 0:
                v_b[1] = LINEAR_SPEED  # Move right
            else:
                v_b[1] = 0
        elif ch == "e":
            if v_b[2] == 0:
                v_b[2] = -LINEAR_SPEED  # Move up
            else:
                v_b[2] = 0
        elif ch == "q":
            if v_b[2] == 0:
                v_b[2] = LINEAR_SPEED  # Move down
            else:
                v_b[2] = 0
        elif ch == "j":
            if omega_b[2] == 0:
                omega_b[2] = -ANGULAR_SPEED  # Yaw left
            else:
                omega_b[2] = 0
        elif ch == "l":
            if omega_b[2] == 0:
                omega_b[2] = ANGULAR_SPEED  # Yaw right
            else:
                omega_b[2] = 0
        elif ch == "i":
            if omega_b[1] == 0:
                omega_b[1] = ANGULAR_SPEED
            else:
                omega_b[1] = 0
        elif ch == "k":
            if omega_b[1] == 0:
                omega_b[1] = -ANGULAR_SPEED
            else:
                omega_b[1] = 0
        elif ch == "u":
            if omega_b[0] == 0:
                omega_b[0] = -ANGULAR_SPEED
            else:
                omega_b[0] = 0 
        elif ch == "o":
            if omega_b[0] == 0:
                omega_b[0] = ANGULAR_SPEED
            else:
                omega_b[0] = 0 
        elif ch == "x":
            print("Exiting...")
            sys.exit(0)
            break


async def main():
    # print("yaw test")
    # for i in range(8):
    #     e = np.array([0, 0, i * np.pi / 4], dtype=Num)  
        
    #     print(R_b_v(e) @ np.array([1, 0, 0], dtype=Num))  

    # print("pitch test")
    # for i in range(8):
    #     e = np.array([0, i * np.pi / 4, 0], dtype=Num)  
        
    #     print(R_b_v(e) @ np.array([1, 0, 0], dtype=Num))  

    # print("roll test")
    # for i in range(8):
    #     e = np.array([i * np.pi / 4, 0, 0], dtype=Num)  
        
    #     print(R_b_v(e) @ np.array([1, 0, 0], dtype=Num))  
    print("Starting key reader...")
    asyncio.create_task(key_reader())

    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(broadcaster, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
