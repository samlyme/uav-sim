import asyncio
import json
import sys
from typing import Annotated
import numpy as np
import numpy.typing as npt
import websockets
import tty
import termios

Num = np.float64
EulerAngle = Annotated[npt.NDArray[Num], (3,)]
Vector3 = Annotated[npt.NDArray[Num], (3,)]
Quaternion = Annotated[npt.NDArray[Num], (4,)]
RotationMatrix = Annotated[npt.NDArray[Num], (3, 3)]

def e_to_q(e: EulerAngle) -> Quaternion:
    """
    Convert Euler angles to quaternion.
    This is a placeholder function; actual implementation may vary.
    """
    phi, theta, psi = e[0], e[1], e[2]
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ], dtype=Num)

    return q / np.linalg.norm(q)

def qm(a: Quaternion, b: Quaternion) -> Quaternion:
    """
    Quaternion multiplication.
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=Num)

def q_to_R(q: Quaternion) -> RotationMatrix:
    """
    Convert quaternion to rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ], dtype=Num)
    
LINEAR_SPEED = 1
ANGULAR_SPEED = np.pi / 4


class KinematicObject:
    def __init__(self, P_i_0: Vector3, q_i_0: Quaternion):
        self.P_i = P_i_0  # Position in inertial frame (pn, pe, -h)
        self.q_i = q_i_0

    def update(self, v_b: Vector3, omega_b: EulerAngle, dt: Num):
        R = q_to_R(self.q_i)  # Rotation matrix from inertial to body frame
        v_i = R @ v_b  
        self.P_i += v_i * dt

        dq = e_to_q(omega_b * dt)  
        self.q_i = qm(self.q_i, dq)  # Update quaternion

        return self.P_i, self.q_i


v_b = np.array([0.0, 0.0, 0.0], dtype=Num)
omega_b = np.array([0.0, 0.0, 0.0], dtype=Num)
dt = np.float64(0.02)


async def broadcaster(ws):
    # TODO: move this outside of the broadcaster function
    t = np.float64(0.0)  
    P_i = np.array([0.0, 0.0, 0.0], dtype=Num)  # Initial position in inertial frame
    q = e_to_q(np.zeros(3, dtype=Num))  # Initial orientation as quaternion

    drone = KinematicObject(P_i_0=P_i, q_i_0=e_to_q(np.zeros(3, dtype=Num)))

    while True:
        P_i, q = drone.update(v_b, omega_b, dt)

        state = {
            "time": t.astype(float),  
            "p_i": P_i.tolist(),  
            "q": q.tolist(),  
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
