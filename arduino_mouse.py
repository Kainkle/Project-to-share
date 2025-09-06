import time
import threading

try:
    import serial  # pip install pyserial
except Exception as e:
    raise SystemExit("pyserial is required. Install with: pip install pyserial") from e


class ArduinoMouse:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.lock = threading.Lock()
        self._keepalive_thread = None
        self._keepalive_stop = threading.Event()

    def connect(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Leonardo resets on open; wait for it to enumerate
        time.sleep(2.0)
        # Basic handshake and arm
        self._write("PING")
        _ = self._readline()
        self.arm()

    def start_keepalive(self, interval_sec: float = 4.0):
        # Sends PING periodically to keep the board ARMED while aiming
        self._keepalive_stop.clear()

        def loop():
            while not self._keepalive_stop.is_set():
                try:
                    self._write("PING")
                    # Drain any replies
                    while self.ser and self.ser.in_waiting:
                        self.ser.readline()
                except Exception:
                    pass
                self._keepalive_stop.wait(interval_sec)

        if self._keepalive_thread is None or not self._keepalive_thread.is_alive():
            self._keepalive_thread = threading.Thread(target=loop, daemon=True)
            self._keepalive_thread.start()

    def stop_keepalive(self):
        self._keepalive_stop.set()

    def arm(self):
        self._write("ARM")
        _ = self._readline()

    def disarm(self):
        if self.ser:
            self._write("DISARM")
            _ = self._readline()

    def move(self, dx: int, dy: int):
        self._write(f"M {int(dx)} {int(dy)}")
        self._drain()

    def click_left(self):
        self._write("CL"); self._drain()

    def press_left(self):
        self._write("LD"); self._drain()

    def release_left(self):
        self._write("LU"); self._drain()

    def press_right(self):
        self._write("RD"); self._drain()

    def release_right(self):
        self._write("RU"); self._drain()

    def close(self):
        try:
            self.stop_keepalive()
            self.disarm()
        except Exception:
            pass
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _write(self, line: str):
        if not self.ser:
            return
        data = (line + "\n").encode("ascii", errors="ignore")
        with self.lock:
            self.ser.write(data)

    def _readline(self) -> bytes:
        if not self.ser:
            return b""
        try:
            return self.ser.readline()
        except Exception:
            return b""

    def _drain(self):
        if not self.ser:
            return
        try:
            while self.ser.in_waiting:
                self.ser.readline()
        except Exception:
            pass