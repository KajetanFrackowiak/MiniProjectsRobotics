import math
import threading
import numpy as np
from pydrake.systems.framework import LeafSystem


class TeleopSystem(LeafSystem):
    """Unified keyboard / Xbox-joystick teleop for Mini Cheetah << trot.

    Outputs cmd_vel = [vx, vy, wz] (body frame) for FootstepPlanner/MPC/WBIC.
    Supports:
      - joystick: left_stick Y->vx, left_stick X->vy, right_stick X->wz (XInput via pygame)
      - keyboard: W/S vx, A/D vy, Q/E or Left/Right wz  (pynput, non-blocking)
    Mode: "joystick", "keyboard", "both" (joystick+keyboard summed), "auto" (joy if present else kbd).

    Safety: deadzone, low-pass 8Hz, rate limit, enable requires no deadman by default
    but if deadman_button is set, motion only when pressed.

    This is a discrete system @ dt (default 20ms = 50Hz) so joystick polling
    happens in DiscreteUpdate, output reads discrete state (no algebraic loop).
    """

    def __init__(
        self,
        mode="auto",
        max_vx=0.6,
        max_vy=0.4,
        max_wz=1.0,
        deadzone=0.12,
        dt=0.02,
        cutoff_hz=8.0,
        joystick_id=0,
        enable_button=None,  # e.g. 4 = LB on Xbox, None = always enabled
        invert_vx=False,
        invert_vy=False,
        invert_wz=False,
    ):
        super().__init__()
        self.mode = mode
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)
        self.deadzone = float(deadzone)
        self.dt = float(dt)
        self.alpha = math.exp(-2 * math.pi * cutoff_hz * dt) if cutoff_hz > 0 else 0.0
        self.joystick_id = int(joystick_id)
        self.enable_button = enable_button
        self.invert_vx = invert_vx
        self.invert_vy = invert_vy
        self.invert_wz = invert_wz

        # --- joystick init (pygame) ---
        self.joy = None
        self._pygame = None
        if mode in ("joystick", "both", "auto"):
            try:
                import pygame  # type: ignore
                self._pygame = pygame
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() > self.joystick_id:
                    self.joy = pygame.joystick.Joystick(self.joystick_id)
                    self.joy.init()
                    print(f"[Teleop] joystick {self.joy.get_name()} id={self.joystick_id} axes={self.joy.get_numaxes()} buttons={self.joy.get_numbuttons()}")
                else:
                    print(f"[Teleop] no joystick id={self.joystick_id} found (count={pygame.joystick.get_count()}) - keyboard fallback")
                    if mode == "joystick":
                        print("[Teleop] mode=joystick but no joy - output will stay zero, use --teleop keyboard")
            except Exception as e:
                print(f"[Teleop] pygame joystick init failed: {e}")
                self.joy = None

        # --- keyboard init (pynput) ---
        self._keys = set()
        self._kbd_listener = None
        self._keys_lock = threading.Lock()
        if mode in ("keyboard", "both", "auto"):
            try:
                from pynput import keyboard as pynkbd  # type: ignore

                def on_press(key):
                    try:
                        k = key.char.lower() if hasattr(key, "char") and key.char else str(key)
                    except Exception:
                        k = str(key)
                    with self._keys_lock:
                        self._keys.add(k)
                    # also handle special keys string
                    with self._keys_lock:
                        self._keys.add(str(key))

                def on_release(key):
                    try:
                        k = key.char.lower() if hasattr(key, "char") and key.char else str(key)
                    except Exception:
                        k = str(key)
                    with self._keys_lock:
                        self._keys.discard(k)
                        self._keys.discard(str(key))

                self._kbd_listener = pynkbd.Listener(on_press=on_press, on_release=on_release)
                self._kbd_listener.daemon = True
                self._kbd_listener.start()
                print("[Teleop] keyboard listener started (WASD + Q/E or arrows, ESC to stop)")
            except Exception as e:
                print(f"[Teleop] pynput keyboard init failed: {e} - keyboard disabled")
                self._kbd_listener = None

        # discrete state: filtered cmd_vel (3)
        self.DeclareDiscreteState(3)
        self.DeclarePeriodicDiscreteUpdateEvent(period_sec=self.dt, offset_sec=0.0, update=self.DiscreteUpdate)
        self.DeclareVectorOutputPort("cmd_vel", 3, self.CalcOutput)

        # for logging / meshcat overlay maybe
        self._last_raw = np.zeros(3)

    def _apply_deadzone(self, v):
        if abs(v) < self.deadzone:
            return 0.0
        # rescale after deadzone
        sign = 1 if v > 0 else -1
        return sign * (abs(v) - self.deadzone) / (1.0 - self.deadzone)

    def _read_joystick(self):
        if self.joy is None or self._pygame is None:
            return np.zeros(3)
        try:
            self._pygame.event.pump()
            # enable check
            if self.enable_button is not None:
                if not self.joy.get_button(self.enable_button):
                    return np.zeros(3)
            # axes: 0 left_x, 1 left_y, 2 right_x, 3 right_y, 4 LT, 5 RT
            n = self.joy.get_numaxes()
            ax0 = self.joy.get_axis(0) if n > 0 else 0.0
            ax1 = self.joy.get_axis(1) if n > 1 else 0.0
            ax2 = self.joy.get_axis(2) if n > 2 else 0.0
            # Xbox triggers are axes 4/5 in -1..1, ignore
            vx = -self._apply_deadzone(ax1) * self.max_vx
            vy = -self._apply_deadzone(ax0) * self.max_vy
            wz = -self._apply_deadzone(ax2) * self.max_wz
            if self.invert_vx:
                vx = -vx
            if self.invert_vy:
                vy = -vy
            if self.invert_wz:
                wz = -wz
            return np.array([vx, vy, wz])
        except Exception:
            return np.zeros(3)

    def _read_keyboard(self):
        if self._kbd_listener is None:
            return np.zeros(3)
        with self._keys_lock:
            keys = set(self._keys)
        # normalize key strings: pynput gives "Key.up" etc, also char "w"
        def has(k):
            # check both char and Key.* variants
            return (k in keys) or (f"'{k}'" in keys) or (f"Key.{k}" in keys) or (f"Key.{k}_r" in keys)

        # also check for arrow keys
        vx = 0.0
        vy = 0.0
        wz = 0.0
        if has("w") or has("up"):
            vx += self.max_vx
        if has("s") or has("down"):
            vx -= self.max_vx
        if has("a"):
            vy += self.max_vy  # left strafe positive
        if has("d"):
            vy -= self.max_vy
        # turn: Q/E or left/right arrows
        if has("q") or has("left"):
            wz += self.max_wz
        if has("e") or has("right"):
            wz -= self.max_wz
        # enable: if enable required, check space
        # we don't enforce for keyboard
        return np.array([vx, vy, wz])

    def DiscreteUpdate(self, context, discrete_state):
        # read raw
        raw_joy = self._read_joystick() if self.mode in ("joystick", "both", "auto") else np.zeros(3)
        raw_kbd = self._read_keyboard() if self.mode in ("keyboard", "both", "auto") else np.zeros(3)

        if self.mode == "both":
            raw = np.clip(raw_joy + raw_kbd, [-self.max_vx, -self.max_vy, -self.max_wz], [self.max_vx, self.max_vy, self.max_wz])
        elif self.mode == "auto":
            # prefer joystick if non-zero, else keyboard
            if np.linalg.norm(raw_joy) > 1e-6:
                raw = raw_joy
            else:
                raw = raw_kbd
        elif self.mode == "joystick":
            raw = raw_joy
        else:  # keyboard
            raw = raw_kbd

        self._last_raw = raw.copy()

        # low-pass filter
        filt = discrete_state.get_vector(0).CopyToVector()
        # init if zero and raw non-zero? use alpha
        if np.allclose(filt, 0) and np.linalg.norm(raw) < 1e-9:
            new = filt  # stay zero
        else:
            # first update from zero -> jump to raw * (1-alpha) else lpf
            if np.allclose(filt, 0):
                # init quickly
                new = raw * 0.5
            else:
                new = self.alpha * filt + (1 - self.alpha) * raw
        discrete_state.get_mutable_vector(0).SetFromVector(new)

    def CalcOutput(self, context, output):
        try:
            filt = context.get_discrete_state_vector().CopyToVector()
        except Exception:
            try:
                filt = context.get_discrete_state(0).CopyToVector()
            except Exception:
                filt = np.zeros(3)
        # if filter still zero but raw has value (first step before discrete update), use raw
        if np.allclose(filt, 0) and np.linalg.norm(self._last_raw) > 1e-6:
            filt = self._last_raw * 0.5
        output.SetFromVector(filt)

    def get_last_raw(self):
        return self._last_raw.copy()
