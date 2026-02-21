from __future__ import annotations

from dataclasses import dataclass

from Quartz import (
    CGAssociateMouseAndMouseCursorPosition,
    CGDisplayPixelsHigh,
    CGDisplayPixelsWide,
    CGEventCreateMouseEvent,
    CGEventPost,
    CGMainDisplayID,
    CGWarpMouseCursorPosition,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)


@dataclass(frozen=True)
class ScreenSize:
    width: int
    height: int


class MacOSMouseController:
    def __init__(self) -> None:
        display = CGMainDisplayID()
        self.screen = ScreenSize(
            width=int(CGDisplayPixelsWide(display)),
            height=int(CGDisplayPixelsHigh(display)),
        )

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        x_clamped = min(max(x, 0.0), self.screen.width - 1.0)
        y_clamped = min(max(y, 0.0), self.screen.height - 1.0)
        return x_clamped, y_clamped

    def move(self, x: float, y: float) -> None:
        x, y = self.clamp(x, y)
        point = (x, y)
        move_event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, point, kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, move_event)
        CGWarpMouseCursorPosition(point)
        CGAssociateMouseAndMouseCursorPosition(True)

    def click_left(self, x: float, y: float) -> None:
        x, y = self.clamp(x, y)
        point = (x, y)
        down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, kCGMouseButtonLeft)
        up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
