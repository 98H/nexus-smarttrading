"""FIX Engine Gateway managing sessions, logon authentication, and sequence tracking."""

from dataclasses import dataclass
from enum import Enum

from src.gateway.fix.fix_parser import FIXMessage, FIXParser, FixParseError


class SessionState(Enum):
    """Lifecycle states of a FIX session connection."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    AUTHENTICATING = "AUTHENTICATING"
    AUTHENTICATED = "AUTHENTICATED"
    REJECTED = "REJECTED"
    TERMINATED = "TERMINATED"


@dataclass
class FIXSession:
    """Stateful representation of an active FIX session."""

    session_id: str
    begin_string: str
    sender_comp_id: str
    target_comp_id: str
    heartbeat_interval: int
    inbound_seq_num: int = 1
    outbound_seq_num: int = 1
    state: SessionState = SessionState.AUTHENTICATED

    @property
    def is_authenticated(self) -> bool:
        """Check if session is successfully authenticated."""
        return self.state == SessionState.AUTHENTICATED

    @property
    def is_active(self) -> bool:
        """Check if session is currently active and authenticated."""
        return self.state == SessionState.AUTHENTICATED


class FIXEngineGateway:
    """Inbound FIX Gateway maintaining authenticated session states and sequence tracking."""

    def __init__(
        self,
        target_comp_id: str,
        supported_begin_strings: tuple[str, ...] | list[str] | set[str] | None = None,
        parser: FIXParser | None = None,
    ) -> None:
        self.target_comp_id = target_comp_id
        self._parser = parser or FIXParser(
            supported_begin_strings=supported_begin_strings
        )
        self._sessions: dict[str, FIXSession] = {}
        self._sessions_by_sender: dict[str, FIXSession] = {}

    def process_inbound(self, raw_data: bytes) -> tuple[FIXMessage, FIXSession]:
        """Parse raw FIX wire frame and process session-level state transitions."""
        try:
            msg = self._parser.parse(raw_data)
        except FixParseError:
            self._reject_potential_sender(raw_data)
            raise

        sender_comp_id = msg.get(49)
        target_comp_id = msg.get(56)

        if not target_comp_id or target_comp_id != self.target_comp_id:
            if sender_comp_id:
                self._terminate_session(sender_comp_id)
            raise FixParseError(
                f"TargetCompID mismatch: expected {self.target_comp_id}, got {target_comp_id}"
            )

        if not sender_comp_id:
            raise FixParseError("Missing SenderCompID (tag 49)")

        session = self._sessions_by_sender.get(sender_comp_id)

        if session is None or not session.is_authenticated:
            if msg.msg_type != "A":
                if session is not None:
                    self._terminate_session(sender_comp_id)
                raise FixParseError(
                    f"First message must be Logon (MsgType=A), received MsgType={msg.msg_type}"
                )

            heartbeat_interval = msg.get_int(108, 30)
            inbound_seq_num = msg.get_int(34, 1)
            session_id = f"{msg.begin_string}:{self.target_comp_id}->{sender_comp_id}"

            session = FIXSession(
                session_id=session_id,
                begin_string=msg.begin_string,
                sender_comp_id=sender_comp_id,
                target_comp_id=self.target_comp_id,
                heartbeat_interval=heartbeat_interval,
                inbound_seq_num=inbound_seq_num,
                state=SessionState.AUTHENTICATED,
            )
            self._sessions[session_id] = session
            self._sessions_by_sender[sender_comp_id] = session
        else:
            inbound_seq_num = msg.get_int(34)
            if inbound_seq_num is not None:
                session.inbound_seq_num = inbound_seq_num
            if msg.msg_type == "A":
                hb = msg.get_int(108)
                if hb is not None:
                    session.heartbeat_interval = hb

        return msg, session

    def get_session(self, session_id: str) -> FIXSession | None:
        """Lookup session by session_id."""
        return self._sessions.get(session_id)

    def get_session_by_sender(self, sender_comp_id: str) -> FIXSession | None:
        """Lookup session by SenderCompID."""
        return self._sessions_by_sender.get(sender_comp_id)

    def list_sessions(self) -> list[FIXSession]:
        """Return all managed sessions."""
        return list(self._sessions.values())

    def is_connected(self, sender_comp_id: str) -> bool:
        """Determine if a client currently has an active authenticated session."""
        session = self._sessions_by_sender.get(sender_comp_id)
        if session is None:
            return False
        return session.is_authenticated

    def _terminate_session(self, sender_comp_id: str) -> None:
        """Reject and remove an existing session."""
        session = self._sessions_by_sender.pop(sender_comp_id, None)
        if session is not None:
            session.state = SessionState.REJECTED
            self._sessions.pop(session.session_id, None)

    def _reject_potential_sender(self, raw_data: bytes) -> None:
        """Attempt best-effort identification of sender to reject upon parse error."""
        tag = b"\x0149="
        idx = raw_data.find(tag)
        if idx != -1:
            start = idx + len(tag)
            end = raw_data.find(b"\x01", start)
            if end != -1:
                sender = raw_data[start:end].decode("ascii", errors="ignore")
                if sender in self._sessions_by_sender:
                    self._terminate_session(sender)