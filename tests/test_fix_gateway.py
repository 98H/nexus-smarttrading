from datetime import datetime, timezone
import pytest

from src.gateway.fix.fix_parser import FIXMessage, FIXParser, FixParseError
from src.gateway.fix.fix_gateway import FIXEngineGateway, SessionState


SOH = b"\x01"


def build_raw_fix_message(
    begin_string: str,
    body_fields: list[tuple[int, str]],
    override_checksum: str | None = None,
    override_body_length: int | None = None,
) -> bytes:
    """Helper to construct deterministic raw FIX wire protocol byte frames."""
    body_serialized = "".join(f"{tag}={val}\x01" for tag, val in body_fields).encode(
        "ascii"
    )

    body_length = (
        len(body_serialized) if override_body_length is None else override_body_length
    )
    header_serialized = f"8={begin_string}\x019={body_length}\x01".encode("ascii")

    full_payload = header_serialized + body_serialized

    if override_checksum is not None:
        checksum_str = override_checksum
    else:
        checksum_str = f"{sum(full_payload) % 256:03d}"

    return full_payload + f"10={checksum_str}\x01".encode("ascii")


def create_logon_fields(
    sender: str = "CLIENT_A",
    target: str = "GATEWAY_B",
    seq_num: int = 1,
    heartbeat_interval: int = 30,
) -> list[tuple[int, str]]:
    """Standard required fields for a FIX Logon (MsgType=A)."""
    now_utc = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
    return [
        (35, "A"),  # MsgType: Logon
        (49, sender),  # SenderCompID
        (56, target),  # TargetCompID
        (34, str(seq_num)),  # MsgSeqNum
        (52, now_utc),  # SendingTime
        (98, "0"),  # EncryptMethod: None
        (108, str(heartbeat_interval)),  # HeartBtInt
    ]


class TestFIXParser:
    """Unit tests covering low-level protocol parsing in src/gateway/fix/fix_parser.py."""

    @pytest.fixture
    def parser(self) -> FIXParser:
        return FIXParser()

    def test_parse_valid_fix44_logon_message(self, parser: FIXParser) -> None:
        fields = create_logon_fields(sender="BUY_SIDE", target="SELL_SIDE", seq_num=1)
        raw_stream = build_raw_fix_message("FIX.4.4", fields)

        msg = parser.parse(raw_stream)

        assert isinstance(msg, FIXMessage)
        assert msg.begin_string == "FIX.4.4"
        assert msg.msg_type == "A"
        assert msg.get(35) == "A"
        assert msg.get(49) == "BUY_SIDE"
        assert msg.get(56) == "SELL_SIDE"
        assert msg.get_int(34) == 1
        assert msg.get_int(108) == 30
        assert msg.body_length == len(
            "".join(f"{tag}={val}\x01" for tag, val in fields).encode("ascii")
        )

    def test_parse_valid_fixt11_fix50_logon_message(self, parser: FIXParser) -> None:
        fields = create_logon_fields(sender="FX_TRADER", target="GATEWAY_V5", seq_num=1)
        # FIX 5.0 typically uses FIXT.1.1 transport header with MsgType=A
        raw_stream = build_raw_fix_message("FIXT.1.1", fields)

        msg = parser.parse(raw_stream)

        assert isinstance(msg, FIXMessage)
        assert msg.begin_string == "FIXT.1.1"
        assert msg.msg_type == "A"
        assert msg.get(49) == "FX_TRADER"
        assert msg.get(56) == "GATEWAY_V5"
        assert msg.get_int(34) == 1

    def test_parse_invalid_checksum_raises_fix_parse_error(
        self, parser: FIXParser
    ) -> None:
        fields = create_logon_fields()
        raw_stream = build_raw_fix_message("FIX.4.4", fields, override_checksum="999")

        with pytest.raises(FixParseError):
            parser.parse(raw_stream)

    def test_parse_unsupported_begin_string_raises_fix_parse_error(
        self, parser: FIXParser
    ) -> None:
        fields = create_logon_fields()
        # FIX 4.2 or proprietary begin strings are not supported
        raw_stream = build_raw_fix_message("FIX.4.2", fields)

        with pytest.raises(FixParseError):
            parser.parse(raw_stream)

    def test_parse_invalid_body_length_raises_fix_parse_error(
        self, parser: FIXParser
    ) -> None:
        fields = create_logon_fields()
        # Body length deliberately incorrect
        raw_stream = build_raw_fix_message("FIX.4.4", fields, override_body_length=4)

        with pytest.raises(FixParseError):
            parser.parse(raw_stream)

    def test_parse_malformed_empty_payload_raises_fix_parse_error(
        self, parser: FIXParser
    ) -> None:
        with pytest.raises(FixParseError):
            parser.parse(b"")

    def test_parse_missing_mandatory_checksum_tag_raises_fix_parse_error(
        self, parser: FIXParser
    ) -> None:
        truncated_bytes = b"8=FIX.4.4\x019=15\x0135=A\x0149=SENDER\x01"
        with pytest.raises(FixParseError):
            parser.parse(truncated_bytes)

    def test_parse_missing_begin_string_tag_raises_fix_parse_error(
        self, parser: FIXParser
    ) -> None:
        raw = b"9=20\x0135=A\x0149=SENDER\x0110=100\x01"
        with pytest.raises(FixParseError):
            parser.parse(raw)


class TestFIXEngineGateway:
    """Unit tests covering gateway session state management in src/gateway/fix/fix_gateway.py."""

    @pytest.fixture
    def gateway(self) -> FIXEngineGateway:
        # Gateway configured with supported targets and protocols FIX 4.4 and FIXT.1.1
        return FIXEngineGateway(target_comp_id="TARGET_GW")

    def test_valid_fix44_logon_initializes_authenticated_session(
        self, gateway: FIXEngineGateway
    ) -> None:
        sender_id = "CLIENT_44"
        fields = create_logon_fields(
            sender=sender_id, target="TARGET_GW", seq_num=1, heartbeat_interval=30
        )
        raw_stream = build_raw_fix_message("FIX.4.4", fields)

        decoded_msg, session = gateway.process_inbound(raw_stream)

        assert isinstance(decoded_msg, FIXMessage)
        assert decoded_msg.msg_type == "A"
        assert decoded_msg.begin_string == "FIX.4.4"

        assert session is not None
        assert session.session_id == f"FIX.4.4:TARGET_GW->{sender_id}"
        assert session.state == SessionState.AUTHENTICATED
        assert session.is_authenticated is True
        assert session.heartbeat_interval == 30
        assert session.inbound_seq_num == 1

        active_session = gateway.get_session(session.session_id)
        assert active_session is session
        assert active_session.is_active is True

    def test_valid_fix50_fixt11_logon_initializes_authenticated_session(
        self, gateway: FIXEngineGateway
    ) -> None:
        sender_id = "CLIENT_50"
        fields = create_logon_fields(
            sender=sender_id, target="TARGET_GW", seq_num=1, heartbeat_interval=60
        )
        raw_stream = build_raw_fix_message("FIXT.1.1", fields)

        decoded_msg, session = gateway.process_inbound(raw_stream)

        assert isinstance(decoded_msg, FIXMessage)
        assert decoded_msg.msg_type == "A"
        assert decoded_msg.begin_string == "FIXT.1.1"

        assert session.state == SessionState.AUTHENTICATED
        assert session.is_authenticated is True
        assert session.heartbeat_interval == 60
        assert session.inbound_seq_num == 1

    def test_invalid_checksum_rejects_session_and_raises(
        self, gateway: FIXEngineGateway
    ) -> None:
        sender_id = "CLIENT_ERR_CSUM"
        fields = create_logon_fields(sender=sender_id, target="TARGET_GW")
        corrupted_raw = build_raw_fix_message("FIX.4.4", fields, override_checksum="000")

        with pytest.raises(FixParseError):
            gateway.process_inbound(corrupted_raw)

        # Ensure no authenticated session was created or retained
        sessions = gateway.list_sessions()
        assert not any(s.sender_comp_id == sender_id for s in sessions)
        assert gateway.is_connected(sender_id) is False

    def test_unsupported_begin_string_rejects_connection_and_raises(
        self, gateway: FIXEngineGateway
    ) -> None:
        sender_id = "CLIENT_ERR_VER"
        fields = create_logon_fields(sender=sender_id, target="TARGET_GW")
        unsupported_raw = build_raw_fix_message("FIX.4.1", fields)

        with pytest.raises(FixParseError):
            gateway.process_inbound(unsupported_raw)

        assert gateway.is_connected(sender_id) is False
        assert gateway.get_session_by_sender(sender_id) is None

    def test_non_logon_as_first_message_is_rejected_without_authentication(
        self, gateway: FIXEngineGateway
    ) -> None:
        # First message sent is Heartbeat (MsgType=0) rather than Logon (MsgType=A)
        now_utc = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
        heartbeat_fields = [
            (35, "0"),
            (49, "ROGUE_CLIENT"),
            (56, "TARGET_GW"),
            (34, "1"),
            (52, now_utc),
        ]
        raw_stream = build_raw_fix_message("FIX.4.4", heartbeat_fields)

        with pytest.raises(FixParseError):
            gateway.process_inbound(raw_stream)

        session = gateway.get_session_by_sender("ROGUE_CLIENT")
        assert session is None or session.state != SessionState.AUTHENTICATED

    def test_target_comp_id_mismatch_is_rejected(
        self, gateway: FIXEngineGateway
    ) -> None:
        fields = create_logon_fields(
            sender="CLIENT_A", target="WRONG_TARGET", seq_num=1
        )
        raw_stream = build_raw_fix_message("FIX.4.4", fields)

        with pytest.raises(FixParseError):
            gateway.process_inbound(raw_stream)

        assert gateway.is_connected("CLIENT_A") is False