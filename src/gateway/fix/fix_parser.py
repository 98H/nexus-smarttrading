"""FIX protocol parser supporting FIX 4.4 and FIXT.1.1 / FIX 5.0 framing."""

SOH = b"\x01"
DEFAULT_SUPPORTED_BEGIN_STRINGS = frozenset({"FIX.4.4", "FIXT.1.1"})


class FixParseError(Exception):
    """Raised when a FIX message violates wire protocol specifications."""


class FIXMessage:
    """Structured representation of a parsed FIX protocol message."""

    def __init__(
        self,
        begin_string: str,
        body_length: int,
        msg_type: str,
        fields: dict[int, str],
        raw_bytes: bytes = b"",
    ) -> None:
        self.begin_string = begin_string
        self.body_length = body_length
        self.msg_type = msg_type
        self.fields = fields
        self.raw_bytes = raw_bytes

    def get(self, tag: int | str, default: str | None = None) -> str | None:
        """Retrieve a field value by tag identifier."""
        try:
            return self.fields.get(int(tag), default)
        except (ValueError, TypeError):
            return default

    def get_int(self, tag: int | str, default: int | None = None) -> int | None:
        """Retrieve a field value converted to an integer."""
        val = self.get(tag)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def __getitem__(self, tag: int | str) -> str:
        val = self.get(tag)
        if val is None:
            raise KeyError(tag)
        return val

    def __contains__(self, tag: int | str) -> bool:
        try:
            return int(tag) in self.fields
        except (ValueError, TypeError):
            return False

    def __repr__(self) -> str:
        return (
            f"<FIXMessage begin_string={self.begin_string!r} "
            f"msg_type={self.msg_type!r} fields_count={len(self.fields)}>"
        )


class FIXParser:
    """Decodes raw byte streams into structured FIXMessage instances."""

    def __init__(
        self,
        supported_begin_strings: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> None:
        if supported_begin_strings is None:
            self.supported_begin_strings = set(DEFAULT_SUPPORTED_BEGIN_STRINGS)
        else:
            self.supported_begin_strings = set(supported_begin_strings)

    def parse(self, raw_data: bytes) -> FIXMessage:
        """Parse raw FIX byte frame, validating header framing, length, and checksum."""
        if not raw_data or not isinstance(raw_data, (bytes, bytearray)):
            raise FixParseError("Raw payload must be non-empty bytes")

        if not raw_data.endswith(SOH):
            raise FixParseError("FIX message must terminate with SOH delimiter")

        raw_without_trailing = raw_data[:-1]
        last_soh = raw_without_trailing.rfind(SOH)
        if last_soh == -1:
            raise FixParseError("Malformed FIX frame: insufficient delimited fields")

        checksum_field = raw_without_trailing[last_soh + 1 :]
        if not checksum_field.startswith(b"10="):
            raise FixParseError("Missing mandatory CheckSum (tag 10) as final field")

        checksum_str = checksum_field[3:].decode("ascii", errors="replace")
        full_payload = raw_data[: last_soh + 1]
        expected_checksum = f"{sum(full_payload) % 256:03d}"

        if (
            len(checksum_str) != 3
            or not checksum_str.isdigit()
            or checksum_str != expected_checksum
        ):
            raise FixParseError(
                f"Checksum mismatch: expected {expected_checksum}, got {checksum_str}"
            )

        if not full_payload.startswith(b"8="):
            raise FixParseError("Message must start with BeginString (tag 8)")

        first_soh = full_payload.find(SOH)
        if first_soh == -1:
            raise FixParseError("Malformed BeginString (tag 8)")

        begin_string = full_payload[2:first_soh].decode("ascii", errors="replace")
        if begin_string not in self.supported_begin_strings:
            raise FixParseError(f"Unsupported BeginString: {begin_string}")

        second_field_start = first_soh + 1
        if not full_payload[second_field_start:].startswith(b"9="):
            raise FixParseError("Second field must be BodyLength (tag 9)")

        second_soh = full_payload.find(SOH, second_field_start)
        if second_soh == -1:
            raise FixParseError("Malformed BodyLength (tag 9)")

        body_length_str = full_payload[
            second_field_start + 2 : second_soh
        ].decode("ascii", errors="replace")
        try:
            body_length = int(body_length_str)
        except ValueError:
            raise FixParseError(f"BodyLength must be an integer: {body_length_str}")

        if body_length < 0:
            raise FixParseError(f"BodyLength cannot be negative: {body_length}")

        actual_body = full_payload[second_soh + 1 :]
        actual_body_length = len(actual_body)
        if actual_body_length != body_length:
            raise FixParseError(
                f"BodyLength mismatch: declared {body_length}, actual {actual_body_length}"
            )

        fields: dict[int, str] = {
            8: begin_string,
            9: str(body_length),
            10: checksum_str,
        }

        body_parts = actual_body.split(SOH)
        if body_parts and body_parts[-1] == b"":
            body_parts.pop()

        for part in body_parts:
            if not part:
                continue
            if b"=" not in part:
                raise FixParseError(f"Malformed field without '=' delimiter: {part!r}")
            tag_bytes, val_bytes = part.split(b"=", 1)
            try:
                tag_num = int(tag_bytes.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                raise FixParseError(f"Tag is not an integer: {tag_bytes!r}")

            val_str = val_bytes.decode("utf-8", errors="replace")
            fields[tag_num] = val_str

        if 35 not in fields:
            raise FixParseError("Missing mandatory MsgType (tag 35)")

        return FIXMessage(
            begin_string=begin_string,
            body_length=body_length,
            msg_type=fields[35],
            fields=fields,
            raw_bytes=raw_data,
        )