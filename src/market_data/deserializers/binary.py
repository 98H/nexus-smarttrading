"""
Binary Protocol (Protobuf / FlatBuffers) Market Data Serializer and Deserializer.
"""

from dataclasses import dataclass
from enum import Enum
import struct
from typing import Optional, Union


# ============================================================================
# Domain Models & Types
# ============================================================================

class Side(str, Enum):
    """Order / trade side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Represents an executed trade market data event."""
    symbol: str
    price: float
    size: float
    timestamp_ns: int
    trade_id: str
    side: Side


@dataclass
class Quote:
    """Represents a Level 1 / Best Bid and Offer (BBO) quote."""
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp_ns: int


class DeserializationError(ValueError):
    """Raised when deserialization of binary market data fails."""
    pass


class BinaryProtocolType(str, Enum):
    """Supported binary protocol types."""
    PROTOBUF = "PROTOBUF"
    FLATBUFFERS = "FLATBUFFERS"


# ============================================================================
# Protocol Constants
# ============================================================================

_HEADER_FORMAT = ">BBI"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # 1 byte magic + 1 byte type + 4 bytes length

_MAGIC_PROTOBUF = 0x01
_MAGIC_FLATBUFFERS = 0x02

_MSG_TRADE = 0x01
_MSG_QUOTE = 0x02


# ============================================================================
# Protobuf Codec Helpers
# ============================================================================

def _encode_varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        towrite = value & 0x7F
        value >>= 7
        if value:
            out.append(towrite | 0x80)
        else:
            out.append(towrite)
            break
    return bytes(out)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    res = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        res |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            return res, offset
        if shift > 70:
            raise DeserializationError("Varint overflow")
    raise DeserializationError("Truncated varint in payload")


class _ProtobufCodec:
    """Encodes and decodes Trade and Quote using Protobuf wire format."""

    @staticmethod
    def serialize_trade(trade: Trade) -> bytes:
        sym_bytes = trade.symbol.encode("utf-8")
        id_bytes = trade.trade_id.encode("utf-8")
        side_val = 1 if trade.side == Side.BUY else 2

        parts = [
            b"\x0A", _encode_varint(len(sym_bytes)), sym_bytes,
            b"\x11", struct.pack("<d", float(trade.price)),
            b"\x19", struct.pack("<d", float(trade.size)),
            b"\x20", _encode_varint(int(trade.timestamp_ns)),
            b"\x2A", _encode_varint(len(id_bytes)), id_bytes,
            b"\x30", _encode_varint(side_val),
        ]
        return b"".join(parts)

    @staticmethod
    def deserialize_trade(body: bytes) -> Trade:
        symbol: Optional[str] = None
        price: Optional[float] = None
        size: Optional[float] = None
        timestamp_ns: Optional[int] = None
        trade_id: Optional[str] = None
        side: Optional[Side] = None

        offset = 0
        while offset < len(body):
            tag, offset = _decode_varint(body, offset)
            field_num = tag >> 3
            wire_type = tag & 0x07

            if wire_type == 0:
                val, offset = _decode_varint(body, offset)
                if field_num == 4:
                    timestamp_ns = val
                elif field_num == 6:
                    side = Side.BUY if val == 1 else (Side.SELL if val == 2 else None)
                    if side is None:
                        raise DeserializationError(f"Invalid side enum value: {val}")
            elif wire_type == 1:
                if offset + 8 > len(body):
                    raise DeserializationError("Truncated 64-bit field")
                val = struct.unpack_from("<d", body, offset)[0]
                offset += 8
                if field_num == 2:
                    price = val
                elif field_num == 3:
                    size = val
            elif wire_type == 2:
                length, offset = _decode_varint(body, offset)
                if offset + length > len(body):
                    raise DeserializationError("Truncated length-delimited field")
                raw_bytes = body[offset : offset + length]
                offset += length
                try:
                    str_val = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as e:
                    raise DeserializationError("Malformed UTF-8 string") from e

                if field_num == 1:
                    symbol = str_val
                elif field_num == 5:
                    trade_id = str_val
                elif field_num == 6:
                    if str_val in ("BUY", "Side.BUY"):
                        side = Side.BUY
                    elif str_val in ("SELL", "Side.SELL"):
                        side = Side.SELL
                    else:
                        raise DeserializationError(f"Invalid side string: {str_val}")
            elif wire_type == 5:
                if offset + 4 > len(body):
                    raise DeserializationError("Truncated 32-bit field")
                offset += 4
            else:
                raise DeserializationError(f"Unsupported wire type: {wire_type}")

        if (
            symbol is None
            or price is None
            or size is None
            or timestamp_ns is None
            or trade_id is None
            or side is None
        ):
            raise DeserializationError("Incomplete Trade message payload")

        return Trade(
            symbol=symbol,
            price=price,
            size=size,
            timestamp_ns=timestamp_ns,
            trade_id=trade_id,
            side=side,
        )

    @staticmethod
    def serialize_quote(quote: Quote) -> bytes:
        sym_bytes = quote.symbol.encode("utf-8")
        parts = [
            b"\x0A", _encode_varint(len(sym_bytes)), sym_bytes,
            b"\x11", struct.pack("<d", float(quote.bid_price)),
            b"\x19", struct.pack("<d", float(quote.bid_size)),
            b"\x21", struct.pack("<d", float(quote.ask_price)),
            b"\x29", struct.pack("<d", float(quote.ask_size)),
            b"\x30", _encode_varint(int(quote.timestamp_ns)),
        ]
        return b"".join(parts)

    @staticmethod
    def deserialize_quote(body: bytes) -> Quote:
        symbol: Optional[str] = None
        bid_price: Optional[float] = None
        bid_size: Optional[float] = None
        ask_price: Optional[float] = None
        ask_size: Optional[float] = None
        timestamp_ns: Optional[int] = None

        offset = 0
        while offset < len(body):
            tag, offset = _decode_varint(body, offset)
            field_num = tag >> 3
            wire_type = tag & 0x07

            if wire_type == 0:
                val, offset = _decode_varint(body, offset)
                if field_num == 6:
                    timestamp_ns = val
            elif wire_type == 1:
                if offset + 8 > len(body):
                    raise DeserializationError("Truncated 64-bit field")
                val = struct.unpack_from("<d", body, offset)[0]
                offset += 8
                if field_num == 2:
                    bid_price = val
                elif field_num == 3:
                    bid_size = val
                elif field_num == 4:
                    ask_price = val
                elif field_num == 5:
                    ask_size = val
            elif wire_type == 2:
                length, offset = _decode_varint(body, offset)
                if offset + length > len(body):
                    raise DeserializationError("Truncated length-delimited field")
                raw_bytes = body[offset : offset + length]
                offset += length
                try:
                    symbol = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as e:
                    raise DeserializationError("Malformed UTF-8 string") from e
            elif wire_type == 5:
                if offset + 4 > len(body):
                    raise DeserializationError("Truncated 32-bit field")
                offset += 4
            else:
                raise DeserializationError(f"Unsupported wire type: {wire_type}")

        if (
            symbol is None
            or bid_price is None
            or bid_size is None
            or ask_price is None
            or ask_size is None
            or timestamp_ns is None
        ):
            raise DeserializationError("Incomplete Quote message payload")

        return Quote(
            symbol=symbol,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            timestamp_ns=timestamp_ns,
        )


# ============================================================================
# FlatBuffers Codec Helpers
# ============================================================================

class _FlatBuffersCodec:
    """Encodes and decodes Trade and Quote using standard FlatBuffers table format."""

    @staticmethod
    def serialize_trade(trade: Trade) -> bytes:
        vtable_size = 4 + 2 * 6  # 16 bytes
        table_pos = 4 + vtable_size  # 20 bytes
        field_offsets = [4, 8, 16, 24, 32, 36]
        table_size = 37

        vtable = struct.pack(
            "<HH6H",
            vtable_size,
            table_size,
            *field_offsets,
        )

        sym_bytes = trade.symbol.encode("utf-8")
        id_bytes = trade.trade_id.encode("utf-8")
        side_val = 1 if trade.side == Side.BUY else 2

        str1_pos = table_pos + table_size
        str1_offset = str1_pos - (table_pos + 4)
        str1_data = struct.pack("<I", len(sym_bytes)) + sym_bytes + b"\x00"

        str2_pos = str1_pos + len(str1_data)
        str2_offset = str2_pos - (table_pos + 32)
        str2_data = struct.pack("<I", len(id_bytes)) + id_bytes + b"\x00"

        table_header = struct.pack(
            "<iIddqIB",
            table_pos - 4,
            str1_offset,
            float(trade.price),
            float(trade.size),
            int(trade.timestamp_ns),
            str2_offset,
            side_val,
        )

        return struct.pack("<I", table_pos) + vtable + table_header + str1_data + str2_data

    @staticmethod
    def deserialize_trade(body: bytes) -> Trade:
        if len(body) < 8:
            raise DeserializationError("FlatBuffers payload too short")

        root_pos = struct.unpack_from("<I", body, 0)[0]
        if root_pos + 4 > len(body):
            raise DeserializationError("Root table offset out of bounds")

        vtable_delta = struct.unpack_from("<i", body, root_pos)[0]
        vtable_pos = root_pos - vtable_delta
        if vtable_pos < 0 or vtable_pos + 4 > len(body):
            raise DeserializationError("Vtable offset out of bounds")

        vtable_len, table_len = struct.unpack_from("<HH", body, vtable_pos)
        if vtable_pos + vtable_len > len(body) or root_pos + table_len > len(body):
            raise DeserializationError("FlatBuffers table truncated")

        def _get_field(idx: int, size: int) -> int:
            ventry = vtable_pos + 4 + 2 * idx
            if ventry + 2 > vtable_pos + vtable_len:
                raise DeserializationError("Missing vtable field offset")
            foffset = struct.unpack_from("<H", body, ventry)[0]
            if foffset == 0:
                raise DeserializationError(f"Field {idx} not present in table")
            fpos = root_pos + foffset
            if fpos + size > root_pos + table_len or fpos + size > len(body):
                raise DeserializationError("Field extends past buffer boundary")
            return fpos

        def _read_str(idx: int) -> str:
            fpos = _get_field(idx, 4)
            rel = struct.unpack_from("<I", body, fpos)[0]
            spos = fpos + rel
            if spos + 4 > len(body):
                raise DeserializationError("String offset out of bounds")
            slen = struct.unpack_from("<I", body, spos)[0]
            if spos + 4 + slen > len(body):
                raise DeserializationError("String length out of bounds")
            try:
                return body[spos + 4 : spos + 4 + slen].decode("utf-8")
            except UnicodeDecodeError as e:
                raise DeserializationError("Malformed UTF-8 string") from e

        symbol = _read_str(0)
        price = struct.unpack_from("<d", body, _get_field(1, 8))[0]
        size = struct.unpack_from("<d", body, _get_field(2, 8))[0]
        timestamp_ns = struct.unpack_from("<q", body, _get_field(3, 8))[0]
        trade_id = _read_str(4)
        side_val = struct.unpack_from("<B", body, _get_field(5, 1))[0]
        side = Side.BUY if side_val == 1 else (Side.SELL if side_val == 2 else None)
        if side is None:
            raise DeserializationError(f"Invalid side enum: {side_val}")

        return Trade(
            symbol=symbol,
            price=price,
            size=size,
            timestamp_ns=timestamp_ns,
            trade_id=trade_id,
            side=side,
        )

    @staticmethod
    def serialize_quote(quote: Quote) -> bytes:
        vtable_size = 4 + 2 * 6  # 16 bytes
        table_pos = 4 + vtable_size  # 20 bytes
        field_offsets = [4, 8, 16, 24, 32, 40]
        table_size = 48

        vtable = struct.pack(
            "<HH6H",
            vtable_size,
            table_size,
            *field_offsets,
        )

        sym_bytes = quote.symbol.encode("utf-8")
        str_pos = table_pos + table_size
        str_offset = str_pos - (table_pos + 4)
        str_data = struct.pack("<I", len(sym_bytes)) + sym_bytes + b"\x00"

        table_header = struct.pack(
            "<iIddddq",
            table_pos - 4,
            str_offset,
            float(quote.bid_price),
            float(quote.bid_size),
            float(quote.ask_price),
            float(quote.ask_size),
            int(quote.timestamp_ns),
        )

        return struct.pack("<I", table_pos) + vtable + table_header + str_data

    @staticmethod
    def deserialize_quote(body: bytes) -> Quote:
        if len(body) < 8:
            raise DeserializationError("FlatBuffers payload too short")

        root_pos = struct.unpack_from("<I", body, 0)[0]
        if root_pos + 4 > len(body):
            raise DeserializationError("Root table offset out of bounds")

        vtable_delta = struct.unpack_from("<i", body, root_pos)[0]
        vtable_pos = root_pos - vtable_delta
        if vtable_pos < 0 or vtable_pos + 4 > len(body):
            raise DeserializationError("Vtable offset out of bounds")

        vtable_len, table_len = struct.unpack_from("<HH", body, vtable_pos)
        if vtable_pos + vtable_len > len(body) or root_pos + table_len > len(body):
            raise DeserializationError("FlatBuffers table truncated")

        def _get_field(idx: int, size: int) -> int:
            ventry = vtable_pos + 4 + 2 * idx
            if ventry + 2 > vtable_pos + vtable_len:
                raise DeserializationError("Missing vtable field offset")
            foffset = struct.unpack_from("<H", body, ventry)[0]
            if foffset == 0:
                raise DeserializationError(f"Field {idx} not present in table")
            fpos = root_pos + foffset
            if fpos + size > root_pos + table_len or fpos + size > len(body):
                raise DeserializationError("Field extends past buffer boundary")
            return fpos

        def _read_str(idx: int) -> str:
            fpos = _get_field(idx, 4)
            rel = struct.unpack_from("<I", body, fpos)[0]
            spos = fpos + rel
            if spos + 4 > len(body):
                raise DeserializationError("String offset out of bounds")
            slen = struct.unpack_from("<I", body, spos)[0]
            if spos + 4 + slen > len(body):
                raise DeserializationError("String length out of bounds")
            try:
                return body[spos + 4 : spos + 4 + slen].decode("utf-8")
            except UnicodeDecodeError as e:
                raise DeserializationError("Malformed UTF-8 string") from e

        symbol = _read_str(0)
        bid_price = struct.unpack_from("<d", body, _get_field(1, 8))[0]
        bid_size = struct.unpack_from("<d", body, _get_field(2, 8))[0]
        ask_price = struct.unpack_from("<d", body, _get_field(3, 8))[0]
        ask_size = struct.unpack_from("<d", body, _get_field(4, 8))[0]
        timestamp_ns = struct.unpack_from("<q", body, _get_field(5, 8))[0]

        return Quote(
            symbol=symbol,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            timestamp_ns=timestamp_ns,
        )


# ============================================================================
# Serializer & Deserializer
# ============================================================================

def _normalize_protocol(protocol: Union[BinaryProtocolType, str]) -> BinaryProtocolType:
    if isinstance(protocol, BinaryProtocolType):
        return protocol
    if isinstance(protocol, str):
        try:
            return BinaryProtocolType[protocol.upper()]
        except KeyError:
            try:
                return BinaryProtocolType(protocol)
            except ValueError:
                raise ValueError(f"Unsupported binary protocol: {protocol}")
    raise ValueError(f"Invalid protocol type: {type(protocol).__name__}")


class BinaryMarketDataSerializer:
    """Serializes Trade and Quote market data into framed binary protocol payloads."""

    def __init__(self, protocol: Union[BinaryProtocolType, str] = BinaryProtocolType.PROTOBUF):
        self.protocol = _normalize_protocol(protocol)
        self._magic = _MAGIC_PROTOBUF if self.protocol == BinaryProtocolType.PROTOBUF else _MAGIC_FLATBUFFERS

    def serialize_trade(self, trade: Trade) -> bytes:
        if not isinstance(trade, Trade):
            raise TypeError(f"Expected Trade instance, got {type(trade).__name__}")
        if self.protocol == BinaryProtocolType.PROTOBUF:
            body = _ProtobufCodec.serialize_trade(trade)
        else:
            body = _FlatBuffersCodec.serialize_trade(trade)

        header = struct.pack(_HEADER_FORMAT, self._magic, _MSG_TRADE, len(body))
        return header + body

    def serialize_quote(self, quote: Quote) -> bytes:
        if not isinstance(quote, Quote):
            raise TypeError(f"Expected Quote instance, got {type(quote).__name__}")
        if self.protocol == BinaryProtocolType.PROTOBUF:
            body = _ProtobufCodec.serialize_quote(quote)
        else:
            body = _FlatBuffersCodec.serialize_quote(quote)

        header = struct.pack(_HEADER_FORMAT, self._magic, _MSG_QUOTE, len(body))
        return header + body

    def serialize(self, message: Union[Trade, Quote]) -> bytes:
        if isinstance(message, Trade):
            return self.serialize_trade(message)
        if isinstance(message, Quote):
            return self.serialize_quote(message)
        raise TypeError(f"Unsupported message type: {type(message).__name__}")


class BinaryMarketDataDeserializer:
    """Deserializes framed binary protocol payloads into Trade or Quote market data."""

    def __init__(self, protocol: Union[BinaryProtocolType, str] = BinaryProtocolType.PROTOBUF):
        self.protocol = _normalize_protocol(protocol)
        self._magic = _MAGIC_PROTOBUF if self.protocol == BinaryProtocolType.PROTOBUF else _MAGIC_FLATBUFFERS

    def _parse_and_validate(self, payload: Union[bytes, bytearray]) -> tuple[int, bytes]:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"Expected bytes or bytearray, got {type(payload).__name__}")

        raw = bytes(payload)
        if len(raw) < _HEADER_SIZE:
            raise DeserializationError(
                f"Truncated payload: expected at least {_HEADER_SIZE} bytes for header, got {len(raw)}"
            )

        magic, msg_type, body_len = struct.unpack_from(_HEADER_FORMAT, raw, 0)
        if magic != self._magic:
            raise DeserializationError(
                f"Protocol magic mismatch: expected {self._magic:#04x}, got {magic:#04x}"
            )

        if msg_type not in (_MSG_TRADE, _MSG_QUOTE):
            raise DeserializationError(f"Unsupported message type: {msg_type:#04x}")

        if len(raw) != _HEADER_SIZE + body_len:
            raise DeserializationError(
                f"Payload size mismatch: expected {_HEADER_SIZE + body_len} bytes, got {len(raw)}"
            )

        return msg_type, raw[_HEADER_SIZE:]

    def deserialize_trade(self, payload: Union[bytes, bytearray]) -> Trade:
        msg_type, body = self._parse_and_validate(payload)
        if msg_type != _MSG_TRADE:
            raise DeserializationError(f"Expected Trade payload, received message type: {msg_type:#04x}")

        if self.protocol == BinaryProtocolType.PROTOBUF:
            return _ProtobufCodec.deserialize_trade(body)
        return _FlatBuffersCodec.deserialize_trade(body)

    def deserialize_quote(self, payload: Union[bytes, bytearray]) -> Quote:
        msg_type, body = self._parse_and_validate(payload)
        if msg_type != _MSG_QUOTE:
            raise DeserializationError(f"Expected Quote payload, received message type: {msg_type:#04x}")

        if self.protocol == BinaryProtocolType.PROTOBUF:
            return _ProtobufCodec.deserialize_quote(body)
        return _FlatBuffersCodec.deserialize_quote(body)

    def deserialize(self, payload: Union[bytes, bytearray]) -> Union[Trade, Quote]:
        msg_type, body = self._parse_and_validate(payload)
        if msg_type == _MSG_TRADE:
            if self.protocol == BinaryProtocolType.PROTOBUF:
                return _ProtobufCodec.deserialize_trade(body)
            return _FlatBuffersCodec.deserialize_trade(body)
        elif msg_type == _MSG_QUOTE:
            if self.protocol == BinaryProtocolType.PROTOBUF:
                return _ProtobufCodec.deserialize_quote(body)
            return _FlatBuffersCodec.deserialize_quote(body)
        raise DeserializationError(f"Unsupported message type: {msg_type:#04x}")