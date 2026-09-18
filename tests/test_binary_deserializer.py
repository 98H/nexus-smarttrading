"""
Unit tests for Binary Protocol (Protobuf/FlatBuffers) Market Data Deserializer.

Requirement: Story 2.1.2: Implement Binary Protocol (Protobuf/FlatBuffers) Market Data Deserializer
Target Modules:
    - src/market_data/deserializers/binary.py
    - src/market_data/deserializers/__init__.py
"""

import pytest

from src.market_data.deserializers import (
    BinaryMarketDataDeserializer,
    BinaryMarketDataSerializer,
    BinaryProtocolType,
    DeserializationError,
    Quote,
    Side,
    Trade,
)
import src.market_data.deserializers.binary as binary_module


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_trade() -> Trade:
    """Provides a canonical valid Trade message for testing."""
    return Trade(
        symbol="AAPL",
        price=150.25,
        size=100.0,
        timestamp_ns=1_672_531_199_000_000_000,
        trade_id="TRD-2023-0001",
        side=Side.BUY,
    )


@pytest.fixture
def sample_quote() -> Quote:
    """Provides a canonical valid Quote (Level 1 / BBO) message for testing."""
    return Quote(
        symbol="MSFT",
        bid_price=310.20,
        bid_size=500.0,
        ask_price=310.25,
        ask_size=300.0,
        timestamp_ns=1_672_531_200_000_000_000,
    )


@pytest.fixture
def protobuf_serializer() -> BinaryMarketDataSerializer:
    """Provides a binary serializer configured for Protobuf protocol."""
    return BinaryMarketDataSerializer(protocol=BinaryProtocolType.PROTOBUF)


@pytest.fixture
def protobuf_deserializer() -> BinaryMarketDataDeserializer:
    """Provides a binary deserializer configured for Protobuf protocol."""
    return BinaryMarketDataDeserializer(protocol=BinaryProtocolType.PROTOBUF)


@pytest.fixture
def flatbuffers_serializer() -> BinaryMarketDataSerializer:
    """Provides a binary serializer configured for FlatBuffers protocol."""
    return BinaryMarketDataSerializer(protocol=BinaryProtocolType.FLATBUFFERS)


@pytest.fixture
def flatbuffers_deserializer() -> BinaryMarketDataDeserializer:
    """Provides a binary deserializer configured for FlatBuffers protocol."""
    return BinaryMarketDataDeserializer(protocol=BinaryProtocolType.FLATBUFFERS)


# ============================================================================
# 1. Package Exports and Interface Verification
# ============================================================================

class TestBinaryModuleExports:
    """Verifies that all required classes and symbols are exported properly."""

    def test_package_exports(self):
        """Ensure src.market_data.deserializers exposes binary protocol classes."""
        import src.market_data.deserializers as pkg

        assert hasattr(pkg, "BinaryMarketDataDeserializer")
        assert hasattr(pkg, "BinaryMarketDataSerializer")
        assert hasattr(pkg, "BinaryProtocolType")
        assert hasattr(pkg, "DeserializationError")
        assert hasattr(pkg, "Trade")
        assert hasattr(pkg, "Quote")
        assert hasattr(pkg, "Side")

    def test_module_exports(self):
        """Ensure src.market_data.deserializers.binary exposes expected interface."""
        assert hasattr(binary_module, "BinaryMarketDataDeserializer")
        assert hasattr(binary_module, "BinaryMarketDataSerializer")
        assert hasattr(binary_module, "BinaryProtocolType")
        assert hasattr(binary_module, "DeserializationError")
        assert hasattr(binary_module, "Trade")
        assert hasattr(binary_module, "Quote")
        assert hasattr(binary_module, "Side")

    def test_deserialization_error_inheritance(self):
        """Ensure DeserializationError inherits from ValueError."""
        assert issubclass(DeserializationError, ValueError)


# ============================================================================
# 2. Valid Binary Payload Deserialization (Trades)
# ============================================================================

class TestValidTradeDeserialization:
    """Tests deserialization of valid binary trade payloads."""

    def test_deserialize_trade_protobuf(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Given a valid Protobuf binary trade payload, verify exact field deserialization."""
        payload = protobuf_serializer.serialize(sample_trade)
        assert isinstance(payload, bytes)
        assert len(payload) > 0

        deserialized = protobuf_deserializer.deserialize(payload)
        assert isinstance(deserialized, Trade)
        assert deserialized.symbol == sample_trade.symbol
        assert deserialized.price == pytest.approx(sample_trade.price)
        assert deserialized.size == pytest.approx(sample_trade.size)
        assert deserialized.timestamp_ns == sample_trade.timestamp_ns
        assert deserialized.trade_id == sample_trade.trade_id
        assert deserialized.side == sample_trade.side

    def test_deserialize_trade_flatbuffers(
        self,
        flatbuffers_serializer: BinaryMarketDataSerializer,
        flatbuffers_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Given a valid FlatBuffers binary trade payload, verify exact field deserialization."""
        payload = flatbuffers_serializer.serialize(sample_trade)
        assert isinstance(payload, bytes)
        assert len(payload) > 0

        deserialized = flatbuffers_deserializer.deserialize(payload)
        assert isinstance(deserialized, Trade)
        assert deserialized.symbol == sample_trade.symbol
        assert deserialized.price == pytest.approx(sample_trade.price)
        assert deserialized.size == pytest.approx(sample_trade.size)
        assert deserialized.timestamp_ns == sample_trade.timestamp_ns
        assert deserialized.trade_id == sample_trade.trade_id
        assert deserialized.side == sample_trade.side

    def test_deserialize_trade_specific_method(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Verify calling deserialize_trade explicitly returns a Trade."""
        payload = protobuf_serializer.serialize(sample_trade)
        deserialized = protobuf_deserializer.deserialize_trade(payload)

        assert isinstance(deserialized, Trade)
        assert deserialized == sample_trade

    @pytest.mark.parametrize(
        "side",
        [Side.BUY, Side.SELL],
    )
    def test_deserialize_trade_both_sides(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
        side: Side,
    ):
        """Verify that BUY and SELL orders are deserialized correctly."""
        trade = Trade(
            symbol=sample_trade.symbol,
            price=sample_trade.price,
            size=sample_trade.size,
            timestamp_ns=sample_trade.timestamp_ns,
            trade_id=sample_trade.trade_id,
            side=side,
        )
        payload = protobuf_serializer.serialize(trade)
        deserialized = protobuf_deserializer.deserialize_trade(payload)
        assert deserialized.side == side

    def test_deserialize_fractional_precision_trade(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
    ):
        """Verify double precision preservation for high-precision crypto/FX trades."""
        trade = Trade(
            symbol="BTC-USDT",
            price=43210.12345678,
            size=0.00005432,
            timestamp_ns=1_672_531_200_000_000_123,
            trade_id="CRYPTO-9999",
            side=Side.SELL,
        )
        payload = protobuf_serializer.serialize(trade)
        deserialized = protobuf_deserializer.deserialize_trade(payload)

        assert deserialized.price == pytest.approx(43210.12345678, abs=1e-8)
        assert deserialized.size == pytest.approx(0.00005432, abs=1e-8)
        assert deserialized.timestamp_ns == 1_672_531_200_000_000_123


# ============================================================================
# 3. Valid Binary Payload Deserialization (Quotes)
# ============================================================================

class TestValidQuoteDeserialization:
    """Tests deserialization of valid binary quote payloads."""

    def test_deserialize_quote_protobuf(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_quote: Quote,
    ):
        """Given a valid Protobuf binary quote payload, verify exact field deserialization."""
        payload = protobuf_serializer.serialize(sample_quote)
        assert isinstance(payload, bytes)
        assert len(payload) > 0

        deserialized = protobuf_deserializer.deserialize(payload)
        assert isinstance(deserialized, Quote)
        assert deserialized.symbol == sample_quote.symbol
        assert deserialized.bid_price == pytest.approx(sample_quote.bid_price)
        assert deserialized.bid_size == pytest.approx(sample_quote.bid_size)
        assert deserialized.ask_price == pytest.approx(sample_quote.ask_price)
        assert deserialized.ask_size == pytest.approx(sample_quote.ask_size)
        assert deserialized.timestamp_ns == sample_quote.timestamp_ns

    def test_deserialize_quote_flatbuffers(
        self,
        flatbuffers_serializer: BinaryMarketDataSerializer,
        flatbuffers_deserializer: BinaryMarketDataDeserializer,
        sample_quote: Quote,
    ):
        """Given a valid FlatBuffers binary quote payload, verify exact field deserialization."""
        payload = flatbuffers_serializer.serialize(sample_quote)
        assert isinstance(payload, bytes)
        assert len(payload) > 0

        deserialized = flatbuffers_deserializer.deserialize(payload)
        assert isinstance(deserialized, Quote)
        assert deserialized.symbol == sample_quote.symbol
        assert deserialized.bid_price == pytest.approx(sample_quote.bid_price)
        assert deserialized.bid_size == pytest.approx(sample_quote.bid_size)
        assert deserialized.ask_price == pytest.approx(sample_quote.ask_price)
        assert deserialized.ask_size == pytest.approx(sample_quote.ask_size)
        assert deserialized.timestamp_ns == sample_quote.timestamp_ns

    def test_deserialize_quote_specific_method(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_quote: Quote,
    ):
        """Verify calling deserialize_quote explicitly returns a Quote."""
        payload = protobuf_serializer.serialize(sample_quote)
        deserialized = protobuf_deserializer.deserialize_quote(payload)

        assert isinstance(deserialized, Quote)
        assert deserialized == sample_quote

    def test_deserialize_quote_zero_values(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
    ):
        """Verify handling of empty book sides (zero bid/ask price or size)."""
        quote = Quote(
            symbol="ILLIQUID",
            bid_price=0.0,
            bid_size=0.0,
            ask_price=10.0,
            ask_size=50.0,
            timestamp_ns=1_672_531_200_000_000_000,
        )
        payload = protobuf_serializer.serialize(quote)
        deserialized = protobuf_deserializer.deserialize_quote(payload)

        assert deserialized.bid_price == pytest.approx(0.0)
        assert deserialized.bid_size == pytest.approx(0.0)


# ============================================================================
# 4. Message Type Discrimination and Cross-Type Invalidation
# ============================================================================

class TestMessageTypeHandling:
    """Verifies type discrimination and strict target method parsing."""

    def test_polymorphic_deserialize_identifies_both_types(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
        sample_quote: Quote,
    ):
        """Verify deserialize() dynamically resolves Trade or Quote based on message header/tag."""
        trade_payload = protobuf_serializer.serialize(sample_trade)
        quote_payload = protobuf_serializer.serialize(sample_quote)

        assert isinstance(protobuf_deserializer.deserialize(trade_payload), Trade)
        assert isinstance(protobuf_deserializer.deserialize(quote_payload), Quote)

    def test_deserialize_trade_rejects_quote_payload(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_quote: Quote,
    ):
        """Calling deserialize_trade on a quote payload must raise DeserializationError."""
        quote_payload = protobuf_serializer.serialize(sample_quote)
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize_trade(quote_payload)

    def test_deserialize_quote_rejects_trade_payload(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Calling deserialize_quote on a trade payload must raise DeserializationError."""
        trade_payload = protobuf_serializer.serialize(sample_trade)
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize_quote(trade_payload)


# ============================================================================
# 5. Truncated Binary Payloads
# ============================================================================

class TestTruncatedPayloads:
    """Verifies that truncated binary sequences raise DeserializationError."""

    @pytest.mark.parametrize(
        "truncation_point",
        [0, 1, 2, 4, 8, 16],
    )
    def test_truncated_trade_payload_from_start(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
        truncation_point: int,
    ):
        """Given a payload sliced to very short lengths, DeserializationError must be raised."""
        full_payload = protobuf_serializer.serialize(sample_trade)
        truncated = full_payload[:truncation_point]
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(truncated)

    @pytest.mark.parametrize(
        "missing_bytes",
        [1, 2, 5, 10],
    )
    def test_truncated_trade_payload_from_end(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
        missing_bytes: int,
    ):
        """Given a payload missing 1 or more bytes from the end, DeserializationError must be raised."""
        full_payload = protobuf_serializer.serialize(sample_trade)
        truncated = full_payload[:-missing_bytes]
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(truncated)

    def test_truncated_quote_payload(
        self,
        flatbuffers_serializer: BinaryMarketDataSerializer,
        flatbuffers_deserializer: BinaryMarketDataDeserializer,
        sample_quote: Quote,
    ):
        """Given a truncated FlatBuffers quote payload, DeserializationError must be raised."""
        full_payload = flatbuffers_serializer.serialize(sample_quote)
        truncated = full_payload[: len(full_payload) // 2]
        with pytest.raises(DeserializationError):
            flatbuffers_deserializer.deserialize(truncated)


# ============================================================================
# 6. Malformed and Corrupt Binary Payloads
# ============================================================================

class TestMalformedPayloads:
    """Verifies that invalid or corrupted byte sequences raise DeserializationError."""

    def test_empty_payload_raises_error(
        self,
        protobuf_deserializer: BinaryMarketDataDeserializer,
    ):
        """Empty byte string is invalid and must raise DeserializationError."""
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(b"")

    @pytest.mark.parametrize(
        "corrupted_bytes",
        [
            b"\x00",
            b"\xFF" * 10,
            b"\xDE\xAD\xBE\xEF",
            b"GARBAGE_NOT_PROTOBUF_DATA",
            b"\x08\x96\x01\x12\x16\x07",  # Incomplete/corrupted protobuf tags
        ],
    )
    def test_random_invalid_byte_sequences(
        self,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        corrupted_bytes: bytes,
    ):
        """Arbitrary invalid byte sequences must raise DeserializationError."""
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(corrupted_bytes)

    def test_corrupted_payload_via_bit_flip(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Flipping bits in valid payload header/tag positions must cause DeserializationError."""
        payload = bytearray(protobuf_serializer.serialize(sample_trade))
        # Corrupt the first byte (message type/magic byte or proto tag)
        payload[0] ^= 0xFF
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(bytes(payload))

    def test_invalid_magic_or_unknown_message_type(
        self,
        protobuf_deserializer: BinaryMarketDataDeserializer,
    ):
        """Payload claiming an unknown message type ID must raise DeserializationError."""
        # E.g., framing header with valid magic but unsupported msg_type (0xFF)
        bogus_message = b"\x01\xFF\x00\x00\x00\x04DATA"
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(bogus_message)

    def test_corrupted_string_length_header(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Payload with an invalid length-delimited field header must raise DeserializationError."""
        payload = bytearray(protobuf_serializer.serialize(sample_trade))
        # Find where symbol AAPL starts and set length byte to an excessively large value
        aapl_idx = payload.find(b"AAPL")
        if aapl_idx > 0:
            payload[aapl_idx - 1] = 0x7F  # Claim string length is 127 when only 4 bytes follow
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(bytes(payload))


# ============================================================================
# 7. Invalid Input Types
# ============================================================================

class TestInvalidInputTypes:
    """Verifies that non-bytes inputs are rejected with TypeError."""

    @pytest.mark.parametrize(
        "invalid_input",
        [
            None,
            "AAPL,150.25,100",
            123456,
            150.25,
            {"symbol": "AAPL", "price": 150.25},
            ["\x01", "\x02"],
        ],
    )
    def test_non_bytes_payload_raises_type_error(
        self,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        invalid_input,
    ):
        """Passing non-bytes input to deserialize must raise TypeError."""
        with pytest.raises(TypeError):
            protobuf_deserializer.deserialize(invalid_input)

    def test_bytearray_input_support(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Bytearray should either be accepted and deserialized or cleanly rejected."""
        payload = protobuf_serializer.serialize(sample_trade)
        ba = bytearray(payload)
        # Verify that either bytearray works or raises TypeError consistently
        try:
            result = protobuf_deserializer.deserialize(ba)
            assert isinstance(result, Trade)
            assert result == sample_trade
        except TypeError:
            # If implementation strictly expects `bytes` only
            pass


# ============================================================================
# 8. Protocol Configuration and Initialization
# ============================================================================

class TestProtocolConfigurations:
    """Verifies initialization and protocol enforcement."""

    def test_default_protocol_is_protobuf(self):
        """Deserializer instantiated without arguments should default to Protobuf."""
        deserializer = BinaryMarketDataDeserializer()
        assert deserializer.protocol == BinaryProtocolType.PROTOBUF

    def test_explicit_flatbuffers_protocol(self):
        """Deserializer accepts explicit FLATBUFFERS protocol type."""
        deserializer = BinaryMarketDataDeserializer(protocol=BinaryProtocolType.FLATBUFFERS)
        assert deserializer.protocol == BinaryProtocolType.FLATBUFFERS

    def test_unsupported_protocol_raises_error(self):
        """Passing an invalid protocol type or string raises ValueError."""
        with pytest.raises(ValueError):
            BinaryMarketDataDeserializer(protocol="AVRO")

    def test_cross_protocol_incompatibility(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        flatbuffers_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
    ):
        """Protobuf-encoded payload passed to FlatBuffers deserializer must raise DeserializationError."""
        proto_payload = protobuf_serializer.serialize(sample_trade)
        with pytest.raises(DeserializationError):
            flatbuffers_deserializer.deserialize(proto_payload)


# ============================================================================
# 9. Deserializer Reusability and Idempotency
# ============================================================================

class TestDeserializerReusability:
    """Ensures a single deserializer instance maintains state isolation across calls."""

    def test_interleaved_valid_and_invalid_calls(
        self,
        protobuf_serializer: BinaryMarketDataSerializer,
        protobuf_deserializer: BinaryMarketDataDeserializer,
        sample_trade: Trade,
        sample_quote: Quote,
    ):
        """An error on an invalid payload must not break subsequent valid deserializations."""
        trade_payload = protobuf_serializer.serialize(sample_trade)
        quote_payload = protobuf_serializer.serialize(sample_quote)

        # 1. Valid trade
        res1 = protobuf_deserializer.deserialize(trade_payload)
        assert isinstance(res1, Trade)

        # 2. Corrupted payload raises
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(b"\xFF\xFE\xFD")

        # 3. Valid quote succeeds immediately after error
        res2 = protobuf_deserializer.deserialize(quote_payload)
        assert isinstance(res2, Quote)

        # 4. Truncated payload raises
        with pytest.raises(DeserializationError):
            protobuf_deserializer.deserialize(trade_payload[:3])

        # 5. Valid trade succeeds again
        res3 = protobuf_deserializer.deserialize(trade_payload)
        assert isinstance(res3, Trade)
        assert res3 == sample_trade