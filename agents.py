"""
Agent-NEE FinAI — Agent System
LocalBandSDK for agent coordination + predefined agent roles + synthesis prompt.
"""

import logging

import config


logger = logging.getLogger("agent_nee")


# ─── LocalBandSDK — In-Memory Agent Coordination ─────────────────────

class LocalBandSDK:
    """In-memory room/message simulation for agent coordination."""

    def __init__(self):
        self.rooms: dict[str, list] = {}
        self.room_counter = 0

    def create_room(self, name: str) -> str:
        """Create a new room. Returns room_id (e.g., 'room_1')."""
        self.room_counter += 1
        room_id = f"room_{self.room_counter}"
        self.rooms[room_id] = []
        logger.debug(f"Created room {room_id}: {name}")
        return room_id

    def send_message(self, room_id: str, sender: str, text: str, msg_type: str = "contribution"):
        """Send a message to a room.
        msg_type: "contribution" | "synthesis" | "system"
        """
        if room_id not in self.rooms:
            logger.warning(f"Room {room_id} not found")
            return

        turn = len(self.rooms[room_id])
        message = {
            "sender": sender,
            "text": text,
            "type": msg_type,
            "turn": turn,
        }
        self.rooms[room_id].append(message)
        logger.debug(f"Room {room_id}: {sender} sent {msg_type} (turn {turn})")

    def get_room_history(self, room_id: str) -> list:
        """Return all messages in chronological order."""
        return self.rooms.get(room_id, [])

    def get_history_formatted(self, room_id: str, max_chars: int = 3000) -> str:
        """Return formatted history string for logging/display."""
        messages = self.get_room_history(room_id)
        lines = []
        for msg in messages:
            prefix = f"[{msg['sender']}]"
            lines.append(f"{prefix} {msg['text']}")
        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars] + "...[truncated]"
        return result

    def cleanup_room(self, room_id: str):
        """Remove a room to free memory."""
        if room_id in self.rooms:
            del self.rooms[room_id]
        # Auto-cleanup old rooms to prevent memory leak
        self.cleanup_old_rooms(max_rooms=100)

    def cleanup_old_rooms(self, max_rooms: int = 100):
        """Remove oldest rooms if count exceeds max_rooms."""
        while len(self.rooms) > max_rooms:
            oldest = min(self.rooms.keys())
            del self.rooms[oldest]


# Singleton instance
band = LocalBandSDK()


# ─── Agent Roles ─────────────────────────────────────────────────────

AGENT_ROLES = {
    "technical_analyst": {
        "name": "Technical Analyst",
        "prefix": "TECH",
        "prompt": """You are a technical analysis specialist for NSE equities.
Analyze the price action, VWAP, RSI, MACD, Bollinger Bands, and ATR.
Given the market data below, predict:
1. Direction (UP/DOWN/SIDEWAYS)
2. Target return percentage
3. Confidence level (LOW/MED/HIGH)
Cite specific indicator values. Max 200 words.""",
        "temperature": config.INFERENCE_TEMPERATURE,
    },
    "volatility_analyst": {
        "name": "Volatility Analyst",
        "prefix": "VOL",
        "prompt": """You are a volatility and risk specialist for NSE equities.
Focus on ATR, Bollinger Band width, and recent volatility patterns.
Given the market data below, produce:
1. Volatility regime (LOW/MED/HIGH)
2. Risk-adjusted confidence score
3. Any anomaly or squeeze signals
Cite specific volatility metrics. Max 200 words.""",
        "temperature": config.INFERENCE_TEMPERATURE,
    },
    "volume_analyst": {
        "name": "Volume Analyst",
        "prefix": "VOLM",
        "prompt": """You are a volume and liquidity analyst for NSE equities.
Analyze volume trends, volume vs VWAP, and volume spike patterns.
Given the market data below, assess:
1. Conviction behind current price move (STRONG/WEAK/NEUTRAL)
2. Any divergence between price and volume
3. Liquidity conditions for the predicted move
Cite specific volume figures. Max 200 words.""",
        "temperature": config.INFERENCE_TEMPERATURE,
    },
}

# Synthesis prompt template
SYNTHESIS_PROMPT = """You are the Agent-NEE Prediction Synthesizer.
Below are analyses from multiple specialist agents for the same ticker.
Merge them into a single final prediction.

Agent Analyses:
{agent_analyses}

Output ONLY a valid JSON object with these exact keys:
{{"direction": "UP|DOWN|SIDEWAYS", "target_return_pct": <float>, "confidence": "LOW|MED|HIGH"}}
No markdown, no explanation, no extra text."""
