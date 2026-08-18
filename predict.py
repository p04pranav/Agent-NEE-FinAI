"""
Agent-NEE FinAI — Prediction Engine
Ollama API wrapper + PredictionSquad multi-agent orchestration.
"""

import json
import time
import logging

import requests

import config
import utils
from agents import band, AGENT_ROLES, SYNTHESIS_PROMPT

logger = logging.getLogger("agent_nee")


def ollama_generate(
    prompt: str,
    temperature: float = None,
    max_tokens: int = None,
    format_json: bool = False,
    grammar: str = None,
) -> str:
    """Call Ollama API for text generation.
    
    Fully stateless: no context array passed (fresh KV cache per call).
    Model stays warm in memory (no keep_alive: 0).
    Supports format="json" or explicit grammar (GBNF).
    """
    temperature = temperature or config.INFERENCE_TEMPERATURE
    max_tokens = max_tokens or config.INFERENCE_MAX_TOKENS

    payload = {
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": config.INFERENCE_TOP_P,
            "num_predict": max_tokens,
            "num_ctx": config.N_CTX,
        },
    }

    if format_json:
        payload["format"] = "json"
    elif grammar:
        payload["grammar"] = grammar

    try:
        response = requests.post(
            config.OLLAMA_API_GENERATE,
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.ConnectionError as e:
        raise utils.OllamaError("Ollama not running. Start with: ollama serve") from e
    except requests.exceptions.Timeout as e:
        raise utils.OllamaError(f"Ollama request timed out after {config.OLLAMA_TIMEOUT}s") from e
    except Exception as e:
        raise utils.OllamaError(f"Ollama API error: {e}") from e


def verify_ollama() -> bool:
    """Verify Ollama is running and the configured model is available.
    Returns True if ready, raises OllamaError if not.
    """
    try:
        response = requests.get(config.OLLAMA_API_TAGS, timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        # Check if model is available (exact match or prefix match)
        found = any(
            config.MODEL_NAME in name or name.startswith(config.MODEL_NAME.split(":")[0])
            for name in model_names
        )

        if not found:
            logger.warning(
                f"Model '{config.MODEL_NAME}' not found. "
                f"Available: {model_names}. "
                f"Run: ollama pull {config.MODEL_NAME}"
            )
            return False

        logger.info(f"Ollama ready. Model: {config.MODEL_NAME}")
        return True

    except requests.exceptions.ConnectionError:
        raise utils.OllamaError(
            "Ollama is not running. Start it with: ollama serve"
        )


class PredictionSquad:
    """Runs all agents sequentially on the same Ollama model, then synthesizes."""

    def __init__(self, band_sdk=None):
        self.band = band_sdk or band

    def run(self, ticker: str, ticker_data: str, room_id: str) -> dict:
        """Execute full multi-agent prediction cycle for one ticker.
        
        Args:
            ticker: NSE ticker symbol
            ticker_data: Formatted market data string for agents
            room_id: LocalBandSDK room ID
            
        Returns:
            {"direction": str, "target_return_pct": float, "confidence": str}
        """
        agent_analyses = []
        total_latency = 0

        # Run each specialist agent
        for role_key in config.AGENT_ROLES_TO_RUN:
            role = AGENT_ROLES[role_key]
            prompt = f"{role['prompt']}\n\n{ticker_data}"

            start = time.time()
            try:
                response = ollama_generate(
                    prompt=prompt,
                    temperature=role["temperature"],
                    max_tokens=config.AGENT_MAX_TOKENS,
                )
                latency_ms = (time.time() - start) * 1000
                total_latency += latency_ms

                self.band.send_message(room_id, role["name"], response)
                agent_analyses.append(f"--- {role['name']} ---\n{response}")

                logger.debug(
                    f"{ticker} {role['prefix']}: {latency_ms:.0f}ms"
                )

            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                logger.warning(f"{ticker} {role['prefix']} failed: {e}")
                agent_analyses.append(f"--- {role['name']} --- [FAILED: {e}]")

        # Synthesize
        synth_prompt = SYNTHESIS_PROMPT.format(
            agent_analyses="\n\n".join(agent_analyses)
        )

        start = time.time()
        try:
            synth_response = ollama_generate(
                prompt=synth_prompt,
                temperature=config.SYNTHESIS_TEMPERATURE,
                max_tokens=config.SYNTH_MAX_TOKENS,
                format_json=True,
            )
            latency_ms = (time.time() - start) * 1000
            total_latency += latency_ms

            self.band.send_message(room_id, "Synthesizer", synth_response, msg_type="synthesis")

            # Parse JSON response
            prediction = json.loads(synth_response)

            # Validate keys
            result = {
                "direction": prediction.get("direction", "SIDEWAYS").upper(),
                "target_return_pct": float(prediction.get("target_return_pct", 0.0)),
                "confidence": prediction.get("confidence", "LOW").upper(),
            }

            # Clamp direction
            if result["direction"] not in ("UP", "DOWN", "SIDEWAYS"):
                result["direction"] = "SIDEWAYS"

            # Clamp confidence
            if result["confidence"] not in ("LOW", "MED", "HIGH"):
                result["confidence"] = "LOW"

            result["inference_latency_ms"] = round(total_latency, 1)
            result["agent_contributions"] = self.band.get_room_history(room_id)

            logger.info(
                f"{ticker}: {result['direction']} {result['target_return_pct']:+.2f}% "
                f"{result['confidence']} ({total_latency:.0f}ms)"
            )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"{ticker} Synthesizer returned invalid JSON: {e}")
            # Retry with lower temperature
            try:
                synth_response = ollama_generate(
                    prompt=synth_prompt,
                    temperature=0.05,
                    max_tokens=config.SYNTH_MAX_TOKENS,
                    format_json=True,
                )
                prediction = json.loads(synth_response)
                result = {
                    "direction": prediction.get("direction", "SIDEWAYS").upper(),
                    "target_return_pct": float(prediction.get("target_return_pct", 0.0)),
                    "confidence": prediction.get("confidence", "LOW").upper(),
                    "inference_latency_ms": round(total_latency, 1),
                    "agent_contributions": self.band.get_room_history(room_id),
                }
                return result
            except Exception:
                logger.error(f"{ticker} Synthesizer retry failed. Returning default.")
                return {
                    "direction": "SIDEWAYS",
                    "target_return_pct": 0.0,
                    "confidence": "LOW",
                    "inference_latency_ms": round(total_latency, 1),
                    "agent_contributions": self.band.get_room_history(room_id),
                }

        except Exception as e:
            logger.error(f"{ticker} Synthesizer failed: {e}")
            return {
                "direction": "SIDEWAYS",
                "target_return_pct": 0.0,
                "confidence": "LOW",
                "inference_latency_ms": round(total_latency, 1),
                "agent_contributions": self.band.get_room_history(room_id),
            }
