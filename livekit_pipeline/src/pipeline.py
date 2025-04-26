import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, elevenlabs, openai, silero

# from livekit.plugins.openai.llm import LLMRetryOptions

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_envs() -> None:
    """Check for the presence of all required environment variables."""
    required_envs = {
        "LIVEKIT_URL": "LiveKit server URL",
        "LIVEKIT_API_KEY": "API Key for LiveKit",
        "LIVEKIT_API_SECRET": "API Secret for LiveKit",
        "DEEPGRAM_API_KEY": "API key for Deepgram (used for STT)",
        "ELEVEN_API_KEY": "API key for ElevenLabs (used for TTS)",
    }
    for key, description in required_envs.items():
        if not os.environ.get(key):
            logger.warning("Environment variable %s (%s) is not set.", key, description)

# Validate environments at module load
validate_envs()

def prewarm(proc: JobProcess) -> None:
    logger.info("Prewarming: loading VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded successfully.")

async def entrypoint(ctx: JobContext) -> None:
    metadata = ctx.job.metadata

    logger.info("job metadata: %s", metadata)

    if isinstance(metadata, str):
        try:
            metadata_obj = json.loads(metadata)
        except json.JSONDecodeError:
            try:
                normalized = metadata.replace("'", '"')
                metadata_obj = json.loads(normalized)
            except json.JSONDecodeError as norm_error:
                logger.warning(
                    "Normalization failed, using default values: %s", norm_error
                )
                metadata_obj = {}
    else:
        metadata_obj = metadata

    logger.info("metadata_obj: %s", metadata_obj)

    agent_name = metadata_obj.get("agent_name")
    agent_id = metadata_obj.get("agent_id")
    run_id = metadata_obj.get("run_id")

    # Retrieve the Host from environment variables.
    engine_api_address = os.environ.get("RESTACK_ENGINE_API_ADDRESS")
    if not engine_api_address:
        agent_backend_host = "http://localhost:9233"
    elif not engine_api_address.startswith("https://"):
        agent_backend_host = "https://" + engine_api_address
    else:
        agent_backend_host = engine_api_address

    logger.info("Using RESTACK_ENGINE_API_ADDRESS: %s", agent_backend_host)

    agent_url = f"{agent_backend_host}/stream/agents/{agent_name}/{agent_id}/{run_id}"
    logger.info("Agent URL: %s", agent_url)

    logger.info("Connecting to room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info("Starting voice assistant for participant: %s", participant.identity)

    # Create an Agent with instructions
    agent = Agent(
        instructions="You are HeirloomStories AI, a helpful assistant that engages in conversation with users. Be friendly, responsive, and helpful. Your purpose is to assist users with their questions and engage in natural conversation.",
        llm=openai.LLM(
            api_key=f"{agent_id}-livekit",
            base_url=agent_url,
            timeout=60.0,
        ),
        # llm=openai.LLM(
        #     api_key=f"{agent_id}-livekit",
        #     base_url=agent_url,
        #     retry_options=LLMRetryOptions(
        #         max_retries=5,  # Increase from default 4
        #         initial_backoff=1.0,  # Start with 1 second
        #         max_backoff=30.0,  # Maximum 30 seconds between retries
        #     )
        # ),
    )

    # Create an AgentSession with the components
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
    )

    # Start the session with the agent and room
    await session.start(agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="AgentVoice",
            prewarm_fnc=prewarm,
        )
    )
