import json
import logging
import os
import httpx
import asyncio

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

    # Modify the event handler to be synchronous and create an async task inside it
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant):
        logger.info("Participant connected: %s", participant.identity)
        
        # Skip if this is the agent or a system participant
        if (participant.identity.startswith("agent-") or 
            participant.identity == "transcript-listener"):
            return
        
        # If this is a user, send a welcome message
        logger.info("User participant detected: %s", participant.identity)
        
        # Create an async task to handle the welcome message
        asyncio.create_task(handle_new_participant(participant, session))

    async def handle_new_participant(participant, session):
        """Handle a new participant connection asynchronously."""
        # Wait a moment for connection to stabilize
        await asyncio.sleep(2)
        
        # Generate and send welcome message
        await send_welcome_message(session)

    async def send_welcome_message(session):
        """Send a welcome message using the agent session."""
        logger.info("Sending welcome message")
        
        try:
            # Use the session's say method to generate and speak a message
            session.say(
                "Welcome to HeirloomStories! I'm your AI assistant. How can I help you today?"
            )
        except Exception as e:
            logger.error("Error sending welcome message: %s", e)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info("Starting voice assistant for participant: %s", participant.identity)

    # Create an Agent with instructions
    agent = Agent(
        instructions="You are HeirloomStories AI, a helpful AI voice assistant that engages in conversation with users. Be friendly, responsive, and helpful. Your purpose is to assist users with their questions and engage in natural conversation.",
        llm=openai.LLM(
            api_key=f"{agent_id}-livekit",
            base_url=agent_url,
            timeout=60.0,
        ),

    )

    # Create an AgentSession with the components
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
    )

    # Start the session with the agent and room
    await session.start(agent, room=ctx.room)

    # Send welcome message if this is the first user
    if not participant.identity.startswith("agent-"):
        await send_welcome_message(session)
    

async def send_transcript_to_app(speaker, text):
    """Send transcript to the FastHTML app."""
    # Get the app URL from environment variable, with a default fallback
    app_url = os.environ.get("FASTHTML_APP_URL", "http://localhost:5001")
    
    print(f"[PIPELINE] Sending transcript to {app_url}/api/transcript")
    print(f"[PIPELINE] Speaker: {speaker}, Text: {text}")
    
    try:
        # Send HTTP POST request to your FastHTML app
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{app_url}/api/transcript",
                json={
                    "speaker": speaker,
                    "text": text
                },
                timeout=10.0  # Add a timeout to avoid hanging
            )
            
            print(f"[PIPELINE] Response status: {response.status_code}")
            print(f"[PIPELINE] Response body: {response.text}")
            
    except Exception as e:
        print(f"[PIPELINE] Error sending transcript to app: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="AgentVoice",
            prewarm_fnc=prewarm,
        )
    )
