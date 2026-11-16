"""Reference: a minimal inbound telephony agent for a Vietnamese hotline.

NOT run in CI and not imported by the tests. It needs LiveKit Cloud credentials and:

    uv add "livekit-agents[silero]~=1.8" livekit-plugins-krisp python-dotenv

API forms follow the LiveKit docs for livekit-agents 1.8 (checked 2026-11-16):
- quickstart skeleton: https://docs.livekit.io/agents/start/voice-ai
- turn handling:       https://docs.livekit.io/reference/agents/turn-handling-options
- noise cancellation:  https://docs.livekit.io/transport/media/noise-cancellation
- SIP attributes:      https://docs.livekit.io/reference/telephony/sip-participant
- hanging up:          https://docs.livekit.io/agents/server/job

Run in development mode with:  python livekit_agent.py dev
Then point a dispatch rule with agent_name="hotline-agent" at your inbound trunk (lesson 01).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    function_tool,
    get_job_context,
    room_io,
)
from livekit.plugins import krisp

load_dotenv(".env.local")
logger = logging.getLogger("hotline-agent")

# A LiveKit Inference TTS model id that has a Vietnamese voice. Pick one from
# https://docs.livekit.io/agents/models/tts and set it in .env.local; there is no safe default.
TTS_MODEL = os.environ["AICC_TTS_MODEL"]


class HotlineAgent(Agent):
    def __init__(self, caller_number: str | None) -> None:
        super().__init__(
            instructions=(
                "Bạn là tổng đài viên AI của một công ty giao hàng. Trả lời ngắn gọn, lịch sự, bằng tiếng Việt. "
                "Không dùng ký hiệu, gạch đầu dòng hay emoji vì câu trả lời sẽ được đọc qua điện thoại. "
                f"Số điện thoại của người gọi là {caller_number or 'không rõ'}."
            ),
        )

    @function_tool
    async def end_call(self, ctx: RunContext) -> None:
        """Called when the caller says goodbye or asks to end the call."""
        await ctx.wait_for_playout()  # let the goodbye finish playing
        job_ctx = get_job_context()
        if job_ctx is not None:
            await job_ctx.delete_room()  # disconnects everyone, including the SIP caller


server = AgentServer()


@server.rtc_session(agent_name="hotline-agent")
async def hotline(ctx: JobContext) -> None:
    participant = await ctx.wait_for_participant()
    caller = None
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        caller = participant.attributes.get("sip.phoneNumber")
        logger.info("call %s from %s", participant.attributes.get("sip.callID"), caller)

    session = AgentSession(
        stt="deepgram/nova-3:vi",            # LiveKit Inference model:language string
        llm="openai/gpt-4.1-mini",
        tts=TTS_MODEL,
        turn_handling=TurnHandlingOptions(
            # The audio turn detector does not cover Vietnamese, so end turns on VAD silence.
            # Lesson 07 tunes these numbers; they are the documented defaults.
            turn_detection="vad",
            endpointing={"min_delay": 0.5, "max_delay": 3.0},
        ),
    )

    await session.start(
        room=ctx.room,
        agent=HotlineAgent(caller),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=krisp.voice_isolation_telephony(),  # tuned for SIP callers
            ),
        ),
    )

    # Starting the session published the agent's audio track, which lets LiveKit answer the call.
    await session.generate_reply(instructions="Chào người gọi và hỏi bạn có thể giúp gì.")


if __name__ == "__main__":
    agents.cli.run_app(server)
