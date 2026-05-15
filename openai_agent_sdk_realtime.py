import argparse
import asyncio
import queue
import threading

from dotenv import load_dotenv
load_dotenv()

import pyaudio
import numpy as np
from agents import function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.realtime import RealtimeAgent, RealtimeRunner, realtime_handoff

from tool_definitions import (
    calculate,
    get_weather,
    run_python_code,
    write_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent realtime example")
    parser.add_argument(
        "--input-device",
        type=int,
        default=0,
        help="PyAudio input (mic) device index. Run mic_detect.py to list devices.",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=1,
        help="PyAudio output (speaker) device index. Run mic_detect.py to list devices.",
    )
    return parser.parse_args()




# Wrap tool functions for the agents SDK
weather_tool = function_tool(get_weather)
calculator_tool = function_tool(calculate)
python_code_tool = function_tool(run_python_code)
file_write_tool = function_tool(write_file)

# Sub-agents: one per tool
weather_agent = RealtimeAgent(
    name="WeatherAgent",
    handoff_description="A helpful agent that can look up weather for any city.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a weather specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the city the customer is asking about.
    2. Use the get_weather tool to look up the weather. Do not rely on your own knowledge.
    3. If the customer asks a question that is not related to weather, transfer back to the triage agent.""",
    tools=[weather_tool],
)

calculator_agent = RealtimeAgent(
    name="CalculatorAgent",
    handoff_description="A helpful agent that can evaluate math expressions and calculations.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a math specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the math expression the customer wants evaluated.
    2. Use the calculate tool to evaluate the expression. Do not rely on your own knowledge.
    3. If the customer asks a question that is not related to math, transfer back to the triage agent.""",
    tools=[calculator_tool],
)

python_code_agent = RealtimeAgent(
    name="PythonCodeAgent",
    handoff_description="A helpful agent that can write and execute Python scripts.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a Python code execution specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Understand what Python code the customer wants to run.
    2. Use the run_python_code tool to write and execute the script.
    3. Report the results back to the customer.
    4. If the customer asks a question that is not related to running Python code, transfer back to the triage agent.""",
    tools=[python_code_tool],
)

file_writer_agent = RealtimeAgent(
    name="FileWriterAgent",
    handoff_description="A helpful agent that can write content to files on disk.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a file writing specialist. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask the customer for the file path and content if not already provided.
    2. Use the write_file tool to write the content to the file.
    3. Confirm the file was written successfully.
    4. If the customer asks a question that is not related to writing files, transfer back to the triage agent.""",
    tools=[file_write_tool],
)

# Triage agent: orchestrates all sub-agents
triage_agent = RealtimeAgent(
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        weather_agent,
        realtime_handoff(calculator_agent),
        realtime_handoff(python_code_agent),
        realtime_handoff(file_writer_agent),
    ],
)

# Set up handoffs back to triage agent
weather_agent.handoffs.append(triage_agent)
calculator_agent.handoffs.append(triage_agent)
python_code_agent.handoffs.append(triage_agent)
file_writer_agent.handoffs.append(triage_agent)


# Required Audio Specs
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024
# Larger output buffer reduces underruns; mic-level meter throttling cuts console stalls.
OUTPUT_CHUNK = 4096
MIC_METER_EVERY_N_CHUNKS = 5

async def main(*, input_device_index: int = 0, output_device_index: int = 1):
    p = pyaudio.PyAudio()
    # Setup Mic and Speaker
    mic = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=input_device_index)
    speaker = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=OUTPUT_CHUNK, output_device_index=output_device_index)

    # Background thread drains this queue into the speaker so blocking writes
    # never stall the asyncio loop (the cause of playback cut-offs).
    playback_queue: queue.Queue[bytes | None] = queue.Queue()

    def playback_worker():
        while True:
            chunk = playback_queue.get()
            if chunk is None:
                return
            try:
                speaker.write(chunk)
            except Exception:
                return

    def flush_playback():
        # Drop any buffered audio so barge-in stops the assistant immediately.
        try:
            while True:
                playback_queue.get_nowait()
        except queue.Empty:
            pass

    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    runner = RealtimeRunner(
        triage_agent,
        config={"model_settings": {"voice": "shimmer"}},
    )

    print("--- Multi-Agent Session Active (Speak into mic) ---")
    print("Agents: Weather | Calculator | Python Code | File Writer")
    print("Triage agent will route your requests.\n")

    async with await runner.run() as session:
        async def send_mic_audio():
            """Reads from hardware and uses the validated 'send_audio' method."""
            tick = 0
            try:
                while True:
                    raw_data = await asyncio.to_thread(mic.read, CHUNK, False)

                    tick += 1
                    if tick % MIC_METER_EVERY_N_CHUNKS == 0:
                        audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
                        rms = np.sqrt(np.mean(audio_data**2))
                        meter = int(min(rms / 50, 50))
                        print(f"Mic Level: {'█' * meter}{' ' * (50-meter)} |", end="\r")

                    await session.send_audio(raw_data)
            except Exception:
                pass

        async def handle_events():
            """Iterates over the session as an AsyncIterator."""
            async for event in session:
                if event.type == "audio":
                    playback_queue.put_nowait(event.audio.data)
                elif event.type == "audio_interrupted":
                    flush_playback()
                    print("\n[interrupted]")
                elif event.type == "transcript_delta":
                    print(event.delta, end="", flush=True)

        # Execute
        mic_task = asyncio.create_task(send_mic_audio())
        try:
            await handle_events()
        finally:
            mic_task.cancel()

    # Cleanup
    playback_queue.put(None)
    playback_thread.join(timeout=1.0)
    mic.close()
    speaker.close()
    p.terminate()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(input_device_index=args.input_device, output_device_index=args.output_device))
