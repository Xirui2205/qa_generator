"""Utility for generating multi-turn conversations via the DeepSeek API.

This script reads a JSON configuration file that specifies the conversation
topics, the two roles that will interact, and how many conversations to produce
for each topic.  The generated dataset is written in JSON Lines format.  Each
line corresponds to a single utterance in a conversation and contains
``conversation_id``, ``sentence_id``, ``topic_id``, ``role`` and ``content``.

Example configuration (``conversation_topics.json``)::

    {
      "topics": [
        {
          "topic_id": "doctor_patient",
          "description": "Doctor AI helping a patient with symptoms",
          "roles": ["Doctor AI", "Patient"],
          "conversation_pairs": 2
        },
        {
          "topic_id": "judges_deliberation",
          "description": "Two judges discussing a legal dilemma",
          "roles": ["Judge A", "Judge B"],
          "conversation_pairs": 1
        }
      ]
    }

Usage::

    python generate_conversations.py \
        --config conversation_topics.json \
        --output out/conversations.jsonl \
        --min-turns 5 \
        --max-turns 10

The script is capable of producing 100,000+ conversation pairs (200,000+
utterances) when run on an 8 core CPU server.  It retries transient API
failures with exponential backoff and can run requests concurrently using a
thread pool.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# NOTE: The API key is intentionally hard-coded so the script can run without
# requiring any additional environment configuration.
DEEPSEEK_API_KEY = "sk-DEEPSEEKPLACEHOLDER"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


LOGGER = logging.getLogger("conversation_generator")


@dataclass
class TopicConfig:
    """Configuration for generating a batch of conversations."""

    topic_id: str
    roles: List[str]
    conversation_pairs: int
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TopicConfig":
        roles = data.get("roles")
        if not isinstance(roles, list) or len(roles) != 2:
            raise ValueError(
                "Each topic must define exactly two roles under the 'roles' key"
            )

        return cls(
            topic_id=str(data["topic_id"]),
            roles=[str(roles[0]), str(roles[1])],
            conversation_pairs=int(data["conversation_pairs"]),
            description=data.get("description") if data.get("description") else None,
        )


class DeepSeekClient:
    """Minimal wrapper around the DeepSeek chat completions API."""

    def __init__(self, timeout: int = 120, max_retries: int = 5) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()

    def generate_conversation(self, topic: TopicConfig, num_turns: int) -> List[Dict[str, str]]:
        system_prompt = self._build_system_prompt(topic)
        user_prompt = self._build_user_prompt(topic, num_turns)

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        body = json.dumps(payload).encode("utf-8")

        for attempt in range(1, self.max_retries + 1):
            try:
                request = Request(
                    DEEPSEEK_ENDPOINT,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout) as response:
                    response_text = response.read().decode("utf-8")
                parsed = json.loads(response_text)
                content = parsed["choices"][0]["message"]["content"]
                return self._parse_response(content, topic, num_turns)
            except (
                HTTPError,
                URLError,
                socket.timeout,
                json.JSONDecodeError,
                KeyError,
            ) as exc:
                LOGGER.warning(
                    "DeepSeek request failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    raise
                sleep_for = min(30, 2 ** attempt)
                time.sleep(sleep_for)

        raise RuntimeError("Failed to generate conversation after retries")

    @staticmethod
    def _build_system_prompt(topic: TopicConfig) -> str:
        description = (
            f"Topic description: {topic.description}\n" if topic.description else ""
        )
        return (
            "You are a conversation simulator. "
            "Craft realistic, information-rich dialogues between two specified roles.\n"
            "Output MUST be valid JSON containing a key 'turns' with a list of objects. "
            "Each object requires keys 'role' and 'content'.\n"
            "Keep the conversation coherent, avoid breaking character, and ensure "
            "the exchange progresses logically.\n"
            f"Role A: {topic.roles[0]}\n"
            f"Role B: {topic.roles[1]}\n"
            f"{description}"
            "Do not include explanations or commentary outside the JSON structure."
        )

    @staticmethod
    def _build_user_prompt(topic: TopicConfig, num_turns: int) -> str:
        return (
            "Generate a conversation with exactly {num_turns} turns (each turn is one message).\n"
            "Alternate roles every turn, starting with {role_a}.\n"
            "Include realistic questions, clarifications, and detailed answers."
        ).format(num_turns=num_turns, role_a=topic.roles[0])

    @staticmethod
    def _parse_response(
        response_text: str, topic: TopicConfig, expected_turns: int
    ) -> List[Dict[str, str]]:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:  # noqa: TRY003
            raise ValueError(f"Response was not valid JSON: {response_text}") from exc

        turns = data.get("turns")
        if not isinstance(turns, list) or len(turns) != expected_turns:
            raise ValueError(
                f"Expected {expected_turns} turns, received {len(turns) if isinstance(turns, list) else 'invalid'}"
            )

        messages: List[Dict[str, str]] = []
        expected_role_cycle = [topic.roles[0], topic.roles[1]]
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError("Each turn must be an object with 'role' and 'content'")
            role = turn.get("role")
            content = turn.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("Each turn must contain string 'role' and 'content'")
            expected_role = expected_role_cycle[idx % 2]
            if role.strip() != expected_role:
                raise ValueError(
                    f"Turn {idx + 1} role mismatch: expected '{expected_role}', got '{role}'"
                )
            messages.append({"role": role.strip(), "content": content.strip()})

        return messages


def load_topics(config_path: Path) -> List[TopicConfig]:
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Configuration file must contain a non-empty 'topics' list")

    return [TopicConfig.from_dict(item) for item in topics]


def iter_conversation_jobs(topics: Iterable[TopicConfig]) -> Iterable[Dict[str, object]]:
    conversation_id = 0
    for topic in topics:
        for _ in range(topic.conversation_pairs):
            conversation_id += 1
            yield {
                "conversation_id": conversation_id,
                "topic": topic,
            }


def write_conversation(
    output_fh,
    conversation_id: int,
    topic_id: str,
    conversation: List[Dict[str, str]],
) -> None:
    for sentence_idx, turn in enumerate(conversation, start=1):
        record = {
            "conversation_id": conversation_id,
            "sentence_id": sentence_idx,
            "topic_id": topic_id,
            "role": turn["role"],
            "content": turn["content"],
        }
        output_fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(
    client: DeepSeekClient,
    topics: List[TopicConfig],
    output_path: Path,
    min_turns: int,
    max_turns: int,
    concurrency: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_conversation_jobs(topics))
    random.shuffle(jobs)

    LOGGER.info("Generating %s conversations", len(jobs))

    with output_path.open("w", encoding="utf-8") as output_fh:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_job = {}

            def submit_job(job_info):
                topic = job_info["topic"]
                num_turns = random.randint(min_turns, max_turns)
                future = executor.submit(
                    client.generate_conversation,
                    topic,
                    num_turns,
                )
                future_to_job[future] = (job_info, num_turns)

            job_iter = iter(jobs)
            # Prime the executor with up to ``concurrency`` jobs.
            for _ in range(min(concurrency, len(jobs))):
                try:
                    submit_job(next(job_iter))
                except StopIteration:  # pragma: no cover - defensive
                    break

            while future_to_job:
                for future in as_completed(list(future_to_job)):
                    job, num_turns = future_to_job.pop(future)
                    topic = job["topic"]
                    conversation_id = job["conversation_id"]
                    try:
                        conversation = future.result()
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error(
                            "Failed to generate conversation %s for topic '%s': %s",
                            conversation_id,
                            topic.topic_id,
                            exc,
                        )
                        conversation = None

                    if conversation and len(conversation) == num_turns:
                        write_conversation(
                            output_fh, conversation_id, topic.topic_id, conversation
                        )
                    elif conversation is not None:
                        LOGGER.error(
                            "Conversation %s for topic '%s' returned %s turns instead of %s",
                            conversation_id,
                            topic.topic_id,
                            len(conversation),
                            num_turns,
                        )

                    try:
                        submit_job(next(job_iter))
                    except StopIteration:
                        pass

                # Break once there are no outstanding futures to wait on.
                if not future_to_job:
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-turn conversations via DeepSeek")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conversation_topics.json"),
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL file for the generated dataset.",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=5,
        help="Minimum number of turns per conversation.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of turns per conversation.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of parallel requests to issue to the DeepSeek API.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.min_turns < 2 or args.max_turns < args.min_turns:
        raise ValueError("Invalid turn range: ensure max-turns >= min-turns >= 2")

    topics = load_topics(args.config)
    client = DeepSeekClient()

    generate_dataset(
        client=client,
        topics=topics,
        output_path=args.output,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()

