from __future__ import annotations as _annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import logfire
from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic import BaseModel


if TYPE_CHECKING:
    from .server import EmailInfo


@dataclass
class EmailOk:
    reason: str
    status: Literal['ok'] = 'ok'


@dataclass
class EmailReply:
    text: str
    reason: str
    status: Literal['reply'] = 'reply'


@dataclass
class EmailDrop:
    reason: str
    status: Literal['drop'] = 'drop'


agent = Agent[None, EmailOk | EmailReply](
    'openai:gpt-4o',
    result_type=EmailOk | EmailReply  # type: ignore
)


class Example(BaseModel):
    subject: str
    body: str
    response: Literal["EmailReply", "EmailOk"]


class Prompt(BaseModel):
    prompt: str
    examples: list[Example]


@agent.system_prompt
def load_prompt():
    file = Path(__file__).parent / 'prompt.toml'
    with file.open('rb') as f:
        raw_prompt = tomllib.load(f)
    prompt = Prompt.model_validate(raw_prompt)
    return f'{prompt.prompt}\n{format_as_xml(prompt.examples)}'


@logfire.instrument()
async def analyse_email(email: EmailInfo) -> EmailOk | EmailReply | EmailDrop:
    result = await agent.run(f'subject: {email.subject}\nbody: {email.text or email.html}')
    return result.data
