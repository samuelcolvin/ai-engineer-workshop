from __future__ import annotations as _annotations

import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
import logfire
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, Graph, End
from pydantic_ai.format_as_xml import format_as_xml
from pydantic import BaseModel, TypeAdapter


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


new_thread_agent: Agent[None, EmailOk | EmailReply] = Agent(
    'openai:gpt-4o',
    result_type=EmailOk | EmailReply  # type: ignore
)


class Example(BaseModel):
    subject: str
    body: str
    response: Literal["EmailReply", "EmailOk"]


class Prompt(BaseModel):
    new_thread_prompt: str
    reply_prompt: str
    examples: list[Example]


@new_thread_agent.system_prompt
def load_thread_prompt():
    file = Path(__file__).parent / 'prompt.toml'
    with file.open('rb') as f:
        raw_prompt = tomllib.load(f)
    prompt = Prompt.model_validate(raw_prompt)
    return f'{prompt.new_thread_prompt}\n{format_as_xml(prompt.examples)}'


new_reply_agent = Agent[None, EmailOk | EmailReply | EmailDrop](
    'openai:gpt-4o',
    result_type=EmailOk | EmailReply | EmailDrop  # type: ignore
)


@new_reply_agent.system_prompt
def load_reply_prompt():
    file = Path(__file__).parent / 'prompt.toml'
    with file.open('rb') as f:
        raw_prompt = tomllib.load(f)
    prompt = Prompt.model_validate(raw_prompt)
    return f'{prompt.reply_prompt}\n{format_as_xml(prompt.examples)}'


@dataclass
class ThreadMessage:
    sender: str
    plain_text: str | None


extract_agent = Agent(
    'openai:gpt-4o',
    result_type=list[ThreadMessage],
    system_prompt='Extract the individual emails from the thread thread of replies.',
)


class ThreadStatus(StrEnum):
    forwarding = 'forwarding'
    dropping = 'dropping'


ta = TypeAdapter(ThreadStatus)


def get_path(email: EmailInfo) -> Path:
    if email.references:
        # we're in an email thread
        thread_reference = email.references.split(' ', 1)[0]
    else:
        thread_reference = email.message_id

    thread_reference = re.sub(r'[^0-9a-zA-Z_\-.]', '_', thread_reference)
    return Path(f'threads/{thread_reference:.12}.txt')


@dataclass
class NewEmail(BaseNode[None, None, EmailOk | EmailReply | EmailDrop]):
    email: EmailInfo

    async def run(self, ctx: GraphRunContext) -> ExtractEmails | NewEmailExchange | End[EmailOk | EmailReply | EmailDrop]:
        if self.email.references:
            # we're in an email thread
            thread_path = get_path(self.email)
            logfire.info('checking {path=!r} {exists=}', path=thread_path, exists=thread_path.exists())
            if thread_path.exists():
                status = ta.validate_json(thread_path.read_bytes())
                if status == ThreadStatus.forwarding:
                    return End(EmailOk('already forwarding'))
                else:
                    assert status == ThreadStatus.dropping
                    return End(EmailDrop('already dropping'))

            return ExtractEmails(self.email)
        else:
            return NewEmailExchange(self.email)


@dataclass
class NewEmailExchange(BaseNode):
    email: EmailInfo

    async def run(self, ctx: GraphRunContext) -> FinishStore:
        msg = ThreadMessage(sender=self.email.from_, plain_text=self.email.text or self.email.html)
        result = await new_thread_agent.run(f'New email thread:\n{format_as_xml(msg)}')
        return FinishStore(self.email, result.data)


@dataclass
class ExtractEmails(BaseNode):
    email: EmailInfo

    async def run(self, ctx: GraphRunContext) -> NewReply:
        result = await extract_agent.run(self.email.text or '')
        thread = cast(list[ThreadMessage], result.data)
        # reverse because the top message is the most recent
        thread.reverse()
        return NewReply(self.email, thread)


@dataclass
class NewReply(BaseNode):
    email: EmailInfo
    thread: list[ThreadMessage]

    async def run(self, ctx: GraphRunContext) -> FinishStore:
        result = await new_reply_agent.run(f'New reply in thread:\n{format_as_xml(self.thread)}')
        return FinishStore(self.email, result.data)


@dataclass
class FinishStore(BaseNode[None, None, EmailOk | EmailReply | EmailDrop]):
    email: EmailInfo
    result: EmailOk | EmailReply | EmailDrop

    async def run(self, ctx: GraphRunContext) -> End[EmailOk | EmailReply | EmailDrop]:
        if isinstance(self.result, EmailReply):
            # don't save status if we're replying
            return End(self.result)

        thread_path = get_path(self.email)
        status = ThreadStatus.forwarding if isinstance(self.result, EmailOk) else ThreadStatus.dropping
        logfire.info('writing {path!r=} {status=}', path=thread_path, status=status)
        thread_path.parent.mkdir(exist_ok=True)
        thread_path.write_bytes(ta.dump_json(status))
        return End(self.result)


email_graph = Graph(nodes=[NewEmail, NewEmailExchange, ExtractEmails, NewReply, FinishStore])


@logfire.instrument()
async def analyse_email(email: EmailInfo) -> EmailOk | EmailReply | EmailDrop:
    response, _ = await email_graph.run(NewEmail(email))
    return response


if __name__ == '__main__':
    print('generating mermaid diagram...')
    mermaid_path = Path('email_graph.jpg')
    email_graph.mermaid_save(mermaid_path, start_node=[NewEmail])
    print(f'saved mermaid diagram to {mermaid_path}')
