from __future__ import annotations as _annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Never
import logfire
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, Graph, End
from pydantic_ai.format_as_xml import format_as_xml
from pydantic import BaseModel, Field


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
class Email:
    sender: str
    plain_text: str | None


class ThreadState(BaseModel):
    emails: list[Email] = Field(default_factory=list)
    state: Literal['replying', 'forwarding', 'dropping'] = 'replying'


@dataclass
class NewEmailExchange(BaseNode[ThreadState]):
    email: EmailInfo

    async def run(self, ctx: GraphRunContext[ThreadState]) -> AgentReply | ForwardEmail:
        email = Email(sender=self.email.from_, plain_text=self.email.text or self.email.html)
        ctx.state.emails.append(email)
        result = await new_thread_agent.run(f'New email thread:\n{format_as_xml(email)}')
        if result.data.status == 'ok':
            ctx.state.state = 'forwarding'
            return ForwardEmail(result.data)
        else:
            return AgentReply(result.data)


@dataclass
class NewReply(BaseNode[ThreadState]):
    email: EmailInfo

    async def run(self, ctx: GraphRunContext[ThreadState]) -> AgentReply | ForwardEmail | DropEmail:
        email = Email(sender=self.email.from_, plain_text=self.email.text)
        if ctx.state.state == 'dropping':
            return DropEmail(EmailDrop('already dropping'))
        elif ctx.state.state == 'forwarding':
            ctx.state.emails.append(email)
            return ForwardEmail(EmailOk('already forwarding'))

        ctx.state.emails.append(email)
        result = await new_reply_agent.run(f'New reply in thread:\n{format_as_xml(ctx.state.emails)}')
        if result.data.status == 'ok':
            ctx.state.state = 'forwarding'
            return ForwardEmail(result.data)
        elif result.data.status == 'reply':
            return AgentReply(result.data)
        else:
            ctx.state.state = 'dropping'
            return DropEmail(result.data)


@dataclass
class AgentReply(BaseNode[ThreadState]):
    response: EmailReply

    async def run(self, ctx: GraphRunContext[ThreadState]) -> NewReply:
        raise NotImplementedError('AgentReply should not be run')


@dataclass
class ForwardEmail(BaseNode[ThreadState]):
    response: EmailOk

    async def run(self, ctx: GraphRunContext[ThreadState]) -> NewReply:
        raise NotImplementedError('ForwardEmail should not be run')


@dataclass
class DropEmail(BaseNode[ThreadState]):
    response: EmailDrop

    async def run(self, ctx: GraphRunContext[ThreadState]) -> NewReply:
        raise NotImplementedError('DropEmail should not be run')


email_graph = Graph(
    nodes=[NewEmailExchange, NewReply, AgentReply, ForwardEmail, DropEmail]
)


@logfire.instrument()
async def analyse_email(email: EmailInfo) -> EmailOk | EmailReply | EmailDrop:
    state: ThreadState | None = None
    if email.references:
        thread_reference = email.references.split(' ', 1)[0]
        ref_hash = hashlib.md5(thread_reference.encode()).hexdigest()
        state_path = Path(f'threads/{ref_hash:.7}.json')
        if state_path.exists():
            state = ThreadState.model_validate_json(state_path.read_bytes())
    else:
        ref_hash = hashlib.md5(email.message_id.encode()).hexdigest()
        state_path = Path(f'threads/{ref_hash:.7}.json')

    node: BaseNode[ThreadState, None, Never]
    if state is None:
        state = ThreadState()
        node = NewEmailExchange(email)
    else:
        node = NewReply(email)

    with logfire.span('running graph', state_path=state_path):
        history = []
        while True:
            next_node = await email_graph.next(node, history, state=state)
            if isinstance(next_node, AgentReply | ForwardEmail | DropEmail):
                response = next_node.response
                state_path.parent.mkdir(exist_ok=True)
                state_path.write_text(state.model_dump_json(indent=2))
                return response
            else:
                assert isinstance(next_node, BaseNode)
                node = next_node


if __name__ == '__main__':
    print('generating mermaid diagram...')
    path = Path('email_graph.jpg')
    email_graph.mermaid_save(path, start_node=[NewEmailExchange], highlighted_nodes=[NewReply])
    print(f'saved mermaid diagram to {path}')
