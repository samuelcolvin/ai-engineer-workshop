import json
import subprocess
from pathlib import Path
from typing import Sequence, Any

from pydantic_ai import Agent
import logfire

logfire.configure()
logfire.instrument_openai()


agent = Agent('openai:gpt-4o')


@agent.tool_plain
@logfire.instrument
def run_code(python_code: str, python_dependencies: Sequence[str] = ()) -> dict[str, Any] | str:
    """
    Use this tool to run python code to answer any quantitative, temporal or numerical questions.

    Arguments:
        python_code: The python code to run
        python_dependencies: The python dependencies to install before running the code

    Returns:
        An object with stdout, stderr, and the return value of the code - output from the last line of code
    """
    working_directory = Path(__file__).parent / 'pyodide_run_code'
    try:
        p = subprocess.run(
            ['node', 'run.mjs', repr(list(python_dependencies)), python_code],
            cwd=str(working_directory),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        logfire.exception('error running code')
        return 'error running code'
    else:
        index = p.stdout.index(b'!!OUTPUT:') + len('!!OUTPUT:')
        result = json.loads(p.stdout[index:])
        logfire.info('ran code {result=}', result=result)
        return result


async def main():
    result = await agent.run('how many days between 1970-01-01 and 2022-01-31?')
    print(result.data)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
