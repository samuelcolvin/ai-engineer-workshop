import { loadPyodide } from 'pyodide'

async function main() {
  const args = process.argv.slice(2)
  const deps = args[0]
  const code = args[1]

  const stdout = []
  const stderr = []
  const pyodide = await loadPyodide({

    stdout: (msg) => {
      stdout.push(msg)
    },
    stderr: (msg) => {
      stderr.push(msg)
    }
  })
  const packages = ['micropip', 'pydantic']
  if (deps.includes('numpy')) {
    packages.push('numpy')
  }
  await pyodide.loadPackage(packages)
  try {
    const rawReturnValue = await pyodide.runPythonAsync(`
deps = ${deps}
if deps:
    import micropip, importlib

    await micropip.install(deps)
    importlib.invalidate_caches()

${code}
  `)
    // hack to avoid serialization issues
    const returnValue = await pyodide.runPythonAsync(`
from typing import Any
from pydantic_core import to_json

def fallback(value: Any) -> Any:
    tp: Any = type(value)
    module = tp.__module__
    if module == 'numpy':
        if tp.__name__ in ('ndarray', 'matrix'):
            return value.tolist()
        else:
            return value.item()
    elif module == 'pyodide.ffi':
        return value.to_py()
    else:
        return repr(value)

to_json(rawReturnValue, indent=2, fallback=fallback).decode()
`, {globals: pyodide.toPy({rawReturnValue})})
    output({stdout: stdout.join(''), stderr: stderr.join(''), returnValue})
  } catch (e) {
    output({error: e.toString()})
  }
}

function output(data) {
  console.log('!!OUTPUT:' + JSON.stringify(data, null, 2))
}

main()
