interface Env {}

interface AnalysisResult {
  forward: boolean
}

export default {
  async email(message, env, ctx) {
    const r = await fetch('https://samuelcolvin.eu.ngrok.io', {
      method: 'POST',
      body: message.raw,
    })
    const { forward }: AnalysisResult = await r.json()
    if (forward) {
      console.log('forwarding email')
      await message.forward('samuel@pydantic.dev')
    }
  },
} satisfies ExportedHandler<Env>
