# PodcastX

PodcastX is an AI-powered podcast creation and discovery application. Users can turn a script into a two-speaker audio episode, create or upload cover art, publish podcasts, and explore work from other creators.

## Highlights

- Generate conversational audio with selectable voices.
- Create AI cover art or upload a custom image.
- Browse trending podcasts, search episodes, and view creator profiles.
- Play episodes with in-app playback controls.
- Manage identity with Clerk and application data with Convex.
- Keep Clerk's client code off the public homepage by scoping authentication providers to auth and protected routes.

## Technology

Next.js App Router, React, TypeScript, Convex, Clerk, OpenAI, Deepgram, Hugging Face, and Tailwind CSS

## Documentation

- [Documentation index](./docs/README.md)
- [System architecture](./docs/architecture.md)
- [Local development](./docs/development.md)
- [Production and Clerk setup](./docs/deployment.md)
- [Performance notes and Lighthouse assessment](./docs/performance.md)
- [Lighthouse report PDF](./docs/Lighthouse_Report.pdf)

## Quick start

```bash
npm install
npx convex dev
```

In a second terminal, run `npm run dev` and open [http://localhost:3000](http://localhost:3000). See the [development guide](./docs/development.md) for environment setup.

## License

MIT. See [`LICENSE`](./LICENSE).
