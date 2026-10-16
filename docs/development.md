# Local development

## Requirements

- Node.js and npm
- A Convex project
- A Clerk application
- OpenAI, Deepgram, and Hugging Face credentials to use generation features

## Install

```bash
git clone https://github.com/Amitkr1311/podcastX.git
cd podcastX
npm install
```

Create `.env.local` in the repository root with the Next.js app values:

```env
NEXT_PUBLIC_CONVEX_URL=
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
```

Start the Convex development deployment in one terminal:

```bash
npx convex dev
```

Configure the following values as Convex deployment environment variables. They are read by Convex actions and the Clerk webhook handler:

```text
OPENAI_API_KEY
DEEPGRAM_API_KEY
HUGGING_FACE_API_KEY
CLERK_WEBHOOK_SECRET
```

Start Next.js in a second terminal:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Available commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Run the Next.js development server. |
| `npx convex dev` | Run and synchronize the Convex development deployment. |
| `npm run build` | Create a production build. |
| `npm run start` | Serve the production build. |
| `npm run lint` | Run the configured lint command. |

## Credential handling

Keep secret keys in local environment files or the corresponding service's environment settings. Never commit `.env.local`, Convex secrets, Clerk secret keys, or AI provider keys. The public Convex URL and Clerk publishable key are frontend configuration; the remaining values must stay server-side.

For live Clerk and DNS setup, see [Production deployment](./deployment.md).
