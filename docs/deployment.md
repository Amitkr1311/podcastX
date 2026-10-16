# Production deployment

## Application configuration

Set production environment variables in the hosting platform before deploying:

- `NEXT_PUBLIC_CONVEX_URL`
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
- `CLERK_SECRET_KEY`

Set the AI provider and webhook secrets in the Convex production deployment:

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `HUGGING_FACE_API_KEY`
- `CLERK_WEBHOOK_SECRET`

Use separate development and production credentials. Do not copy local development secrets into public build artifacts.

## Clerk production setup

The saved Lighthouse capture used a Clerk development frontend domain ending in `accounts.dev`. A production deployment should use the Clerk production instance and its production keys (`pk_live_` and `sk_live_`). Clerk development instances are intended for development and use different domain and session behavior.

1. In the Clerk Dashboard, create or select the production instance.
2. Add the application's production domain in the Dashboard's **Domains** settings.
3. Copy the production publishable and secret keys into the hosting platform's environment variables.
4. Configure any OAuth providers and redirect URLs for production if social sign-in is enabled.
5. Configure the Clerk webhook to call the Convex deployment's `/clerk` endpoint and set its signing secret as `CLERK_WEBHOOK_SECRET` in Convex.
6. Deploy the app with the production environment variables, then verify sign-in, sign-out, and a protected route.

Clerk displays the DNS records required for its production domain in the Dashboard. Add those exact records at the DNS provider for the selected domain. The application's web-host DNS and Clerk's frontend API DNS are separate settings; do not substitute guessed records. DNS changes may take time to propagate.

Official guidance: [Deploy a Clerk app to production](https://clerk.com/docs/guides/development/deployment/production), [manage Clerk environments](https://clerk.com/docs/guides/development/managing-environments), and [change a production domain](https://clerk.com/docs/guides/development/deployment/changing-domains).

## Post-deployment checks

- Confirm the public homepage loads and its network requests do not include the development Clerk handshake.
- Confirm sign-in and sign-up complete successfully with production keys.
- Confirm protected routes redirect signed-out visitors to sign-in.
- Confirm Convex queries, podcast generation, uploads, and the Clerk webhook are configured for the production deployment.
- Re-run Lighthouse against the deployed homepage after the configuration changes.
