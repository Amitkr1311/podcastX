# Performance notes

## Lighthouse assessment

The supplied Lighthouse report is available at [`Lighthouse_Report.pdf`](./Lighthouse_Report.pdf). Lighthouse results are snapshots of a particular URL, build, browser, and network profile; use the PDF as the record of that run rather than as a guarantee of current production metrics.

## Public homepage work

The homepage is a public route and does not need the authenticated client provider. The current route structure keeps Clerk's JavaScript off the homepage:

- `app/layout.tsx` provides shared UI state without mounting Clerk globally.
- `app/(public)/layout.tsx` uses a Convex-only provider.
- The public homepage loads initial trending podcast data on the server.
- `app/(auth)/layout.tsx` and `app/(root)/layout.tsx` mount Clerk where authentication is needed.
- `middleware.ts` excludes the exact root path from its matcher while continuing to protect non-public app routes and APIs.

## Redirect findings

The earlier Lighthouse capture of `https://podcastx.iamamit.me/` recorded three `307` responses during a Clerk development handshake: a redirect from the app, a redirect through the Clerk `accounts.dev` frontend, and a return request to the app carrying a handshake parameter. The report showed the domain redirecting back to the homepage with a `200` response after the handshake.

The public route changes remove this handshake from `/`. Production Clerk configuration is still needed for authentication flows on sign-in and protected routes. See [Production deployment](./deployment.md).
