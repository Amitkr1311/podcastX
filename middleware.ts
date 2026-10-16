import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const isPublicRoute = createRouteMatcher(['/sign-in(.*)', '/sign-up(.*)', '/'])

export default clerkMiddleware((auth, req) => {
  if (!isPublicRoute(req)) auth().protect();
});

export const config = {
  // The homepage is public and does not need Clerk's development handshake.
  // Keep Clerk middleware on all non-root application routes and APIs.
  matcher: ['/((?!.*\\..*|_next).+)', '/(api|trpc)(.*)'],
};
