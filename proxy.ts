import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextRequest } from 'next/server';

// Define your public routes here
const isPublicRoute = createRouteMatcher([
  '/',          // Homepage
  '/sign-in(.*)', // Sign-in and subpaths
  '/sign-up(.*)', // Sign-up and subpaths
]);

export default clerkMiddleware(async (auth, req : NextRequest) => {
  // If route is not public, require authentication
    //  const session = await auth();

    //  if(!isPublicRoute(req) && !session.userId) {
    //     return Response.redirect(new URL('/sign-in', req.url));
    //  }
    if (!isPublicRoute(req)) {
    (await auth()).protect(); // This handles the redirect automatically
  }
});

export const config = {
  matcher: [
    // Run Clerk on all routes except Next.js internals/static files
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API and tRPC routes
    '/(api|trpc)(.*)',
  ],
};
