import Home from "@/components/Home";
import { api } from "@/convex/_generated/api";
import { ConvexHttpClient } from "convex/browser";
import { PodcastProps } from "@/types";

export default async function HomePage() {
  const convex = new ConvexHttpClient(process.env.NEXT_PUBLIC_CONVEX_URL as string);
  const initialPodcasts = (await convex.query(api.podcasts.getTrendingPodcasts)) as PodcastProps[];

  return <Home initialPodcasts={initialPodcasts} />;
}