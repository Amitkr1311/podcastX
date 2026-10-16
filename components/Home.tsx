"use client";

import { api } from "@/convex/_generated/api";
import { PodcastProps } from "@/types";
import { useQuery } from "convex/react";
import LoaderSpinner from "./LoaderSpinner";
import PodcastCard from "./PodcastCard";

type HomeProps = {
  initialPodcasts: PodcastProps[];
};

export default function Home({ initialPodcasts }: HomeProps) {
  const livePodcasts = useQuery(api.podcasts.getTrendingPodcasts);
  const trendingPodcasts = livePodcasts ?? initialPodcasts;

  if (!trendingPodcasts) return <LoaderSpinner />;

  return (
    <div className="mt-9 flex flex-col gap-9 md:overflow-hidden">
      <section className="flex flex-col gap-5">
        <h1 className="text-20 font-bold text-white-1">Trending Podcasts</h1>

        <div className="podcast_grid">
          {trendingPodcasts.map(({ _id, podcastTitle, podcastDescription, imageUrl }, index) => (
            <PodcastCard
              key={_id}
              imgUrl={imageUrl as string}
              title={podcastTitle}
              description={podcastDescription}
              podcastId={_id}
              priority={index < 2}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
