import { PodcastCardProps } from '@/types'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import React from 'react'

const PodcastCard = ({
  imgUrl, title, description, podcastId, priority = false
}: PodcastCardProps) => {
  const router = useRouter()

  const handleViews = () => {
    // increase views

    router.push(`/podcasts/${podcastId}`, {
      scroll: true
    })
  }

  return (
    <div className="cursor-pointer" onClick={handleViews}>
      <figure className="flex flex-col gap-2">
        <Image 
          src={imgUrl}
          width={348}
          height={196}
          priority={priority}
          quality={65}
          sizes="(min-width: 1536px) 200px, (min-width: 1024px) 30vw, (min-width: 640px) 45vw, 100vw"
          alt={title}
          className="aspect-video h-auto w-full rounded-xl object-cover 2xl:w-[200px]"
        />
        <div className="flex flex-col">
          <h1 className="text-16 truncate font-bold text-white-1">{title}</h1>
          <h2 className="text-12 truncate font-normal capitalize text-white-4">{description}</h2>
        </div>
      </figure>
    </div>
  )
}

export default PodcastCard