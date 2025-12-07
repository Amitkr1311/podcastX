

const PodcastDetails = async ({ params }: {params : Promise< {podcastId : string}> }) => {
  return (
    <p className='text-white-1'>
        Podcast Details for {(await params).podcastId}
    </p>
  )
}

export default PodcastDetails
