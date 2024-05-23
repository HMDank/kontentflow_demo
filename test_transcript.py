from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound


def get_youtube_transcript(video_id):
    try:
        # Attempt to retrieve the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["vi"]
        )
        transcript = "\n".join([i["text"] for i in transcript_list])
        return transcript
    except NoTranscriptFound:
        # Return an error message if no transcript is found
        return "No transcript found for this video."


# Replace 'video_id' with the YouTube video ID for which you want the transcript
video_id = "nUXNWMVsRPk"
transcript = get_youtube_transcript(video_id)
print(transcript)
