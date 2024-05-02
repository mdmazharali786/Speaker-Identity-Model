from pytube import YouTube
print("Youtube audio extraction Program.")
url = input("Enter youtube link:")
you_tube = YouTube(url)
audi = you_tube.streams.filter(only_audio=True).first()
fname= url.split('=')[1:]
fname = ''.join(fname)
path = audi.download(output_path='sample', filename=fname+'.wav')
print("Downloaded successfully at -", path)