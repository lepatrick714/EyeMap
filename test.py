import Algorithmia
import shutil
import os

print(os.getcwd())
input = ["https://i.imgur.com/U9j0CZj.jpg", "data://.algo/temp/result.jpg"]
Algorithmia.apiAddress = 'https://api-algorithmia-com-2wstoj4gszx4.runscope.net'
client = Algorithmia.client('simIMbhKq/Y4wc/maGC8Nr30Jzc1')
algo = client.algo('opencv/EyeDetection/0.1.1')
filename = algo.pipe(input)
print(filename)
fileExists = client.file("data://.algo/opencv/ObjectDetectionWithModels/temp/result.jpg").exists()
print(fileExists)
ourfile = client.file("data://.algo/opencv/ObjectDetectionWithModels/temp/result.jpg").getFile()
print(ourfile.name)
shutil.copy(ourfile.name, "/Users/Ishan/Hackpoly-2016/EyeDetect/bin/result.jpg")
