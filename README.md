# ComfyUI-RVC
a comfyui custom node for [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git),you can Voice-Conversion in comfyui now!

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-RVC.git
cd ComfyUI-RVC
pip install -r requirements.txt
```
`weights` will be download from huggingface automatically!if you in china,make sure your internet attach the huggingface
or if you still struggle with huggingface, you may try follow [hf-mirror](https://hf-mirror.com/) to config your env.

或者下载[rvc_assets.zip](https://pan.quark.cn/s/039c8d2d59ac)解压后放置到`ComfyUI-RVC/rvc`目录

## Tutorial
[Demo](https://www.bilibili.com/video/BV1bH4y1P7n9/)

## WeChat Group && Donate
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <img alt='donate' src="donate.jpg?raw=true" width="300px"/>
  <figure>
</div>

## Thanks
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git)
