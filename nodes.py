import os
import time
import shutil
import folder_paths
import numpy as np
from pydub import AudioSegment
from multiprocessing import cpu_count
from scipy.io.wavfile import write as wavwrite
from .rvc.train import vc,gpus,default_batch_size,preprocess_dataset,extract_f0_feature,train1key

now_dir = os.path.dirname(os.path.abspath(__file__))
out_path = folder_paths.get_output_directory()
input_path = folder_paths.get_input_directory()
rvc_out_path = os.path.join(out_path,"RVC")
os.environ["weight_root"] = rvc_out_path
os.environ["index_root"] = rvc_out_path
os.environ["rmvpe_root"] = os.path.join(now_dir,"rvc","assets","rmvpe")
os.environ["hubert_base"] = os.path.join(now_dir,"rvc","assets","hubert","hubert_base.pt")
os.makedirs(os.path.join(now_dir,"rvc","assets", "weights"),exist_ok=True)
class RVC_Infer:
    @classmethod
    def INPUT_TYPES(s):
        pth_list = [f for f in os.listdir(rvc_out_path) if os.path.isfile(os.path.join(rvc_out_path, f)) and f.split('.')[-1] == "pth"]
        index_list = [f for f in os.listdir(rvc_out_path) if os.path.isfile(os.path.join(rvc_out_path, f)) and f.split('.')[-1] == "index"]
        return {"required":{
            "audio": ("AUDIO",),
            "sid":(pth_list,),
            "spk_item":("INT",{
                "min":0,
                "max":4,
                "step": 1,
                "default": 0,
                "display": "slider"
            }),
            "vc_transform":("INT",{
                "min":-12,
                "max":12,
                "step": 1,
                "default": 0,
                "display": "slider"
            }),
            "file_index": (index_list,),
            
            "f0_method":(["pm", "harvest", "crepe", "rmvpe"],
                {
                    "default": "rmvpe"
                    
            }),
            "resample_sr":("INT",{
                "min":0,
                "max":48000,
                "step": 1,
                "default": 0,
                "display": "slider"
            }),
            "rms_mix_rate":("FLOAT",{
                "min":0,
                "max":1.0,
                "step": 0.05,
                "default": 0.25,
                "display": "slider"
            }),
            "protect":("FLOAT",{
                "min":0,
                "max":0.5,
                "step": 0.01,
                "default": 0.33,
                "display": "slider"
            }),
            "filter_radius":("INT",{
                "min":0,
                "max":7,
                "step": 1,
                "default": 3,
                "display": "slider"
            }),
            "index_rate":("FLOAT",{
                "min":0,
                "max":1,
                "step": 0.05,
                "default": 0.75,
                "display": "slider"
            }),
        }}
    
    CATEGORY = "AIFSH_RVC"
    RETURN_TYPES = ("AUDIO",)
    OUTPUT_NODE = False

    FUNCTION = "inference"
    
    def inference(self,audio,sid,spk_item,vc_transform,
                  file_index,f0_method,resample_sr,rms_mix_rate,
                  protect,filter_radius,index_rate):
        audio_path = folder_paths.get_annotated_filepath(audio)
        file_index = os.path.join(rvc_out_path,file_index)
        vc.get_vc(sid,protect,protect)
        f0_file = ""
        info, (tgt_sr, audio_opt) = vc.vc_single(spk_item,input_audio_path=audio_path,f0_up_key=vc_transform,
                     f0_file=f0_file,f0_method=f0_method,file_index=file_index,
                     file_index2=file_index,index_rate=index_rate,filter_radius=filter_radius,
                     resample_sr=resample_sr,rms_mix_rate=rms_mix_rate,
                     protect=protect)
        print(info)
        audio_name = os.path.basename(audio_path)
        out_file = os.path.join(out_path,f"{time.time()}_rvc_{audio_name}")
        wavwrite(out_file, tgt_sr,audio_opt)
        return (out_file,)


class RVC_Train:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "audio":("AUDIO",),
            "exp_name": ("STRING",{
                "default": "singer1"
            }),
            "sr":(["32k","40k", "48k"],{
                "default":"40k"
            }),
            "if_f0_3":("BOOLEAN",{
                "default": True
            }),
            "version":(["v1", "v2"],{
                "default": "v2"
            }),
            "speaker_id":("INT",{
                "display": "slider",
                "min": 0,
                "max": 4,
                "step":1,
                "default":1
            }),
            "f0_method":(["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],{
                "default": "rmvpe_gpu",
            }),
            "save_epoch":("INT",{
                "display": "slider",
                "min": 1,
                "max": 50,
                "step":1,
                "default":5
            }),
            "total_epoch":("INT",{
                "display": "slider",
                "min": 2,
                "max": 1000,
                "step":1,
                "default":20
            }),
            "batch_size":("INT",{
                "display": "slider",
                "min": 1,
                "max": 40,
                "step":1,
                "default":default_batch_size
            }),
            "if_save_latest":("BOOLEAN",{
                "default": False
            }),
            "if_cache_gpu":("BOOLEAN",{
                "default": False
            }),
            "if_save_every_weights":("BOOLEAN",{
                "default": False
            }),
        }}
        
    CATEGORY = "AIFSH_RVC"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "train"
    
    def train(self,audio,exp_name,sr,if_f0_3,version,
             speaker_id,f0_method,save_epoch,total_epoch,
             batch_size,if_save_latest,if_cache_gpu,
             if_save_every_weights):
        # step 1
        if sr == "32k" and version == "v1":
            sr = "40k"
        shutil.rmtree(os.path.join(now_dir,"rvc","logs",exp_name),ignore_errors=True)
        wav_path = folder_paths.get_annotated_filepath(audio)
        exp_path = os.path.join(now_dir,"rvc","logs",exp_name)
        trainset_dir = os.path.join(exp_path,"wavs")
        os.makedirs(trainset_dir, exist_ok=True)
        shutil.copy(wav_path,os.path.join(trainset_dir,os.path.basename(wav_path)))
        n_p = int(np.ceil(cpu_count() / 1.5))
        gpus_rmvpe="%s-%s" % (gpus, gpus)
        preprocess_dataset(trainset_dir,exp_dir=exp_name,sr=sr,n_p=n_p)
        
        # step 2
        extract_f0_feature(gpus,n_p,f0method=f0_method,if_f0=if_f0_3,exp_dir=exp_name,version19=version,gpus_rmvpe=gpus_rmvpe)
        
        path_str = "" if version == "v1" else "_v2"
        f0_str = "f0" if if_f0_3 else ""
        '''
        pretrained_G = os.path.join(now_dir,"rvc","assets",f"pretrained{path_str}",f"{f0_str}G{sr}.pth")
        pretrained_D = os.path.join(now_dir,"rvc","assets",f"pretrained{path_str}",f"{f0_str}D{sr}.pth")
        shutil.copy(pretrained_G,os.path.join(exp_path,"%sG%s.pth" % (f0_str, sr)))
        shutil.copy(pretrained_D,os.path.join(exp_path,"%sD%s.pth" % (f0_str, sr)))
        '''
        # step 3 "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr)
        train1key(exp_dir1=exp_name,sr2=sr,if_f0_3=if_f0_3,trainset_dir4=trainset_dir,spk_id5=speaker_id,
                  np7=n_p,f0method8=f0_method,save_epoch10=save_epoch,total_epoch11=total_epoch,
                  batch_size12=batch_size,if_save_latest13=if_save_latest,
                  pretrained_G14="assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr),
                  pretrained_D15="assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr),
                  gpus16=gpus,if_cache_gpu17=if_cache_gpu,if_save_every_weights18=if_save_every_weights,
                  version19=version,gpus_rmvpe=gpus_rmvpe)
        weight_path = os.path.join(now_dir,"rvc","assets","weights",f"{exp_name}.pth")
        shutil.copy(weight_path, os.path.join(rvc_out_path,f"{exp_name}.pth"))
        return {"ui": {"train":[rvc_out_path]}}

class CombineAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"vocal_AUDIO": ("AUDIO",),
                     "bgm_AUDIO": ("AUDIO",)
                     }
                }

    CATEGORY = "AIFSH_RVC"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("AUDIO",)

    OUTPUT_NODE = True

    FUNCTION = "combine_audio"

    def combine_audio(self, vocal_AUDIO, bgm_AUDIO):
        vocal = AudioSegment.from_file(vocal_AUDIO)
        bgm = AudioSegment.from_file(bgm_AUDIO)
        audio = vocal.overlay(bgm)
        audio_file = os.path.join(out_path,f"{time.time()}_rvc_result_voice.wav")
        audio.export(audio_file, format="wav")
        return (audio_file,)

class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO",),}
                }

    CATEGORY = "AIFSH_RVC"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        tmp_path = os.path.dirname(audio)
        audio_root = os.path.basename(tmp_path)
        return {"ui": {"audio":[audio_name,audio_root]}}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_RVC"

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)
