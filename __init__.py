import os
import sys,site
from subprocess import Popen
from server import PromptServer
now_dir = os.path.dirname(os.path.abspath(__file__))

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "a") as f:
                f.write(
                    "%s\n%s/rvc\n%s/rvc/infer"
                    % (now_dir,now_dir,now_dir)
                )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/users.pth" % (site_packages_root)):
    print("!!!RVC path was added to " + "%s/users.pth" % (site_packages_root) 
    + "\n if meet `No module` error,try `python main.py` again")

model_path = os.path.join(now_dir,"rvc", "assets")

if not os.path.exists(os.path.join(model_path, "pretrained_v2")):
    cmd = "python %s/download_models.py" % (now_dir)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
else:
    print("!!!RVC use cache models,make sure your 'assets' complete")


WEB_DIRECTORY = "./web"
from .nodes import LoadAudio, PreViewAudio,RVC_Train,RVC_Infer,CombineAudio

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadAudio": LoadAudio,
    "PreViewAudio": PreViewAudio,
    "RVC_Train": RVC_Train,
    "RVC_Infer": RVC_Infer,
    "CombineAudio": CombineAudio
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "AudioLoader",
    "PreViewAudio": "PreView Audio",
    "RVC_Train": "RVC Train",
    "RVC_Infer": "RVC Inference",
    "CombineAudio": "CombineAudio"
}

@PromptServer.instance.routes.get("/rvc/reboot")
def restart(self):
    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    return os.execv(sys.executable, [sys.executable] + sys.argv)