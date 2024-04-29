import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "RVC.alertMSG",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "RVC_Train") {
			nodeType.prototype.onExecuted = function (data) {
				// alert("Success!you can find weights in:\n" + data.finetune[0] + "\n" + data.finetune[1] + "\n Now you can tts or inference");
				let msg = "Success! you can find weights in:\n" + data.train[0] + "\n you'd like to reboot the server to inference?"
				if (confirm(msg)) {
					try {
						api.fetchApi("/rvc/reboot");
					}
					catch(exception) {
						console.log(exception);
					}
				}
			}
		}
	},
});